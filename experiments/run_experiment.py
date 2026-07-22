"""
Sweep runner.

Two ways to choose which (orchestration, serving) cells to run:

(A) MODULAR — `--layer`, the compartmentalized interface (preferred).
    When `--layer` is given it overrides `--design`. Each layer runs in
    isolation; filter flags select exactly what varies.

      --layer serving         Vary serving; hold orchestration at the L1
                              anchor (serving_sweep_anchor). Filter with
                              --serving NAME [NAME...]; override the held
                              orchestration with a single --orchestration NAME.
      --layer orchestration   Vary orchestration; hold serving at the L2
                              anchor (orchestration_sweep_anchor). Pick a
                              sub-axis with --axis, or specific configs with
                              --orchestration NAME [NAME...]; override the held
                              serving with a single --serving NAME.
      --layer interactions    Run the named H1-H4 hypothesis cells only.
                              Filter with --interaction H1 [H2...].
      --layer custom          Arbitrary cross-product of --serving x
                              --orchestration (each defaults to ALL if omitted).
                              Use this to run any single cell.

    Examples:
      # Only the vllm_lru serving config, at the serving anchor orchestration
      python experiments/run_experiment.py --layer serving --serving vllm_lru
      # Only the context-management orchestration sub-axis
      python experiments/run_experiment.py --layer orchestration --axis context_mgmt
      # One exact cell
      python experiments/run_experiment.py --layer custom \\
          --orchestration summarization --serving vllm_continuum
      # Just hypothesis H1 and H3
      python experiments/run_experiment.py --layer interactions --interaction H1 H3

(B) LEGACY — `--design`, the preset bundles (kept for backward compat):

  grid               every pair  (5x4 = 20 cells)
  ofat               baseline + orchestration main effects + serving main effects
                     (1 + 4 + 3 = 8 cells; the baseline cell is the shared corner)
  ofat+interactions  ofat + hypothesis-driven interaction cells from
                     configs/interactions.yaml (default).

Selects which tasks to run via --task-source:

  mock               canned in-memory JWT bugfix task (default, no GPU/Docker)
  swebench           load instances from princeton-nlp/SWE-bench dataset

Selects which serving configs are compatible via --gpu-class:

  A100  | H100      cells whose serving config lacks the chosen class are skipped.

Backend (where the LLM actually runs):

  mock              MockLLMClient — no network, simulated TTFT
  vllm              VLLMClient against --vllm-base-url

Example invocations:

  # Local dev sanity check
  python experiments/run_experiment.py

  # Real SWE-bench tasks against vLLM on RunPod
  python experiments/run_experiment.py \\
      --backend vllm --vllm-base-url http://localhost:8000/v1 \\
      --task-source swebench --task-limit 5 --gpu-class A100
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import time
from pathlib import Path

import yaml

from harness.agent_loop import DEFAULT_MOCK_TASK, Task, run_episode
from harness.llm_client import (MockLLMClient, VLLMClient, preflight_serving_check,
                                resolve_base_url)
from harness.metrics_logger import MetricsLogger
from harness.tools import MockTools


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


# ---------- config loading -------------------------------------------------


def load_configs(subdir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((CONFIGS_DIR / subdir).glob("*.yaml")):
        with path.open() as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("name", path.stem)
        out[cfg["name"]] = cfg
    return out


def load_interactions() -> dict:
    path = CONFIGS_DIR / "interactions.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def filter_by_gpu_class(serving: dict[str, dict], gpu_class: str) -> dict[str, dict]:
    out = {}
    for name, cfg in serving.items():
        supported = cfg.get("gpu_classes")
        if supported is None or gpu_class in supported:
            out[name] = cfg
        else:
            print(f"  [skip serving='{name}'] not compatible with {gpu_class} "
                  f"(supports {supported})")
    return out


# ---------- experimental design --------------------------------------------


def _require(cfg_map: dict[str, dict], name: str, kind: str) -> dict:
    if name not in cfg_map:
        raise ValueError(f"{kind} config '{name}' not found "
                         f"(may have been filtered by --gpu-class, or the "
                         f"YAML file is missing). Available: {list(cfg_map)}")
    return cfg_map[name]


def select_cells(orch: dict[str, dict], serving: dict[str, dict],
                 interactions_cfg: dict, design: str) -> list[dict]:
    # ---- legacy modes (unchanged) ----
    if design == "grid":
        return [
            {"orchestration": o, "serving": s, "role": "grid"}
            for o, s in itertools.product(orch.values(), serving.values())
        ]

    if design in ("ofat", "ofat+interactions"):
        base_orch_name = interactions_cfg["baseline"]["orchestration"]
        base_serv_name = interactions_cfg["baseline"]["serving"]
        base_orch = _require(orch, base_orch_name, "Baseline orchestration")
        base_serv = _require(serving, base_serv_name, "Baseline serving")
        cells: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def _add(o: dict, s: dict, role: str, **extra):
            key = (o["name"], s["name"])
            if key in seen:
                return
            seen.add(key)
            cells.append({"orchestration": o, "serving": s, "role": role,
                          **extra})

        _add(base_orch, base_serv, "baseline")
        for o in orch.values():
            if o["name"] != base_orch_name:
                _add(o, base_serv, "main_effect_orch")
        for s in serving.values():
            if s["name"] != base_serv_name:
                _add(base_orch, s, "main_effect_serving")

        if design == "ofat+interactions":
            for entry in interactions_cfg.get("interactions") or []:
                o_name = entry["orchestration"]
                s_name = entry["serving"]
                if o_name not in orch or s_name not in serving:
                    continue
                _add(orch[o_name], serving[s_name], "interaction",
                     interaction_name=entry.get("name"),
                     hypothesis=entry.get("hypothesis"))
        return cells

    # ---- L1: serving_sweep ----
    if design == "serving_sweep":
        anchor_name = interactions_cfg["serving_sweep_anchor"]["orchestration"]
        anchor_orch = _require(orch, anchor_name, "serving_sweep_anchor orchestration")
        return [
            {"orchestration": anchor_orch, "serving": s, "role": "serving_sweep"}
            for s in serving.values()
        ]

    # ---- L2: per-sub-axis orchestration sweeps ----
    sub_axis_modes = {
        "context_mgmt_sweep": "context_mgmt",
        "prompt_assembly_sweep": "prompt_assembly",
        "tool_output_sweep": "tool_output",
    }
    if design in sub_axis_modes:
        axis_key = sub_axis_modes[design]
        axes = interactions_cfg.get("orchestration_axes") or {}
        axis_orch_names = axes.get(axis_key)
        if not axis_orch_names:
            raise ValueError(f"orchestration_axes.{axis_key} missing or empty "
                             f"in interactions.yaml")
        anchor_serv_name = interactions_cfg["orchestration_sweep_anchor"]["serving"]
        anchor_serv = _require(serving, anchor_serv_name,
                               "orchestration_sweep_anchor serving")
        return [
            {"orchestration": _require(orch, name, "axis member"),
             "serving": anchor_serv,
             "role": design}
            for name in axis_orch_names
        ]

    # ---- L3: novel interaction cells only ----
    if design == "interactions":
        out: list[dict] = []
        for entry in interactions_cfg.get("interactions") or []:
            o_name = entry["orchestration"]
            s_name = entry["serving"]
            if o_name not in orch or s_name not in serving:
                continue
            out.append({
                "orchestration": orch[o_name],
                "serving": serving[s_name],
                "role": "interaction",
                "interaction_name": entry.get("name"),
                "hypothesis": entry.get("hypothesis"),
                "comparison_baselines": entry.get("comparison_baselines"),
            })
        return out

    raise ValueError(f"Unknown design mode: {design}")


# ---------- modular cell selection (--layer) -------------------------------


def _interaction_matches(name: str | None, wanted: set[str]) -> bool:
    """Match an interaction entry by full name or short token (e.g. 'H1')."""
    if name is None:
        return False
    if name in wanted:
        return True
    token = name.split("_", 1)[0]  # 'H1_summarization_...' -> 'H1'
    return token in wanted


def select_cells_modular(orch: dict[str, dict], serving: dict[str, dict],
                         interactions_cfg: dict, args) -> list[dict]:
    """Compartmentalized selection. Each --layer runs in isolation.

    Filter flags:
      --serving / --orchestration : restrict which configs participate.
          In a sweep layer the flag for the *varying* axis filters the
          sweep; a single value for the *held* axis overrides its anchor.
      --axis        : in the orchestration layer, pick one sub-axis.
      --interaction : in the interactions layer, pick named hypotheses.
    """
    layer = args.layer
    serving_filter = args.serving            # list[str] | None
    orch_filter = args.orchestration         # list[str] | None

    if layer == "serving":
        anchor_name = (orch_filter[0] if orch_filter
                       else interactions_cfg["serving_sweep_anchor"]["orchestration"])
        anchor_orch = _require(orch, anchor_name, "serving-layer held orchestration")
        serv_names = serving_filter if serving_filter else list(serving)
        return [
            {"orchestration": anchor_orch,
             "serving": _require(serving, sn, "serving"),
             "role": "serving_sweep"}
            for sn in serv_names
        ]

    if layer == "orchestration":
        anchor_name = (serving_filter[0] if serving_filter
                       else interactions_cfg["orchestration_sweep_anchor"]["serving"])
        anchor_serv = _require(serving, anchor_name, "orchestration-layer held serving")
        if orch_filter:
            orch_names = orch_filter
            role = "orchestration_sweep"
        elif args.axis:
            axes = interactions_cfg.get("orchestration_axes") or {}
            orch_names = axes.get(args.axis)
            if not orch_names:
                raise ValueError(f"orchestration_axes.{args.axis} missing or empty "
                                 f"in interactions.yaml")
            role = f"{args.axis}_sweep"
        else:
            orch_names = list(orch)          # full orchestration layer
            role = "orchestration_sweep"
        return [
            {"orchestration": _require(orch, on, "orchestration"),
             "serving": anchor_serv,
             "role": role}
            for on in orch_names
        ]

    if layer == "interactions":
        wanted = set(args.interaction) if args.interaction else None
        out: list[dict] = []
        for entry in interactions_cfg.get("interactions") or []:
            name = entry.get("name")
            if wanted is not None and not _interaction_matches(name, wanted):
                continue
            o_name, s_name = entry["orchestration"], entry["serving"]
            if o_name not in orch or s_name not in serving:
                continue
            out.append({
                "orchestration": orch[o_name],
                "serving": serving[s_name],
                "role": "interaction",
                "interaction_name": name,
                "hypothesis": entry.get("hypothesis"),
                "comparison_baselines": entry.get("comparison_baselines"),
            })
        if wanted is not None and not out:
            raise ValueError(f"No interactions matched {sorted(wanted)}. "
                             f"Available: "
                             f"{[e.get('name') for e in interactions_cfg.get('interactions') or []]}")
        return out

    if layer == "custom":
        orch_names = orch_filter if orch_filter else list(orch)
        serv_names = serving_filter if serving_filter else list(serving)
        return [
            {"orchestration": _require(orch, on, "orchestration"),
             "serving": _require(serving, sn, "serving"),
             "role": "custom"}
            for on in orch_names
            for sn in serv_names
        ]

    raise ValueError(f"Unknown layer: {layer}")


# ---------- task selection -------------------------------------------------


def load_tasks(task_source: str, swebench_split: str, task_limit: int,
               task_id: str | None, hard_only: bool = False) -> list[Task]:
    if task_source == "mock":
        return [DEFAULT_MOCK_TASK]
    if task_source == "swebench":
        from harness.swebench_tasks import load_swebench_tasks
        ids = [task_id] if task_id else None
        instances = load_swebench_tasks(split=swebench_split,
                                        limit=task_limit if not ids else None,
                                        instance_ids=ids,
                                        hard_only=hard_only)
        return [i.as_task() for i in instances]
    raise ValueError(f"Unknown task source: {task_source}")


def make_tools(task_source: str, task: Task):
    """Build the Tools object for one episode."""
    if task_source == "mock":
        return MockTools()
    if task_source == "swebench":
        from harness.sweenv_tools import SWEEnvTools
        from harness.swebench_tasks import load_swebench_tasks
        # Re-fetch the SWEBenchTask wrapping the underlying row.
        # Cheaper than passing it through Task — keeps Task lean.
        [inst] = load_swebench_tasks(split="full", instance_ids=[task.task_id])
        tools = SWEEnvTools.from_swebench(inst)
        tools.start()
        return tools
    raise ValueError(f"Unknown task source: {task_source}")


# ---------- LLM client factory ---------------------------------------------


def serving_base_url(serving_cfg: dict, override: str | None) -> str:
    """Where the harness should send requests for this serving config.

    Default: derive from the config's own `port` (each engine binds a
    distinct port, so picking --serving automatically targets the right
    server). `--vllm-base-url` overrides this when explicitly passed.

    Note: we always connect to localhost — the config's `host` (0.0.0.0)
    is the server's *bind* address, not a client connect address.

    An explicit --vllm-base-url override or the LLM_BASE_URL env var takes
    precedence over the per-config port (single-box: one server on :8000).
    """
    if override or os.environ.get("LLM_BASE_URL"):
        return resolve_base_url(override)
    port = serving_cfg.get("port", 8000)
    return f"http://localhost:{port}/v1"


def make_llm_client(backend: str, serving_cfg: dict, vllm_base_url: str):
    if backend == "mock":
        return MockLLMClient(model="mock-qwen2.5-coder-7b")
    if backend == "vllm":
        return VLLMClient(base_url=vllm_base_url,
                          model=serving_cfg.get("model",
                                                "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"),
                          engine=serving_cfg.get("engine", "vllm"))
    raise ValueError(f"Unknown backend: {backend}")


# ---------- resumability ---------------------------------------------------


def completed_episode_keys(results_dir: Path) -> set[tuple[str, str, str]]:
    """Scan prior per-episode logs and return the set of already-finished
    (task_id, orchestration, serving) triples.

    A per-episode row is written only after run_episode returns without
    crashing, so a row's presence means that episode is DONE. This lets a
    sweep resume: a crash at task 15/20 re-runs only tasks 15-20, never the
    completed ones — zero repeated GPU work on restart. Keyed on the triple
    (not experiment_id) because each restart mints a fresh time-based sweep_id.
    """
    done: set[tuple[str, str, str]] = set()
    for path in results_dir.glob("*__episodes.jsonl"):
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tid, orch, serv = (row.get("task_id"),
                                       row.get("orchestration"),
                                       row.get("serving"))
                    if tid and orch and serv:
                        done.add((tid, orch, serv))
        except OSError:
            continue
    return done


# ---------- sweep ----------------------------------------------------------


async def run_sweep(args) -> list[dict]:
    orch_configs = load_configs("orchestration")
    serving_configs = filter_by_gpu_class(load_configs("serving"), args.gpu_class)
    interactions_cfg = load_interactions()

    if args.layer:
        cells = select_cells_modular(orch_configs, serving_configs,
                                     interactions_cfg, args)
        design_label = f"layer:{args.layer}"
    else:
        cells = select_cells(orch_configs, serving_configs, interactions_cfg, args.design)
        design_label = args.design
    # Hard tasks need room for long trajectories; make sure max_turns isn't
    # the thing that ends the episode. We do NOT force a minimum turn count —
    # the agent still submits when it's done; the difficulty filter is what
    # produces long episodes.
    if args.hard_only and args.max_turns < 40:
        print(f"[--hard-only] raising max_turns {args.max_turns} -> 60 so long "
              f"episodes aren't truncated.")
        args.max_turns = 60

    tasks = load_tasks(args.task_source, args.swebench_split,
                       args.task_limit, args.task_id, hard_only=args.hard_only)

    # Resumability: skip (task, orch, serving) triples already completed in a
    # prior sweep. Disabled by --no-resume and during --calibrate (which needs
    # fresh timings). See completed_episode_keys().
    resume_enabled = not args.no_resume and not args.calibrate
    done_keys = completed_episode_keys(RESULTS_DIR) if resume_enabled else set()

    # Calibration: remember the full task count for projection, then run only
    # the first N tasks. Projection scales by (full tasks x number of cells).
    full_task_n = len(tasks)
    if args.calibrate:
        tasks = tasks[:args.calibrate]

    sweep_id = int(time.time())
    summaries: list[dict] = []
    calib_samples: list[tuple[float, float]] = []  # (wall_s, gpu_cost_usd) per episode

    print(f"Sweep id:      {sweep_id}")
    print(f"Backend:       {args.backend}")
    print(f"Design:        {design_label}")
    print(f"GPU class:     {args.gpu_class}")
    print(f"Resume:        {'on' if resume_enabled else 'off'} "
          f"({len(done_keys)} completed episodes on disk will be skipped)")
    print(f"Task source:   {args.task_source} (n={len(tasks)})")
    print(f"Orchestrations: {list(orch_configs)}")
    print(f"Servings:       {list(serving_configs)}")
    print(f"Cells:          {len(cells)} "
          f"(grid would be {len(orch_configs) * len(serving_configs)})")
    print()

    # Preflight: confirm the running engine matches each serving config we're
    # about to hit. Only meaningful against a real server (--backend vllm).
    if args.backend == "vllm" and not args.skip_serving_check:
        seen_serv: dict[str, dict] = {}
        for cell in cells:
            seen_serv.setdefault(cell["serving"]["name"], cell["serving"])
        for name, serv_cfg in seen_serv.items():
            base_url = serving_base_url(serv_cfg, args.vllm_base_url)
            warns = await preflight_serving_check(base_url, serv_cfg)
            if warns:
                print(f"  [serving check: {name} @ {base_url}]")
                for w in warns:
                    print(f"    ⚠  {w}")
            else:
                print(f"  [serving check: {name} @ {base_url}] ok")
        print()

    for cell in cells:
        orch_cfg = cell["orchestration"]
        serv_cfg = cell["serving"]
        role = cell["role"]
        tag = cell.get("interaction_name") or role
        exp_id = (f"sweep{sweep_id}__{tag}__{orch_cfg['name']}__{serv_cfg['name']}")
        print(f"[{role:<22}] orch={orch_cfg['name']:<26} "
              f"serving={serv_cfg['name']:<22}")

        logger = MetricsLogger(RESULTS_DIR, exp_id)
        cell_wall_clock_s = 0.0
        cell_prompt_tokens = 0
        cell_completion_tokens = 0
        cell_episodes = 0
        cell_verified = 0
        cell_validity_sum = 0.0
        cell_error_sum = 0.0
        cell_turns_to_submit: list[int] = []
        cell_recovery_rates: list[float] = []
        for ep, task in enumerate(tasks):
            if resume_enabled and (task.task_id, orch_cfg["name"],
                                   serv_cfg["name"]) in done_keys:
                print(f"  [skip ep={ep:<2} task={task.task_id}] already "
                      f"completed on disk — resume")
                continue
            base_url = serving_base_url(serv_cfg, args.vllm_base_url)
            llm_client = make_llm_client(args.backend, serv_cfg, base_url)
            tools = None
            try:
                tools = make_tools(args.task_source, task)
                ep_result = await run_episode(
                    task=task,
                    orchestration_cfg=orch_cfg,
                    serving_cfg=serv_cfg,
                    llm_client=llm_client,
                    tools=tools,
                    metrics_logger=logger,
                    episode_idx=ep,
                    max_turns=args.max_turns,
                )
                logger.log_episode(
                    task_id=ep_result.task_id,
                    episode_idx=ep,
                    orchestration=orch_cfg["name"],
                    serving=serv_cfg["name"],
                    n_turns=ep_result.n_turns,
                    completed=ep_result.completed,
                    final_history_tokens=ep_result.final_history_tokens,
                    verified=ep_result.verified,
                    total_prompt_tokens=ep_result.total_prompt_tokens,
                    total_completion_tokens=ep_result.total_completion_tokens,
                    wall_clock_s=ep_result.wall_clock_s,
                    gpu_hourly_usd=args.gpu_hourly_usd,
                    input_usd_per_mtok=args.input_usd_per_mtok,
                    output_usd_per_mtok=args.output_usd_per_mtok,
                    trajectory=ep_result.trajectory,
                )
                cell_wall_clock_s += ep_result.wall_clock_s
                cell_prompt_tokens += ep_result.total_prompt_tokens
                cell_completion_tokens += ep_result.total_completion_tokens
                cell_episodes += 1
                calib_samples.append(
                    (ep_result.wall_clock_s,
                     ep_result.wall_clock_s * (args.gpu_hourly_usd / 3600.0)))
                if ep_result.verified is True:
                    cell_verified += 1
                tj = ep_result.trajectory or {}
                cell_validity_sum += tj.get("action_validity_rate", 0.0)
                cell_error_sum += tj.get("error_rate", 0.0)
                if tj.get("turns_to_submit") is not None:
                    cell_turns_to_submit.append(tj["turns_to_submit"])
                if tj.get("error_recovery_rate") is not None:
                    cell_recovery_rates.append(tj["error_recovery_rate"])
            except Exception as e:
                # Isolate per-task failures (Docker pull/exec errors, tool
                # setup, unexpected model output, etc.) so ONE bad task can't
                # abort the whole sweep. It's skipped (no episode row), and
                # resume will retry it on the next run.
                print(f"  [task {task.task_id} FAILED: {type(e).__name__}: {e} "
                      f"— skipping; sweep continues]")
            finally:
                close = getattr(tools, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        if cell_episodes == 0:
            # Every episode was skipped (resume) or none produced turns; there
            # is nothing new to summarize. The prior run's rows are already on
            # disk, so just move on rather than emit an empty/broken summary.
            print("  all episodes already complete for this cell — skipped")
            continue

        s = logger.summary()
        s["orchestration"] = orch_cfg["name"]
        s["serving"] = serv_cfg["name"]
        s["role"] = role
        if "interaction_name" in cell:
            s["interaction_name"] = cell["interaction_name"]
        if "hypothesis" in cell:
            s["hypothesis"] = cell["hypothesis"]
        # Per-cell cost rollup.
        gpu_cost = cell_wall_clock_s * (args.gpu_hourly_usd / 3600.0)
        api_equiv_cost = (
            cell_prompt_tokens * args.input_usd_per_mtok +
            cell_completion_tokens * args.output_usd_per_mtok
        ) / 1_000_000.0
        s["episodes"] = cell_episodes
        s["verified_count"] = cell_verified
        s["task_success_rate"] = (cell_verified / cell_episodes) if cell_episodes else None
        s["cell_wall_clock_s"] = cell_wall_clock_s
        s["cell_total_prompt_tokens"] = cell_prompt_tokens
        s["cell_total_completion_tokens"] = cell_completion_tokens
        s["cell_gpu_cost_usd"] = gpu_cost
        s["cell_api_equiv_cost_usd"] = api_equiv_cost
        s["cost_per_episode_usd"] = (gpu_cost / cell_episodes) if cell_episodes else None
        s["cost_per_verified_task_usd"] = (
            (gpu_cost / cell_verified) if cell_verified > 0 else None)
        # Trajectory-quality rollup (averaged over the cell's episodes).
        s["avg_action_validity_rate"] = (
            cell_validity_sum / cell_episodes) if cell_episodes else None
        s["avg_error_rate"] = (
            cell_error_sum / cell_episodes) if cell_episodes else None
        s["avg_turns_to_submit"] = (
            sum(cell_turns_to_submit) / len(cell_turns_to_submit)
            if cell_turns_to_submit else None)
        s["avg_error_recovery_rate"] = (
            sum(cell_recovery_rates) / len(cell_recovery_rates)
            if cell_recovery_rates else None)
        summaries.append(s)
        print(f"  turns={s['n_turns']:3d} "
              f"avg_ttft={s['avg_ttft_ms']:7.1f}ms "
              f"p95_ttft={s['p95_ttft_ms']:7.1f}ms "
              f"avg_ctx_toks={s['avg_context_tokens']:6.0f} "
              f"compression={s['compression_ratio']:.2f} "
              f"cache_hit={(s.get('avg_cache_hit_rate') or 0):.2f}")
        print(f"  episodes={cell_episodes} verified={cell_verified} "
              f"wall={cell_wall_clock_s:.1f}s "
              f"in_toks={cell_prompt_tokens} out_toks={cell_completion_tokens} "
              f"gpu_cost=${gpu_cost:.4f} api_equiv=${api_equiv_cost:.4f} "
              f"cost/task=${(gpu_cost/cell_episodes if cell_episodes else 0):.4f}")
        _eff = s.get("weighted_cache_efficiency")
        _rec = s.get("avg_error_recovery_rate")
        print(f"  traj: validity={(s['avg_action_validity_rate'] or 0):.2f} "
              f"error_rate={(s['avg_error_rate'] or 0):.2f} "
              f"recovery={('n/a' if _rec is None else f'{_rec:.2f}')} "
              f"turns_to_submit={s['avg_turns_to_submit']}")
        print(f"  cache: weighted_hit={(s.get('weighted_cache_hit_rate') or 0):.2f} "
              f"efficiency={('n/a' if _eff is None else f'{_eff:.2f}')} "
              f"(hit=used/sent, efficiency=used/reusable)")

    if args.calibrate:
        _print_calibration(calib_samples, full_task_n, len(cells), args)
        return summaries

    summary_path = RESULTS_DIR / f"sweep{sweep_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump({
            "sweep_id": sweep_id,
            "design": design_label,
            "backend": args.backend,
            "gpu_class": args.gpu_class,
            "task_source": args.task_source,
            "n_tasks": len(tasks),
            "cells": summaries,
        }, f, indent=2)

    print(f"\nSweep summary written to: {summary_path}")
    _print_table(summaries)
    return summaries


# ---------- output ---------------------------------------------------------


def _print_calibration(samples: list[tuple[float, float]], full_task_n: int,
                       n_cells: int, args) -> None:
    """Report timing/cost from the calibration episodes and project the full
    sweep. Projection assumes strictly sequential episodes (one at a time, per
    the hard constraint) and that the full sweep starts now."""
    import statistics
    from datetime import datetime

    print("\n" + "=" * 64)
    print(f"CALIBRATION  ({len(samples)} episode(s) over "
          f"{min(args.calibrate, full_task_n)} task(s) x {n_cells} cell(s))")
    print("=" * 64)
    if not samples:
        print("  no episodes completed — cannot project. Check the serving "
              "server and the smoke task first.")
        print("=" * 64)
        return

    walls = [w for w, _ in samples]
    costs = [c for _, c in samples]
    mean_wall = statistics.mean(walls)
    med_wall = statistics.median(walls)
    mean_cost = statistics.mean(costs)

    total_episodes = full_task_n * n_cells
    proj_secs = mean_wall * total_episodes
    proj_cost = mean_cost * total_episodes
    finish = datetime.fromtimestamp(time.time() + proj_secs)

    print(f"  mean   episode wall : {mean_wall:8.1f} s   ({mean_wall/60:5.1f} min)")
    print(f"  median episode wall : {med_wall:8.1f} s   ({med_wall/60:5.1f} min)")
    print(f"  mean   cost/episode : ${mean_cost:.4f}")
    print(f"  gpu rate            : ${args.gpu_hourly_usd:.2f}/hr")
    print("-" * 64)
    print(f"  FULL SWEEP projection: {full_task_n} task(s) x {n_cells} "
          f"cell(s) = {total_episodes} episodes")
    print(f"  projected wall time : {proj_secs/3600:6.2f} h   "
          f"({proj_secs/60:.0f} min)")
    print(f"  projected total cost: ${proj_cost:.2f}")
    print(f"  projected finish    : {finish:%Y-%m-%d %H:%M:%S}  "
          f"(if the full sweep starts now, sequential)")
    print("=" * 64)


_ROLE_ORDER = {
    "baseline": 0,
    "serving_sweep": 1,
    "context_mgmt_sweep": 2,
    "prompt_assembly_sweep": 3,
    "tool_output_sweep": 4,
    "orchestration_sweep": 5,
    "main_effect_orch": 6,
    "main_effect_serving": 7,
    "interaction": 8,
    "custom": 9,
    "grid": 10,
}


def _print_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 122)
    print(f"{'role':<22} {'orchestration':<26} {'serving':<22} "
          f"{'avg_ttft':>10} {'p95_ttft':>10} {'ctx_toks':>10} "
          f"{'compr':>7} {'cache_hit':>10}")
    print("-" * 122)
    for s in sorted(summaries, key=lambda x: (_ROLE_ORDER.get(x["role"], 9),
                                              x["orchestration"], x["serving"])):
        cache = s.get("avg_cache_hit_rate") or 0.0
        print(f"{s['role']:<22} {s['orchestration']:<26} {s['serving']:<22} "
              f"{s['avg_ttft_ms']:>9.1f}ms {s['p95_ttft_ms']:>9.1f}ms "
              f"{s['avg_context_tokens']:>10.0f} "
              f"{s['compression_ratio']:>7.2f} {cache:>10.2f}")
    print("=" * 122)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    p.add_argument("--vllm-base-url", default=None,
                   help="Override the server URL. Default: derived from the "
                        "serving config's port (http://localhost:<port>/v1).")
    p.add_argument("--design",
                   choices=["grid", "ofat", "ofat+interactions",
                            "serving_sweep",
                            "context_mgmt_sweep",
                            "prompt_assembly_sweep",
                            "tool_output_sweep",
                            "interactions"],
                   default="ofat+interactions",
                   help="Legacy preset bundles. Ignored when --layer is set.")
    # Modular compartmentalized interface. When set, overrides --design.
    p.add_argument("--layer",
                   choices=["serving", "orchestration", "interactions", "custom"],
                   default=None,
                   help="Run one compartment in isolation. Overrides --design.")
    p.add_argument("--serving", nargs="+", default=None,
                   help="Restrict serving config(s) by name. In --layer serving "
                        "these are the cells swept; elsewhere a single name "
                        "overrides the held serving anchor.")
    p.add_argument("--orchestration", nargs="+", default=None,
                   help="Restrict orchestration config(s) by name. In --layer "
                        "orchestration these are the cells swept; elsewhere a "
                        "single name overrides the held orchestration anchor.")
    p.add_argument("--axis",
                   choices=["context_mgmt", "prompt_assembly", "tool_output"],
                   default=None,
                   help="In --layer orchestration, sweep just this sub-axis.")
    p.add_argument("--interaction", nargs="+", default=None,
                   help="In --layer interactions, run only these hypotheses "
                        "(full name or short token like H1).")
    p.add_argument("--gpu-class", choices=["A100", "H100"], default="A100",
                   help="Filter serving configs incompatible with this GPU class.")
    p.add_argument("--task-source", choices=["mock", "swebench"], default="mock")
    p.add_argument("--swebench-split", choices=["full", "lite", "verified"],
                   default="full")
    p.add_argument("--task-limit", type=int, default=5,
                   help="Number of SWE-bench instances per cell.")
    p.add_argument("--task-id", default=None,
                   help="Single SWE-bench instance_id; overrides --task-limit.")
    p.add_argument("--hard-only", action="store_true",
                   help="Keep only hard-difficulty tasks (1-4 hrs / >4 hrs). "
                        "Requires --swebench-split verified. Long trajectories "
                        "make cache/serving metrics more informative. Auto-"
                        "raises --max-turns to 60 if lower.")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--skip-serving-check", action="store_true",
                   help="Skip the startup check that the running engine "
                        "matches the --serving config (vllm backend only).")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable resume. By default the sweep skips any "
                        "(task, orchestration, serving) episode already present "
                        "in experiments/results/*__episodes.jsonl, so a crash "
                        "mid-sweep costs zero repeated episodes on restart.")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="Calibration mode: run only the first N tasks, print "
                        "timing/cost projections for the full task list, then "
                        "exit. Implies --no-resume (needs fresh timings).")
    p.add_argument("--shutdown-on-complete", action="store_true",
                   help="After the sweep finishes, run 'sudo shutdown -h +5' so "
                        "an overnight run doesn't idle-bill the GPU box. Default "
                        "off. Cancel within the 5 min with 'sudo shutdown -c'. "
                        "Ignored under --calibrate.")
    # Cost accounting (per episode JSONL gets gpu_cost_usd + api_equiv_cost_usd).
    p.add_argument("--gpu-hourly-usd", type=float, default=1.40,
                   help="Rental cost of the GPU $/hr. Default 1.40 (A100 SXM, RunPod).")
    p.add_argument("--input-usd-per-mtok", type=float, default=0.15,
                   help="Hosted-API equivalent input token cost in $ per 1M tokens.")
    p.add_argument("--output-usd-per-mtok", type=float, default=0.60,
                   help="Hosted-API equivalent output token cost in $ per 1M tokens.")
    args = p.parse_args()
    asyncio.run(run_sweep(args))

    # Optional auto-stop for unattended overnight runs. Only after a real sweep
    # (never during --calibrate). shutdown() itself needs sudo; on a non-Linux
    # box (e.g. a laptop dry-run) the binary/permission may be absent — warn,
    # don't crash, so the results are never lost to a shutdown hiccup.
    if args.shutdown_on_complete and not args.calibrate:
        import subprocess
        print("\n[--shutdown-on-complete] sweep finished; scheduling "
              "'sudo shutdown -h +5'.")
        print("  Cancel within 5 minutes with:  sudo shutdown -c")
        try:
            subprocess.run(["sudo", "shutdown", "-h", "+5"], check=False)
        except FileNotFoundError:
            print("  NOTE: 'shutdown' not found — skipping (not a Linux host?).")


if __name__ == "__main__":
    main()
