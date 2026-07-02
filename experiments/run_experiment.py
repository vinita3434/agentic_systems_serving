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
import time
from pathlib import Path

import yaml

from harness.agent_loop import DEFAULT_MOCK_TASK, Task, run_episode
from harness.llm_client import MockLLMClient, VLLMClient, preflight_serving_check
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
               task_id: str | None) -> list[Task]:
    if task_source == "mock":
        return [DEFAULT_MOCK_TASK]
    if task_source == "swebench":
        from harness.swebench_tasks import load_swebench_tasks
        ids = [task_id] if task_id else None
        instances = load_swebench_tasks(split=swebench_split,
                                        limit=task_limit if not ids else None,
                                        instance_ids=ids)
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
    """
    if override:
        return override
    port = serving_cfg.get("port", 8000)
    return f"http://localhost:{port}/v1"


def make_llm_client(backend: str, serving_cfg: dict, vllm_base_url: str):
    if backend == "mock":
        return MockLLMClient(model="mock-qwen2.5-coder-7b")
    if backend == "vllm":
        return VLLMClient(base_url=vllm_base_url,
                          model=serving_cfg.get("model",
                                                "Qwen/Qwen2.5-Coder-7B-Instruct"),
                          engine=serving_cfg.get("engine", "vllm"))
    raise ValueError(f"Unknown backend: {backend}")


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
    tasks = load_tasks(args.task_source, args.swebench_split,
                       args.task_limit, args.task_id)

    sweep_id = int(time.time())
    summaries: list[dict] = []

    print(f"Sweep id:      {sweep_id}")
    print(f"Backend:       {args.backend}")
    print(f"Design:        {design_label}")
    print(f"GPU class:     {args.gpu_class}")
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
        for ep, task in enumerate(tasks):
            base_url = serving_base_url(serv_cfg, args.vllm_base_url)
            llm_client = make_llm_client(args.backend, serv_cfg, base_url)
            tools = make_tools(args.task_source, task)
            try:
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
                if ep_result.verified is True:
                    cell_verified += 1
                tj = ep_result.trajectory or {}
                cell_validity_sum += tj.get("action_validity_rate", 0.0)
                cell_error_sum += tj.get("error_rate", 0.0)
                if tj.get("turns_to_submit") is not None:
                    cell_turns_to_submit.append(tj["turns_to_submit"])
            finally:
                close = getattr(tools, "close", None)
                if callable(close):
                    close()

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
        print(f"  traj: validity={(s['avg_action_validity_rate'] or 0):.2f} "
              f"error_rate={(s['avg_error_rate'] or 0):.2f} "
              f"turns_to_submit={s['avg_turns_to_submit']} "
              f"weighted_cache_hit={(s.get('weighted_cache_hit_rate') or 0):.2f}")

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
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--skip-serving-check", action="store_true",
                   help="Skip the startup check that the running engine "
                        "matches the --serving config (vllm backend only).")
    # Cost accounting (per episode JSONL gets gpu_cost_usd + api_equiv_cost_usd).
    p.add_argument("--gpu-hourly-usd", type=float, default=1.40,
                   help="Rental cost of the GPU $/hr. Default 1.40 (A100 SXM, RunPod).")
    p.add_argument("--input-usd-per-mtok", type=float, default=0.15,
                   help="Hosted-API equivalent input token cost in $ per 1M tokens.")
    p.add_argument("--output-usd-per-mtok", type=float, default=0.60,
                   help="Hosted-API equivalent output token cost in $ per 1M tokens.")
    args = p.parse_args()
    asyncio.run(run_sweep(args))


if __name__ == "__main__":
    main()
