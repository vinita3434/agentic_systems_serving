"""
Sweep runner.

Selects which (orchestration, serving) cells to run via --design:

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
from harness.llm_client import MockLLMClient, VLLMClient
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


def make_llm_client(backend: str, serving_cfg: dict, vllm_base_url: str):
    if backend == "mock":
        return MockLLMClient(model="mock-qwen2.5-coder-7b")
    if backend == "vllm":
        return VLLMClient(base_url=vllm_base_url,
                          model=serving_cfg.get("model",
                                                "Qwen/Qwen2.5-Coder-7B-Instruct"))
    raise ValueError(f"Unknown backend: {backend}")


# ---------- sweep ----------------------------------------------------------


async def run_sweep(args) -> list[dict]:
    orch_configs = load_configs("orchestration")
    serving_configs = filter_by_gpu_class(load_configs("serving"), args.gpu_class)
    interactions_cfg = load_interactions()

    cells = select_cells(orch_configs, serving_configs, interactions_cfg, args.design)
    tasks = load_tasks(args.task_source, args.swebench_split,
                       args.task_limit, args.task_id)

    sweep_id = int(time.time())
    summaries: list[dict] = []

    print(f"Sweep id:      {sweep_id}")
    print(f"Backend:       {args.backend}")
    print(f"Design:        {args.design}")
    print(f"GPU class:     {args.gpu_class}")
    print(f"Task source:   {args.task_source} (n={len(tasks)})")
    print(f"Orchestrations: {list(orch_configs)}")
    print(f"Servings:       {list(serving_configs)}")
    print(f"Cells:          {len(cells)} "
          f"(grid would be {len(orch_configs) * len(serving_configs)})")
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
        for ep, task in enumerate(tasks):
            llm_client = make_llm_client(args.backend, serv_cfg, args.vllm_base_url)
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
                )
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
        summaries.append(s)
        print(f"  turns={s['n_turns']:3d} "
              f"avg_ttft={s['avg_ttft_ms']:7.1f}ms "
              f"p95_ttft={s['p95_ttft_ms']:7.1f}ms "
              f"avg_ctx_toks={s['avg_context_tokens']:6.0f} "
              f"compression={s['compression_ratio']:.2f} "
              f"cache_hit={(s.get('avg_cache_hit_rate') or 0):.2f}")

    summary_path = RESULTS_DIR / f"sweep{sweep_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump({
            "sweep_id": sweep_id,
            "design": args.design,
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
    "main_effect_orch": 5,
    "main_effect_serving": 6,
    "interaction": 7,
    "grid": 8,
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
    p.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    p.add_argument("--design",
                   choices=["grid", "ofat", "ofat+interactions",
                            "serving_sweep",
                            "context_mgmt_sweep",
                            "prompt_assembly_sweep",
                            "tool_output_sweep",
                            "interactions"],
                   default="ofat+interactions")
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
    args = p.parse_args()
    asyncio.run(run_sweep(args))


if __name__ == "__main__":
    main()
