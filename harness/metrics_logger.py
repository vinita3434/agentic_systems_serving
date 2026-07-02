"""
Per-turn metrics logging for ablation experiments.

Writes one JSONL line per LLM call. Each experiment (orchestration x serving
combination) gets its own file. The sweep runner later aggregates these into
a summary CSV/JSON.

Metrics captured:
  - ttft_ms                : time to first token (prefill latency proxy)
  - total_latency_ms       : full request latency
  - prompt_tokens          : tokens actually sent to the model
  - completion_tokens      : tokens generated
  - context_tokens         : same as prompt_tokens; named separately so the
                             orchestration strategy's effect is explicit
  - raw_history_tokens     : tokens BEFORE the orchestration strategy ran
                             (lets you measure how much each strategy saves)
  - cache_hit_rate         : prefix cache hit ratio (vLLM /metrics); None in mock
  - cache_hit_tokens       : tokens served from cache (vLLM /metrics)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TurnMetrics:
    experiment_id: str
    orchestration: str
    serving: str
    episode: int
    turn: int
    prompt_tokens: int
    completion_tokens: int
    context_tokens: int
    raw_history_tokens: int
    ttft_ms: float
    total_latency_ms: float
    cache_hit_rate: Optional[float] = None
    cache_hit_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MetricsLogger:
    def __init__(self, results_dir: str | Path, experiment_id: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.log_path = self.results_dir / f"{experiment_id}.jsonl"
        self.episode_log_path = self.results_dir / f"{experiment_id}__episodes.jsonl"
        self._turns: list[TurnMetrics] = []

    def log(self, metrics: TurnMetrics) -> None:
        self._turns.append(metrics)
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

    def log_episode(self, *, task_id: str, episode_idx: int,
                    orchestration: str, serving: str,
                    n_turns: int, completed: bool,
                    final_history_tokens: int,
                    verified: Optional[bool],
                    total_prompt_tokens: int = 0,
                    total_completion_tokens: int = 0,
                    wall_clock_s: float = 0.0,
                    gpu_hourly_usd: float = 1.40,
                    input_usd_per_mtok: float = 0.15,
                    output_usd_per_mtok: float = 0.60,
                    trajectory: Optional[dict] = None) -> None:
        """Per-episode summary row.

        Cost accounting per episode:
          - gpu_cost_usd       = wall_clock_s * (gpu_hourly_usd / 3600)
                                 The actual rental cost for this episode on
                                 the configured GPU. Self-hosted ground truth.
          - api_equiv_cost_usd = (in_tok * in_rate + out_tok * out_rate) / 1e6
                                 What the same token volume would cost on a
                                 hosted API at the configured per-token rates.
                                 Useful for comparing self-host vs hosted.
          - total_cost_usd     = gpu_cost_usd  (headline; default GPU-based)

        Defaults assume A100 80GB PCIe (RunPod community ~$1.40/hr) and
        Qwen-class hosted-token rates. Override via run_experiment.py
        CLI flags --gpu-hourly-usd / --input-usd-per-mtok / --output-usd-per-mtok.
        """
        gpu_cost_usd = wall_clock_s * (gpu_hourly_usd / 3600.0)
        api_equiv_cost_usd = (
            total_prompt_tokens * input_usd_per_mtok +
            total_completion_tokens * output_usd_per_mtok
        ) / 1_000_000.0

        row = {
            "experiment_id": self.experiment_id,
            "orchestration": orchestration,
            "serving": serving,
            "episode": episode_idx,
            "task_id": task_id,
            "n_turns": n_turns,
            "completed": completed,
            "final_history_tokens": final_history_tokens,
            "verified": verified,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "wall_clock_s": wall_clock_s,
            "gpu_hourly_usd": gpu_hourly_usd,
            "input_usd_per_mtok": input_usd_per_mtok,
            "output_usd_per_mtok": output_usd_per_mtok,
            "gpu_cost_usd": gpu_cost_usd,
            "api_equiv_cost_usd": api_equiv_cost_usd,
            "total_cost_usd": gpu_cost_usd,
            "timestamp": time.time(),
        }
        # Trajectory-quality metrics + within-episode cache-hit curve.
        if trajectory:
            row.update(trajectory)
        with self.episode_log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def summary(self) -> dict:
        if not self._turns:
            return {"experiment_id": self.experiment_id, "n_turns": 0}

        def avg(key: str) -> float:
            vals = [getattr(t, key) for t in self._turns if getattr(t, key) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        def total(key: str) -> int:
            return sum(getattr(t, key) or 0 for t in self._turns)

        return {
            "experiment_id": self.experiment_id,
            "orchestration": self._turns[0].orchestration,
            "serving": self._turns[0].serving,
            "n_turns": len(self._turns),
            "avg_ttft_ms": avg("ttft_ms"),
            "p50_ttft_ms": _percentile([t.ttft_ms for t in self._turns], 50),
            "p95_ttft_ms": _percentile([t.ttft_ms for t in self._turns], 95),
            "avg_total_latency_ms": avg("total_latency_ms"),
            "avg_prompt_tokens": avg("prompt_tokens"),
            "avg_context_tokens": avg("context_tokens"),
            "avg_raw_history_tokens": avg("raw_history_tokens"),
            "compression_ratio": (
                avg("context_tokens") / avg("raw_history_tokens")
                if avg("raw_history_tokens") > 0
                else 1.0
            ),
            "total_prompt_tokens": total("prompt_tokens"),
            "total_completion_tokens": total("completion_tokens"),
            "avg_cache_hit_rate": avg("cache_hit_rate"),
            # Token-weighted hit rate: served-from-cache tokens over all prompt
            # tokens. Truer than avg_cache_hit_rate, which weights every turn
            # equally regardless of prompt size.
            "total_cache_hit_tokens": total("cache_hit_tokens"),
            "weighted_cache_hit_rate": (
                total("cache_hit_tokens") / total("prompt_tokens")
                if total("prompt_tokens") > 0 else 0.0
            ),
        }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
