"""Consolidate every sweep's *_summary.json into one CSV.

One row per cell (orchestration x serving x sweep). Flags cache metrics that
are physically impossible (weighted_cache_hit_rate > ~1) as SUSPECT — those
come from runs where more than one harness hit the same vLLM server, so the
server-global /metrics counter double-counted (see cache_note column).

Usage:  python experiments/consolidate_metrics.py
Output: experiments/results/consolidated_metrics.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RESULTS, "consolidated_metrics.csv")

# Cache hit-rate above this is impossible for a true rate; a small margin over
# 1.0 is allowed for vLLM's 16-token block-granularity slack.
CACHE_SANE_MAX = 1.05

COLUMNS = [
    "sweep_id", "orchestration", "serving", "n_turns", "episodes",
    "verified_count", "task_success_rate",
    "avg_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "avg_total_latency_ms",
    "avg_context_tokens", "compression_ratio",
    "weighted_cache_hit_rate", "weighted_cache_efficiency", "avg_cache_hit_rate",
    "avg_action_validity_rate", "avg_error_rate", "avg_error_recovery_rate",
    "avg_turns_to_submit", "cost_per_episode_usd",
    "cache_note",
]


def rows():
    for path in sorted(glob.glob(os.path.join(RESULTS, "*_summary.json"))):
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        for c in data.get("cells", []):
            whr = c.get("weighted_cache_hit_rate")
            note = ""
            if whr is not None and whr > CACHE_SANE_MAX:
                note = f"SUSPECT: hit_rate={whr:.2f}>1 -> multiple harnesses shared one server (global /metrics double-count); cache cols invalid"
            row = {
                "sweep_id": data.get("sweep_id"),
                "orchestration": c.get("orchestration"),
                "serving": c.get("serving"),
                "n_turns": c.get("n_turns"),
                "episodes": c.get("episodes"),
                "verified_count": c.get("verified_count"),
                "task_success_rate": c.get("task_success_rate"),
                "avg_ttft_ms": _r(c.get("avg_ttft_ms")),
                "p50_ttft_ms": _r(c.get("p50_ttft_ms")),
                "p95_ttft_ms": _r(c.get("p95_ttft_ms")),
                "avg_total_latency_ms": _r(c.get("avg_total_latency_ms")),
                "avg_context_tokens": _r(c.get("avg_context_tokens")),
                "compression_ratio": _r(c.get("compression_ratio"), 3),
                "weighted_cache_hit_rate": _r(c.get("weighted_cache_hit_rate"), 3),
                "weighted_cache_efficiency": _r(c.get("weighted_cache_efficiency"), 3),
                "avg_cache_hit_rate": _r(c.get("avg_cache_hit_rate"), 3),
                "avg_action_validity_rate": _r(c.get("avg_action_validity_rate"), 3),
                "avg_error_rate": _r(c.get("avg_error_rate"), 3),
                "avg_error_recovery_rate": _r(c.get("avg_error_recovery_rate"), 3),
                "avg_turns_to_submit": c.get("avg_turns_to_submit"),
                "cost_per_episode_usd": _r(c.get("cost_per_episode_usd"), 4),
                "cache_note": note,
            }
            yield row


def _r(v, n=1):
    return round(v, n) if isinstance(v, (int, float)) else v


def main():
    data = list(rows())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(data)
    print(f"Wrote {len(data)} cells -> {OUT}")
    # Quick console view of the columns most asked about.
    print(f"\n{'sweep':>12} {'orch':<22} {'serv':<12} "
          f"{'ttft':>8} {'p95':>8} {'w_hit':>6} {'w_eff':>6} "
          f"{'err':>5} {'recov':>5} {'sub':>4} cache_ok")
    for r in data:
        ok = "OK" if not r["cache_note"] else "SUSPECT"
        print(f"{str(r['sweep_id']):>12} {str(r['orchestration'])[:22]:<22} "
              f"{str(r['serving'])[:12]:<12} "
              f"{str(r['avg_ttft_ms']):>8} {str(r['p95_ttft_ms']):>8} "
              f"{str(r['weighted_cache_hit_rate']):>6} "
              f"{str(r['weighted_cache_efficiency']):>6} "
              f"{str(r['avg_error_rate']):>5} {str(r['avg_error_recovery_rate']):>5} "
              f"{str(r['avg_turns_to_submit']):>4} {ok}")


if __name__ == "__main__":
    main()
