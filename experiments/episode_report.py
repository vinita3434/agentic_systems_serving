"""Per-episode metrics report.

Reads the raw per-turn (`<exp>.jsonl`) and per-episode (`<exp>__episodes.jsonl`)
logs for one experiment (a single serving x orchestration cell) and writes a
compact, readable **markdown report per EPISODE** (one task attempt), plus a
consolidated table across that config's episodes.

Deliberately narrow — only the metrics that matter when reading one episode:

  Serving
    - TTFT per turn        : series + inline sparkline + least-squares slope
    - cache hit rate       : avg + trend/slope + sparkline + inflection note
    - weighted cache eff.  : avg + trend/slope + sparkline + inflection note
  Outcome
    - verified (pass/fail) : did the graded patch pass
    - submitted first try  : turns_to_submit == 1
    - turns to submit      : turns_to_submit (None if never submitted)
  Cost
    - gpu_cost_usd         : wall_clock_s * gpu_hourly_usd / 3600
  Trajectory
    - error_rate, action_validity_rate, error_recovery_rate

Usage:
    python experiments/episode_report.py                 # newest experiment
    python experiments/episode_report.py --experiment-id sweep..._vllm_lru_32b
    python experiments/episode_report.py --all           # every experiment on disk

Reports land in experiments/results/reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORTS_DIR = RESULTS_DIR / "reports"

_SPARK = "▁▂▃▄▅▆▇█"


# ---------- small numeric helpers ------------------------------------------


def _sparkline(values: list[Optional[float]]) -> str:
    """Unicode sparkline; None -> a gap so missing turns are visible."""
    nums = [v for v in values if v is not None]
    if not nums:
        return "(no data)"
    lo, hi = min(nums), max(nums)
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        else:
            idx = int((v - lo) / span * (len(_SPARK) - 1))
            out.append(_SPARK[idx])
    return "".join(out)


def _slope(values: list[Optional[float]]) -> Optional[float]:
    """Least-squares slope (units per turn) over the non-None points."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    n = len(pts)
    if n < 2:
        return None
    sx = sum(i for i, _ in pts)
    sy = sum(v for _, v in pts)
    sxx = sum(i * i for i, _ in pts)
    sxy = sum(i * v for i, v in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def _trend(slope: Optional[float], eps: float) -> str:
    if slope is None:
        return "n/a"
    if slope > eps:
        return "rising"
    if slope < -eps:
        return "falling"
    return "flat"


def _avg(values: list[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _biggest_jump(values: list[Optional[float]]) -> Optional[tuple[int, float, float]]:
    """Return (turn_index_1based, from_value, to_value) of the largest
    absolute turn-to-turn change, ignoring None gaps."""
    best = None
    prev = None
    for i, v in enumerate(values):
        if v is None:
            continue
        if prev is not None:
            delta = abs(v - prev[1])
            if best is None or delta > best[0]:
                best = (delta, prev, (i, v))
        prev = (i, v)
    if best is None:
        return None
    (_, (fi, fv), (ti, tv)) = best
    return (ti + 1, fv, tv)


def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# ---------- data loading ---------------------------------------------------


def load_experiment(exp_id: str) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    """Return {episode_idx: [turn rows...]} and {episode_idx: episode row}."""
    turns_by_ep: dict[int, list[dict]] = {}
    turn_path = RESULTS_DIR / f"{exp_id}.jsonl"
    if turn_path.exists():
        for line in turn_path.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            turns_by_ep.setdefault(r["episode"], []).append(r)
    for ep in turns_by_ep.values():
        ep.sort(key=lambda r: r["turn"])

    eps_by_idx: dict[int, dict] = {}
    ep_path = RESULTS_DIR / f"{exp_id}__episodes.jsonl"
    if ep_path.exists():
        for line in ep_path.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            eps_by_idx[r["episode"]] = r
    return turns_by_ep, eps_by_idx


def list_experiment_ids() -> list[str]:
    """All experiment ids that have a per-episode log, newest first."""
    ids = []
    for p in RESULTS_DIR.glob("*__episodes.jsonl"):
        ids.append((p.stat().st_mtime, p.name[: -len("__episodes.jsonl")]))
    return [eid for _, eid in sorted(ids, reverse=True)]


# ---------- per-episode report ---------------------------------------------


def episode_report(exp_id: str, ep_idx: int, turns: list[dict], ep: dict) -> str:
    task = ep.get("task_id", "?")
    orch = ep.get("orchestration", "?")
    serv = ep.get("serving", "?")

    ttft = [t.get("ttft_ms") for t in turns]
    cache = [t.get("cache_hit_rate") for t in turns]
    # per-turn weighted cache efficiency = hit_tokens / reusable_prefix_tokens
    eff: list[Optional[float]] = []
    for t in turns:
        rp = t.get("reusable_prefix_tokens") or 0
        ht = t.get("cache_hit_tokens")
        eff.append((ht / rp) if (ht is not None and rp > 0) else None)

    ttft_slope = _slope(ttft)
    cache_slope = _slope(cache)
    eff_slope = _slope(eff)

    # outcome
    verified = ep.get("verified")
    tts = ep.get("turns_to_submit")
    first_try = (tts == 1)

    # inflection notes
    def _inflect(series, label, unit):
        j = _biggest_jump(series)
        if j is None:
            return f"{label}: not enough data for an inflection."
        turn, fv, tv = j
        direction = "jumped" if tv > fv else "dropped"
        return (f"{label} {direction} most at turn {turn} "
                f"({fv:.2f}{unit} -> {tv:.2f}{unit}).")

    cache_note = _inflect(cache, "Cache hit rate", "")
    eff_note = _inflect(eff, "Weighted cache efficiency", "")
    # correlate a cache drop with a reusable-prefix drop (summarization signal)
    rp_series = [t.get("reusable_prefix_tokens") for t in turns]
    rp_jump = _biggest_jump([float(x) if x is not None else None for x in rp_series])
    corr = ""
    cj = _biggest_jump(cache)
    if cj and rp_jump and abs(cj[0] - rp_jump[0]) <= 1 and rp_jump[2] < rp_jump[1]:
        corr = (f" This coincides with the reusable prefix collapsing around "
                f"turn {rp_jump[0]} — i.e. the orchestration rewrote the prompt "
                f"prefix (e.g. a summarization/compression step), invalidating "
                f"cached blocks.")

    L = []
    L.append(f"# Episode report — {task}")
    L.append("")
    L.append(f"- **experiment**: `{exp_id}`")
    L.append(f"- **serving**: `{serv}`  |  **orchestration**: `{orch}`  |  "
             f"**episode**: {ep_idx}")
    L.append(f"- **turns**: {ep.get('n_turns')}")
    L.append("")

    L.append("## Outcome / success")
    L.append(f"- **verified (patch passed tests)**: **{verified}**")
    L.append(f"- **submitted on first try**: {first_try}")
    L.append(f"- **turns to submit**: {tts if tts is not None else 'never submitted'}")
    L.append("")

    L.append("## Serving")
    L.append("")
    L.append("### TTFT per turn (ms)")
    L.append(f"```\n{_sparkline(ttft)}\n```")
    L.append(f"- values: {[round(x,1) if x is not None else None for x in ttft]}")
    L.append(f"- **mean**: {_fmt(_avg(ttft),1)} ms  |  **slope**: "
             f"{_fmt(ttft_slope,1)} ms/turn ({_trend(ttft_slope, 1.0)})")
    L.append("")
    L.append("### Cache hit rate per turn")
    L.append(f"```\n{_sparkline(cache)}\n```")
    L.append(f"- values: {[round(x,3) if x is not None else None for x in cache]}")
    L.append(f"- **mean**: {_fmt(_avg(cache))}  |  **slope**: "
             f"{_fmt(cache_slope,4)}/turn ({_trend(cache_slope, 1e-3)})")
    L.append(f"- {cache_note}{corr}")
    L.append("")
    L.append("### Weighted cache efficiency per turn (hit_tokens / reusable_prefix)")
    L.append(f"```\n{_sparkline(eff)}\n```")
    L.append(f"- values: {[round(x,3) if x is not None else None for x in eff]}")
    L.append(f"- **mean**: {_fmt(_avg(eff))}  |  **slope**: "
             f"{_fmt(eff_slope,4)}/turn ({_trend(eff_slope, 1e-3)})")
    L.append(f"- {eff_note}")
    L.append("")

    L.append("## Cost")
    L.append(f"- **gpu_cost_usd** (wall_clock_s x gpu_hourly / 3600): "
             f"**${_fmt(ep.get('gpu_cost_usd'), 4)}**  "
             f"(wall {_fmt(ep.get('wall_clock_s'),1)} s @ "
             f"${_fmt(ep.get('gpu_hourly_usd'),2)}/hr)")
    L.append("")

    L.append("## Trajectory quality")
    L.append(f"- **error_rate**: {_fmt(ep.get('error_rate'))}")
    L.append(f"- **action_validity_rate**: {_fmt(ep.get('action_validity_rate'))}")
    L.append(f"- **error_recovery_rate**: {_fmt(ep.get('error_recovery_rate'))}")
    L.append("")

    return "\n".join(L)


# ---------- consolidation across a config's episodes -----------------------


def consolidate(exp_id: str, turns_by_ep, eps_by_idx) -> str:
    L = [f"# Consolidated — `{exp_id}`", "",
         "One row per task (episode) for this serving x orchestration config.",
         "",
         "| task | verified | turns | first_try | to_submit | mean_ttft_ms | "
         "ttft_slope | mean_cache | mean_eff | error_rate | gpu_cost_usd |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for idx in sorted(eps_by_idx):
        ep = eps_by_idx[idx]
        turns = turns_by_ep.get(idx, [])
        ttft = [t.get("ttft_ms") for t in turns]
        cache = [t.get("cache_hit_rate") for t in turns]
        eff = []
        for t in turns:
            rp = t.get("reusable_prefix_tokens") or 0
            ht = t.get("cache_hit_tokens")
            eff.append((ht / rp) if (ht is not None and rp > 0) else None)
        tts = ep.get("turns_to_submit")
        L.append(
            f"| {ep.get('task_id')} | {ep.get('verified')} | {ep.get('n_turns')} "
            f"| {tts == 1} | {tts if tts is not None else '—'} "
            f"| {_fmt(_avg(ttft),1)} | {_fmt(_slope(ttft),1)} "
            f"| {_fmt(_avg(cache))} | {_fmt(_avg(eff))} "
            f"| {_fmt(ep.get('error_rate'))} | {_fmt(ep.get('gpu_cost_usd'),4)} |")
    L.append("")
    return "\n".join(L)


# ---------- driver ---------------------------------------------------------


def process(exp_id: str) -> None:
    turns_by_ep, eps_by_idx = load_experiment(exp_id)
    if not eps_by_idx:
        print(f"[skip] no episode rows for {exp_id}")
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for idx in sorted(eps_by_idx):
        md = episode_report(exp_id, idx, turns_by_ep.get(idx, []), eps_by_idx[idx])
        out = REPORTS_DIR / f"{exp_id}__ep{idx}.md"
        out.write_text(md)
        print(f"wrote {out}")
    cons = consolidate(exp_id, turns_by_ep, eps_by_idx)
    cout = REPORTS_DIR / f"{exp_id}__consolidated.md"
    cout.write_text(cons)
    print(f"wrote {cout}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", default=None,
                   help="Experiment id (file stem). Default: the newest one.")
    p.add_argument("--all", action="store_true",
                   help="Generate reports for every experiment on disk.")
    args = p.parse_args()

    if args.all:
        ids = list_experiment_ids()
    elif args.experiment_id:
        ids = [args.experiment_id]
    else:
        ids = list_experiment_ids()[:1]
    if not ids:
        print("No experiments found in experiments/results/.")
        return
    for eid in ids:
        process(eid)


if __name__ == "__main__":
    main()
