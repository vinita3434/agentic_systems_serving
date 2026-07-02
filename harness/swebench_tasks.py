"""
SWE-bench task loader.

Loads instances from the princeton-nlp/SWE-bench Hugging Face dataset and
wraps them as harness.agent_loop.Task objects.

Splits:
  - 'full'     princeton-nlp/SWE-bench         (2294 instances)
  - 'lite'     princeton-nlp/SWE-bench_Lite    (300 instances)
  - 'verified' princeton-nlp/SWE-bench_Verified (500 instances)

Dependency is lazy: importing this module does not require the `datasets`
package. Only calling load_swebench_tasks() does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class SWEBenchTask:
    """Subset of SWE-bench fields the harness needs to drive an episode."""
    instance_id: str
    repo: str               # e.g. "django/django"
    base_commit: str        # SHA the agent starts from
    problem_statement: str  # the issue body / task description
    hints_text: str = ""
    version: str = ""
    test_patch: str = ""
    patch: str = ""         # the gold patch (used only for evaluation)
    difficulty: str = ""    # Verified-split annotation, e.g. "1-4 hours"

    def as_task(self):
        """Convert to the lightweight Task used by agent_loop.run_episode."""
        from harness.agent_loop import Task  # local import avoids cycles
        return Task(task_id=self.instance_id,
                    description=self.problem_statement)


_DATASET_NAMES = {
    "full": "princeton-nlp/SWE-bench",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
}

# Difficulty buckets in SWE-bench_Verified considered "hard" — tasks that
# take a human 1+ hours, and thus tend to require long agent trajectories.
# Matched loosely (substring, lowercased) since the exact label strings can
# vary slightly across dataset revisions.
_HARD_DIFFICULTY_KEYWORDS = ("1-4 hour", ">4 hour", "1-4 hours", ">4 hours")


def _is_hard(difficulty: str) -> bool:
    d = (difficulty or "").lower()
    return any(k in d for k in _HARD_DIFFICULTY_KEYWORDS)


def load_swebench_tasks(split: str = "full",
                       split_name: str = "test",
                       limit: Optional[int] = None,
                       instance_ids: Optional[Iterable[str]] = None,
                       hard_only: bool = False
                       ) -> list[SWEBenchTask]:
    """
    Load SWE-bench task instances.

    Args:
        split: 'full' | 'lite' | 'verified'
        split_name: HF dataset split, almost always 'test'.
        limit: take first N after filtering (None = all).
        instance_ids: if provided, keep only these IDs.
        hard_only: keep only hard-difficulty tasks (1-4 hrs / >4 hrs).
            Requires the 'verified' split — only it carries a difficulty
            annotation.

    Requires:
        pip install datasets
    """
    if split not in _DATASET_NAMES:
        raise ValueError(f"Unknown SWE-bench split '{split}'. "
                         f"Choose from {list(_DATASET_NAMES)}.")
    if hard_only and split != "verified":
        raise ValueError(
            "hard_only=True requires split='verified' — only the Verified "
            "split carries a difficulty annotation. Re-run with "
            "--swebench-split verified.")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "load_swebench_tasks requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from e

    ds = load_dataset(_DATASET_NAMES[split], split=split_name)

    if instance_ids is not None:
        wanted = set(instance_ids)
        ds = ds.filter(lambda row: row["instance_id"] in wanted)

    if hard_only:
        ds = ds.filter(lambda row: _is_hard(row.get("difficulty", "")))
        if len(ds) == 0:
            raise ValueError(
                "hard_only filter matched 0 instances. Check the Verified "
                "split's difficulty labels (expected buckets like '1-4 hours' "
                "/ '>4 hours').")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    out: list[SWEBenchTask] = []
    for row in ds:
        out.append(SWEBenchTask(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row.get("problem_statement", ""),
            hints_text=row.get("hints_text", ""),
            version=row.get("version", ""),
            test_patch=row.get("test_patch", ""),
            patch=row.get("patch", ""),
            difficulty=row.get("difficulty", ""),
        ))
    return out
