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


def load_swebench_tasks(split: str = "full",
                       split_name: str = "test",
                       limit: Optional[int] = None,
                       instance_ids: Optional[Iterable[str]] = None
                       ) -> list[SWEBenchTask]:
    """
    Load SWE-bench task instances.

    Args:
        split: 'full' | 'lite' | 'verified'
        split_name: HF dataset split, almost always 'test'.
        limit: take first N after filtering (None = all).
        instance_ids: if provided, keep only these IDs.

    Requires:
        pip install datasets
    """
    if split not in _DATASET_NAMES:
        raise ValueError(f"Unknown SWE-bench split '{split}'. "
                         f"Choose from {list(_DATASET_NAMES)}.")
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
        ))
    return out
