"""
SWEEnv adapter — Phase 2 tool execution against real SWE-bench tasks.

Wraps princeton-nlp/SWE-agent's SWEEnv inside the harness.tools.Tools
interface (parse_action + execute), so the same agent_loop.py works
verbatim whether tools=MockTools() or tools=SWEEnvTools(task).

SWE-agent's API has shifted across versions; this adapter is defensive
about imports and method names. Tested against sweagent >= 1.0.

Lifecycle:
    tools = SWEEnvTools.from_swebench(task)
    tools.start()        # launches docker container, checks out base_commit
    ... use tools.execute(action) in the agent loop ...
    tools.close()        # cleanup

Or use as an async context manager:
    async with SWEEnvTools.from_swebench(task) as tools:
        ...
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from harness.tools import Action, Observation, parse_action  # re-exported for convenience


class SWEEnvTools:
    """
    Tools that execute against a real SWEEnv. The class is intentionally
    small — it delegates everything heavy to SWEEnv. The agent loop sees
    the same .execute(action) interface as MockTools.

    parse_action is re-exported from harness.tools so callers don't have
    to know about two parsers.
    """

    def __init__(self, swe_task: Any, env_kwargs: Optional[dict] = None,
                 command_timeout: float = 120.0):
        self.swe_task = swe_task
        self.env_kwargs = env_kwargs or {}
        self.command_timeout = command_timeout
        self._env = None  # SWEEnv instance, created in start()

    # ----- construction -----

    @classmethod
    def from_swebench(cls, swebench_task, **env_kwargs) -> "SWEEnvTools":
        """Build SWEEnvTools from a SWEBenchTask (harness.swebench_tasks)."""
        return cls(swe_task=swebench_task, env_kwargs=env_kwargs)

    # ----- lifecycle -----

    def start(self) -> None:
        """Launch the underlying SWE-agent environment.

        Performs the import lazily so the rest of the harness can run
        without sweagent installed.
        """
        if self._env is not None:
            return
        try:
            # Recent sweagent versions
            from sweagent.environment.swe_env import SWEEnv  # type: ignore
            from sweagent.environment.config import EnvironmentArguments  # type: ignore
        except ImportError as e:
            raise ImportError(
                "SWEEnvTools requires princeton-nlp/SWE-agent. "
                "Install with: pip install sweagent  "
                "(or follow https://github.com/princeton-nlp/SWE-agent for the latest setup)."
            ) from e

        # SWE-agent constructs SWEEnv from EnvironmentArguments; pass through
        # any kwargs the caller provided.
        env_args = EnvironmentArguments(
            data_path=self._build_data_path(),
            repo_path="",                       # SWEEnv resolves from instance
            container_name=f"swe-{self.swe_task.instance_id}",
            image_name=self.env_kwargs.get("image_name", "sweagent/swe-agent:latest"),
            **{k: v for k, v in self.env_kwargs.items() if k != "image_name"},
        )
        self._env = SWEEnv(env_args)
        self._env.reset()  # checks out base_commit, installs deps

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def _build_data_path(self) -> str:
        """SWE-agent expects either a HF dataset name or a local JSONL.
        We pass the instance_id and let SWE-agent pull from SWE-bench."""
        # princeton-nlp/SWE-bench shorthand SWE-agent understands.
        return f"princeton-nlp/SWE-bench__{self.swe_task.instance_id}"

    # ----- async context manager -----

    async def __aenter__(self) -> "SWEEnvTools":
        await asyncio.get_running_loop().run_in_executor(None, self.start)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.close)

    # ----- tools interface -----

    def execute(self, action: Action) -> Observation:
        """Synchronous execute — matches MockTools.execute. The agent_loop
        already runs this in an executor where the LLM call is async."""
        if self._env is None:
            raise RuntimeError("SWEEnvTools not started. Call start() first.")

        if action.kind == "submit":
            # SWE-agent uses 'submit' as a sentinel; the env will diff the
            # working tree against base_commit and return the patch.
            try:
                patch = self._env.submit() or ""
            except Exception as e:
                return Observation(content=f"<output>\n[submit failed: {e}]\n</output>",
                                   is_terminal=True)
            return Observation(content=f"<output>\nPatch submitted.\n{patch}\n</output>",
                               is_terminal=True)

        try:
            stdout = self._env.communicate(
                input=action.command, timeout_duration=self.command_timeout)
        except Exception as e:
            return Observation(content=f"<output>\n[error: {e}]\n</output>")

        return Observation(content=f"<output>\n{stdout}\n</output>")

    # ----- patch evaluation -----

    def evaluate_patch(self) -> Optional[bool]:
        """Run the SWE-bench evaluation harness on the current patch.

        Returns True if the task's tests pass under the agent's patch,
        False if they don't, None if evaluation could not be run.

        Shells out to `swebench.harness.run_evaluation` because that
        module owns the per-instance Docker test container lifecycle.
        Per-episode cost is meaningful (30–120 s typical); call only when
        an evaluation result is actually wanted.
        """
        if self._env is None:
            return None
        try:
            patch = self._collect_patch()
            if not patch:
                return False
            return self._run_swebench_evaluator(patch)
        except Exception as e:
            print(f"[SWEEnvTools.evaluate_patch error: {e}]")
            return None

    def _collect_patch(self) -> str:
        """SWEEnv exposes the working-tree diff via submit() or get_patch().
        The exact method has shifted across sweagent releases; try both."""
        for attr in ("get_patch", "submit"):
            fn = getattr(self._env, attr, None)
            if callable(fn):
                try:
                    out = fn()
                except TypeError:
                    out = fn(None)  # some versions take an arg
                if out:
                    return out
        return ""

    def _run_swebench_evaluator(self, patch: str) -> Optional[bool]:
        import json
        import subprocess
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                          delete=False) as f:
            json.dump([{
                "instance_id": self.swe_task.instance_id,
                "model_patch": patch,
                "model_name_or_path": "agentic_systems_serving",
            }], f)
            predictions_path = f.name

        run_id = f"eval_{self.swe_task.instance_id}"
        cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
               "--predictions_path", predictions_path,
               "--instance_ids", self.swe_task.instance_id,
               "--max_workers", "1",
               "--run_id", run_id]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=600)
        # The harness prints a summary that includes "resolved" instances.
        # We parse loosely; if the harness changes its output format, this
        # returns None and the sweep still completes.
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        if self.swe_task.instance_id in text and "resolved" in text.lower():
            # Heuristic: look for "Instances resolved: N" or "resolved_ids"
            # containing our instance.
            if f'"{self.swe_task.instance_id}"' in text or \
               f"'{self.swe_task.instance_id}'" in text:
                return True
            return False
        return None
