"""
Docker task-execution layer — Phase 2 tools against real SWE-bench tasks.

Drives Docker DIRECTLY via the docker SDK + SWE-bench's own instance images.
No SWE-agent dependency: SWE-agent 1.x is a CLI app (not library-importable)
and its API drifts, so we own the container lifecycle ourselves. This keeps
the custom agent loop, orchestration strategies, and metrics fully intact —
only the "run bash in a container" primitive lives here.

Per task:
  * start()          pull the SWE-bench instance image (repo already at
                     base_commit + deps installed under /testbed) and launch
                     a long-lived container.
  * execute(action)  run the agent's bash command inside that container.
  * evaluate_patch() collect the working-tree diff and run swebench's
                     run_instance() to get resolved True/False.
  * close()          stop + remove the container.

Interface (parse_action + execute + evaluate_patch) is identical to
MockTools, so agent_loop.py is unchanged.

Requires (on the box that runs the harness): a working Docker daemon,
`pip install docker swebench`, and x86_64 (SWE-bench images are x86).
"""

from __future__ import annotations

from typing import Any, Optional

from harness.tools import Action, Observation, parse_action  # re-exported


WORKDIR = "/testbed"  # SWE-bench images check the repo out here


class SWEEnvTools:
    """Tools backed by a SWE-bench Docker container. Same .execute()/
    .evaluate_patch() surface as MockTools."""

    def __init__(self, swe_task: Any, command_timeout: float = 120.0,
                 namespace: str = "swebench", arch: str = "x86_64",
                 image_override: Optional[str] = None):
        self.swe_task = swe_task
        self.command_timeout = command_timeout
        self.namespace = namespace
        self.arch = arch
        self.image_override = image_override
        self._client = None       # docker.DockerClient
        self._container = None
        self._test_spec = None

    # ----- construction -----

    @classmethod
    def from_swebench(cls, swebench_task, **kwargs) -> "SWEEnvTools":
        return cls(swe_task=swebench_task, **kwargs)

    def _make_test_spec(self):
        if self._test_spec is None:
            from swebench.harness.test_spec.test_spec import make_test_spec
            instance = self.swe_task.raw or {}
            self._test_spec = make_test_spec(
                instance, namespace=self.namespace, arch=self.arch)
        return self._test_spec

    def _instance_image(self) -> str:
        if self.image_override:
            return self.image_override
        return self._make_test_spec().instance_image_key

    # ----- lifecycle -----

    def start(self) -> None:
        """Pull the instance image (if needed) and launch a container that
        idles so we can exec commands into it."""
        if self._container is not None:
            return
        import docker
        from docker.errors import ImageNotFound

        self._client = docker.from_env()
        image = self._instance_image()
        try:
            self._client.images.get(image)
        except ImageNotFound:
            # SWE-bench publishes instance images under the 'swebench' Docker
            # Hub namespace. If this instance's image isn't published, it must
            # be built first (swebench.harness build utilities).
            print(f"[SWEEnvTools] pulling {image} ...")
            self._client.images.pull(image)

        self._container = self._client.containers.run(
            image,
            command=["sleep", "infinity"],
            detach=True,
            working_dir=WORKDIR,
            tty=False,
            network_mode="none",   # tasks shouldn't need network; keeps it hermetic
            auto_remove=False,
        )

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.remove(force=True)
            finally:
                self._container = None
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    # ----- async context manager -----

    async def __aenter__(self) -> "SWEEnvTools":
        import asyncio
        await asyncio.get_running_loop().run_in_executor(None, self.start)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        import asyncio
        await asyncio.get_running_loop().run_in_executor(None, self.close)

    # ----- command execution -----

    def _exec(self, command: str) -> str:
        """Run a shell command in the container, bounded by command_timeout
        (enforced with coreutils `timeout` so a hung command can't stall the
        episode). Returns combined stdout+stderr."""
        if self._container is None:
            raise RuntimeError("SWEEnvTools not started. Call start() first.")
        cmd = ["timeout", str(int(self.command_timeout)), "bash", "-c", command]
        res = self._container.exec_run(cmd=cmd, workdir=WORKDIR, demux=False)
        out = res.output
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if res.exit_code == 124:
            out = (out or "") + f"\n[command timed out after {int(self.command_timeout)}s]"
        return out or ""

    def execute(self, action: Action) -> Observation:
        """Synchronous execute — matches MockTools.execute; agent_loop runs
        it in an executor."""
        if action.kind == "submit":
            patch = self._collect_patch()
            body = "Patch submitted." if patch else "Patch submitted (empty diff)."
            return Observation(content=f"<output>\n{body}\n</output>",
                               is_terminal=True)
        try:
            stdout = self._exec(action.command)
        except Exception as e:
            return Observation(content=f"<output>\n[error: {e}]\n</output>")
        return Observation(content=f"<output>\n{stdout}\n</output>")

    # ----- patch collection + evaluation -----

    def _collect_patch(self) -> str:
        """The agent's changes as a unified diff, including new files."""
        try:
            self._exec("git add -A")
            return self._exec("git diff --cached")
        except Exception as e:
            print(f"[SWEEnvTools._collect_patch error: {e}]")
            return ""

    def evaluate_patch(self) -> Optional[bool]:
        """Run swebench's single-instance evaluator on the agent's patch.

        Returns True if the task's hidden tests pass under the patch, False if
        not, None if evaluation could not be run. Fails open.
        """
        try:
            patch = self._collect_patch()
            if not patch.strip():
                return False

            import docker
            from swebench.harness.run_evaluation import run_instance

            instance_id = self.swe_task.instance_id
            pred = {
                "instance_id": instance_id,
                "model_patch": patch,
                "model_name_or_path": "agentic_systems_serving",
            }
            test_spec = self._make_test_spec()
            client = self._client or docker.from_env()
            run_id = f"eval_{instance_id}"

            report = run_instance(
                test_spec, pred,
                rm_image=False, force_rebuild=False,
                client=client, run_id=run_id,
                timeout=1800,
            )
            return _extract_resolved(report, instance_id)
        except Exception as e:
            print(f"[SWEEnvTools.evaluate_patch error: {e}]")
            return None


def _extract_resolved(report: Any, instance_id: str) -> Optional[bool]:
    """Pull the boolean resolved status out of run_instance's report dict,
    defensively (its shape has shifted across swebench versions)."""
    if not isinstance(report, dict):
        return None
    node = report.get(instance_id, report)
    if isinstance(node, dict):
        if "resolved" in node:
            return bool(node["resolved"])
        # some versions nest under the instance id twice or under 'report'
        for v in node.values():
            if isinstance(v, dict) and "resolved" in v:
                return bool(v["resolved"])
    return None
