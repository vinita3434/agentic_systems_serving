"""
Tool execution layer.

Phase 1: MockTools returns canned realistic-looking observations for a
fixed SWE task, so the harness can drive a multi-turn loop without GPU or
Docker. The observations are deliberately long-ish (file dumps, ls output)
so the tool_output_compression strategy has something to chew on.

Phase 2: SWEEnvTools (TODO) wraps princeton-nlp/SWE-agent's SWEEnv to give
real bash + file editing inside the Docker container against an actual
SWE-bench task. The Tools interface (parse_action + execute) is the only
thing the agent loop depends on, so swapping is local.

Tools interface:
    parse_action(text)  -> Action | None
    execute(action)     -> Observation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Action:
    kind: str           # 'bash' | 'submit' | 'noop'
    command: str = ""


@dataclass
class Observation:
    content: str
    is_terminal: bool = False  # True if this ends the episode


# ---------- shared parser --------------------------------------------------


_BASH_RE = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.DOTALL)


def parse_action(model_text: str) -> Optional[Action]:
    """Pull the action out of a model response. SWE-agent style: a bash
    block, or the literal string `submit` inside one."""
    m = _BASH_RE.search(model_text)
    if not m:
        return None
    cmd = m.group(1).strip()
    if cmd == "submit" or cmd.startswith("submit"):
        return Action(kind="submit", command=cmd)
    return Action(kind="bash", command=cmd)


# ---------- mock tools -----------------------------------------------------


class MockTools:
    """
    Canned responses for a fixed JWT-auth bugfix task. Sequence designed so
    the orchestration strategies exercise meaningfully different paths:
      - First few observations are large (repo listing, full file) so
        compression matters.
      - Later turns are short (test results, edit confirmations) so
        sliding_window keeps the recent state.
    """

    REPO_LISTING = "\n".join([
        "total 192",
        "drwxr-xr-x  12 user user  4096 Jun 14 10:01 .",
        "drwxr-xr-x   3 user user  4096 Jun 14 10:00 ..",
        "-rw-r--r--   1 user user   220 Jun 14 10:00 .gitignore",
        "-rw-r--r--   1 user user  1071 Jun 14 10:00 LICENSE",
        "-rw-r--r--   1 user user  3271 Jun 14 10:00 README.md",
        "drwxr-xr-x   3 user user  4096 Jun 14 10:00 docs",
        "drwxr-xr-x   2 user user  4096 Jun 14 10:00 scripts",
        "drwxr-xr-x   5 user user  4096 Jun 14 10:00 src",
        "drwxr-xr-x   4 user user  4096 Jun 14 10:00 tests",
        "-rw-r--r--   1 user user   512 Jun 14 10:00 pyproject.toml",
        "-rw-r--r--   1 user user   189 Jun 14 10:00 requirements.txt",
    ] + [f"-rw-r--r--   1 user user  {i*37 % 9999:4d} Jun 14 10:00 noise_{i:03d}.py"
         for i in range(40)])

    JWT_FILE_DUMP = "\n".join([
        "file:src/auth/jwt_handler.py",
        "```python",
        "import jwt",
        "from datetime import datetime, timedelta",
        "",
        "SECRET_KEY = 'change-me-in-prod'",
        "ALGORITHM = 'HS256'",
        "",
        "def create_token(payload: dict, ttl_seconds: int = 3600) -> str:",
        "    payload = dict(payload)",
        "    payload['exp'] = datetime.utcnow() + timedelta(seconds=ttl_seconds)",
        "    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)",
        "",
        "def validate_token(token: str) -> dict:",
        "    # BUG: verify_exp disabled, expired tokens still validate",
        "    return jwt.decode(",
        "        token,",
        "        SECRET_KEY,",
        "        algorithms=[ALGORITHM],",
        "        options={'verify_exp': False},",
        "    )",
        "",
        "def refresh_token(token: str) -> str:",
        "    payload = validate_token(token)",
        "    payload.pop('exp', None)",
        "    return create_token(payload)",
        "```",
    ])

    EDIT_CONFIRMATION = "[file written: src/auth/jwt_handler.py — 1 substitution]"

    TEST_OUTPUT = "\n".join([
        "============================= test session starts =============================",
        "platform linux -- Python 3.11.4, pytest-7.4.0, pluggy-1.2.0",
        "rootdir: /workspace",
        "collected 4 items",
        "",
        "tests/auth/test_jwt.py::test_create_token PASSED                          [ 25%]",
        "tests/auth/test_jwt.py::test_validate_valid_token PASSED                  [ 50%]",
        "tests/auth/test_jwt.py::test_expired_token_rejected PASSED                [ 75%]",
        "tests/auth/test_jwt.py::test_refresh_token PASSED                         [100%]",
        "",
        "============================== 4 passed in 0.42s ===============================",
    ])

    SUBMIT_OUTPUT = "Patch submitted. Task complete."

    def __init__(self):
        self._step = 0

    def execute(self, action: Action) -> Observation:
        self._step += 1
        if action.kind == "submit":
            return Observation(content=self.SUBMIT_OUTPUT, is_terminal=True)

        cmd = action.command.lower()
        if "ls" in cmd:
            return Observation(content=f"<output>\n{self.REPO_LISTING}\n</output>")
        if "cat" in cmd and "jwt" in cmd:
            return Observation(content=f"<output>\n{self.JWT_FILE_DUMP}\n</output>")
        if "sed" in cmd or "edit" in cmd or "write" in cmd:
            return Observation(content=f"<output>\n{self.EDIT_CONFIRMATION}\n</output>")
        if "pytest" in cmd or "test" in cmd:
            return Observation(content=f"<output>\n{self.TEST_OUTPUT}\n</output>")
        return Observation(content=f"<output>\n[mock] executed: {action.command}\n</output>")


# SWEEnvTools lives in harness/sweenv_tools.py (lazy-imports sweagent).
