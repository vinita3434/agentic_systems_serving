"""
Custom agent loop (Option C).

We own every decision per turn:
    - system prompt construction
    - assembling messages from history via the configured strategy
    - calling the LLM
    - parsing the action
    - executing the tool
    - formatting the observation
    - appending to history

SWE-agent's loop is NOT used. Tools are swappable (MockTools now, SWEEnv later).

Run one episode with:
    asyncio.run(run_episode(
        task=...,
        orchestration_cfg=...,
        serving_cfg=...,
        llm_client=...,
        tools=...,
        metrics_logger=...,
    ))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import httpx

from harness.context_manager import assemble, Message, messages_tokens
from harness.llm_client import CompletionResult
from harness.metrics_logger import MetricsLogger, TurnMetrics
from harness.tools import Action, Observation, parse_action


SYSTEM_PROMPT = (
    "You are an autonomous software engineering agent fixing a bug in a real "
    "repository. Your working directory is the repository root.\n\n"
    "You interact with the codebase through a special command interface. Each "
    "turn: first write a short DISCUSSION of your reasoning, then issue EXACTLY "
    "ONE command inside a single fenced ``` block. Available commands:\n"
    "  open <path> [line]         open a file (shows a numbered window)\n"
    "  goto <line>                move the window to a line\n"
    "  scroll_down | scroll_up    page through the open file\n"
    "  search_dir <term> [dir]    search a term across files\n"
    "  search_file <term> [file]  search within the open file\n"
    "  find_file <name> [dir]     find files by name\n"
    "  create <path>              create a new file\n"
    "  edit <start>:<end> <<'EOF'\n  <new lines>\n  EOF   replace inclusive line "
    "range start..end in the open file\n"
    "  <any shell command>        you may also run normal shell commands and tests\n"
    "  submit                     submit your solution\n\n"
    "Workflow: explore to locate the buggy code, open the file, edit it to fix "
    "the bug, run the project's tests to verify, then submit. Do NOT run "
    "`submit` on the first turn or before you have edited a file and confirmed "
    "the fix — an empty submission fails the task. One command per turn."
)


# A one-shot demonstration on a DIFFERENT toy bug, prepended to the first user
# message. Shows the exact expected format (DISCUSSION + one fenced command)
# and the explore -> open -> edit -> test -> submit workflow. Static text, so
# it's part of the stable prompt prefix (prefix-cache friendly) and is not
# counted as a real turn.
DEMONSTRATION = (
    "Here is a short demonstration of the workflow on a DIFFERENT example bug "
    "(one command per turn):\n\n"
    "DISCUSSION\nFind where the function is defined.\n"
    "```\nsearch_dir \"def add_numbers\"\n```\n"
    "OBSERVATION\n./calc/ops.py:12:def add_numbers(a, b):\n\n"
    "DISCUSSION\nOpen the file at that line.\n```\nopen calc/ops.py 12\n```\n"
    "OBSERVATION\n[File: calc/ops.py (30 lines total)]\n12:def add_numbers(a, b):\n"
    "13:    return a - b\n\n"
    "DISCUSSION\nThe bug: it subtracts instead of adds. Fix line 13.\n"
    "```\nedit 13:13 <<'EOF'\n    return a + b\nEOF\n```\n"
    "OBSERVATION\n[File: calc/ops.py edited: lines 13-13 replaced]\n\n"
    "DISCUSSION\nRun the tests to verify.\n"
    "```\npython -m pytest tests/test_ops.py -q\n```\n"
    "OBSERVATION\n1 passed\n\n"
    "DISCUSSION\nThe fix is verified. Submit.\n```\nsubmit\n```\n\n"
    "Now solve the ACTUAL task below the same way.\n"
)


@dataclass
class Task:
    task_id: str
    description: str


@dataclass
class EpisodeResult:
    task_id: str
    n_turns: int
    completed: bool
    final_history_tokens: int
    summary: dict
    # Patch evaluation result. True/False when the swebench harness ran
    # against the agent's patch; None for mock tasks or when evaluation
    # was skipped.
    verified: Optional[bool] = None
    # Cost accounting per episode.
    total_prompt_tokens: int = 0       # sum of prompt_tokens across all turns
    total_completion_tokens: int = 0   # sum of completion_tokens across all turns
    wall_clock_s: float = 0.0          # episode duration from first turn to terminal
    # Trajectory-quality metrics (how the agent worked, not just whether it
    # succeeded) + the within-episode cache-hit-rate curve. See run_episode.
    trajectory: dict = field(default_factory=dict)


# Substrings that mark an observation as a failed / error action. The first
# three are harness-injected and reliable; the rest are common bash / Python
# failure signatures (heuristic — real stdout has no exit code exposed).
_ERROR_MARKERS = (
    "[error:",
    "[submit failed",
    "[no valid action found]",
    "command not found",
    "no such file or directory",
    "traceback (most recent call last)",
    "syntaxerror",
)


def _is_error_observation(content: str) -> bool:
    low = content.lower()
    return any(marker in low for marker in _ERROR_MARKERS)


def _serialize_prompt(messages: list[Message]) -> str:
    """Flatten assembled messages to a string for prefix comparison. The
    exact form only needs to be stable turn-to-turn, not match the model's
    tokenizer — this is a diagnostic, not the served prompt."""
    return "\n".join(f"<{m.get('role', 'user')}>\n{m.get('content', '')}"
                     for m in messages)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(model: str):
    """Lazily load the model's tokenizer for exact R_n token counting.
    Returns None if transformers / the tokenizer can't be loaded, in which
    case the caller falls back to a ~4-chars/token estimate."""
    if model in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model]
    tok = None
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model)
    except Exception:
        tok = None
    _TOKENIZER_CACHE[model] = tok
    return tok


def _prefix_tokens(prev: Optional[str], cur: str, tokenizer) -> int:
    """Reusable-prefix tokens R_n between two serialized prompts. Uses the
    real tokenizer when available (exact token-sequence common prefix);
    otherwise estimates at ~4 chars/token."""
    if prev is None:
        return 0
    if tokenizer is not None:
        try:
            a = tokenizer.encode(prev, add_special_tokens=False)
            b = tokenizer.encode(cur, add_special_tokens=False)
            n = min(len(a), len(b))
            i = 0
            while i < n and a[i] == b[i]:
                i += 1
            return i
        except Exception:
            pass
    return _common_prefix_len(prev, cur) // 4


def _format_observation(obs: Observation) -> Message:
    return {"role": "user", "content": obs.content, "meta": {"kind": "observation"}}


def _format_assistant(content: str) -> Message:
    return {"role": "assistant", "content": content, "meta": {"kind": "action"}}


async def _summarize_via_llm(llm_client, messages_to_summarize: list[Message]) -> str:
    """Used by the summarization orchestration strategy. Calls the same LLM
    with a one-shot summarize prompt."""
    prompt = (
        "Summarize the following agent turns in <= 200 words. Preserve file "
        "paths touched, bugs identified, edits made, and test outcomes. "
        "Omit boilerplate and verbose tool output.\n\n"
    )
    prompt += "\n".join(f"[{m.get('role')}]\n{m.get('content','')}"
                        for m in messages_to_summarize)
    summarize_msgs = [
        {"role": "system", "content": "You produce dense agent-trace summaries."},
        {"role": "user", "content": prompt},
    ]
    result = await llm_client.chat(summarize_msgs, max_tokens=400, temperature=0.0)
    return result.content


async def run_episode(
    *,
    task: Task,
    orchestration_cfg: dict,
    serving_cfg: dict,
    llm_client,
    tools,
    metrics_logger: MetricsLogger,
    episode_idx: int = 0,
    max_turns: int = 20,
) -> EpisodeResult:
    """Run one agent episode, logging metrics per turn."""

    history: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT, "meta": {"kind": "system"}},
        {"role": "user",
         "content": f"{DEMONSTRATION}\n<task>\n{task.description}\n</task>",
         "meta": {"kind": "task"}},
    ]

    strategy = orchestration_cfg["strategy"]
    params = orchestration_cfg.get("params") or {}

    # Synchronous adapter for the summarization strategy. The strategy
    # function expects a sync callable; we bridge to the async llm_client
    # via a fresh event loop call.
    def _sync_summarizer(msgs: list[Message]) -> str:
        return asyncio.get_event_loop().run_until_complete(
            _summarize_via_llm(llm_client, msgs)
        )

    completed = False
    last_finish_reason = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    t_episode_start = asyncio.get_running_loop().time()

    # Trajectory-quality accumulators.
    n_valid = n_invalid = n_bash = n_submit = n_error_obs = n_repeat = 0
    turns_to_submit: Optional[int] = None
    prev_bash_cmd: Optional[str] = None
    cache_hit_rate_by_turn: list[Optional[float]] = []
    reusable_prefix_by_turn: list[int] = []
    error_flags: list[bool] = []          # per-turn: did this turn hit an error?
    prev_prompt_text: Optional[str] = None
    tokenizer = _get_tokenizer(serving_cfg.get("model",
                                               "Qwen/Qwen2.5-Coder-7B-Instruct"))

    for turn in range(1, max_turns + 1):
        assembled = assemble(strategy, history, params,
                             summarizer=_sync_summarizer if strategy == "summarization" else None)

        # Reusable prefix (R_n): tokens this prompt shares with the previous
        # turn's prompt. Uses the real model tokenizer when available, else a
        # ~4-chars/token estimate. Turn 1 has no predecessor -> 0.
        prompt_text = _serialize_prompt(assembled.messages)
        reusable_prefix_tokens = _prefix_tokens(prev_prompt_text, prompt_text, tokenizer)
        prev_prompt_text = prompt_text
        reusable_prefix_by_turn.append(reusable_prefix_tokens)

        try:
            result: CompletionResult = await llm_client.chat(
                assembled.messages, max_tokens=512, temperature=0.0
            )
        except httpx.HTTPStatusError as e:
            # A 400 here is almost always the prompt exceeding the model's
            # max context length — full-context strategies (full_context,
            # cache_aware_ordering) grow unbounded on long trajectories. End
            # the episode gracefully (keep whatever edits were made, then
            # evaluate) instead of crashing the whole sweep.
            code = e.response.status_code
            print(f"[episode {episode_idx}: server returned {code} at turn "
                  f"{turn} — likely context-length overflow; ending episode "
                  f"and evaluating the patch so far]")
            last_finish_reason = "context_overflow" if code == 400 else f"http_{code}"
            break

        metrics_logger.log(TurnMetrics(
            experiment_id=metrics_logger.experiment_id,
            orchestration=orchestration_cfg["name"],
            serving=serving_cfg.get("name", "unknown"),
            episode=episode_idx,
            turn=turn,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            context_tokens=assembled.context_tokens,
            raw_history_tokens=assembled.raw_history_tokens,
            ttft_ms=result.ttft_ms,
            total_latency_ms=result.total_latency_ms,
            cache_hit_rate=result.cache_hit_rate,
            cache_hit_tokens=result.cache_hit_tokens,
            reusable_prefix_tokens=reusable_prefix_tokens,
            finish_reason=result.finish_reason,
        ))
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        cache_hit_rate_by_turn.append(result.cache_hit_rate)

        last_finish_reason = result.finish_reason
        history.append(_format_assistant(result.content))

        action = parse_action(result.content)
        if action is None:
            # Model produced no parseable action: both an invalid action and
            # an error observation.
            n_invalid += 1
            n_error_obs += 1
            error_flags.append(True)
            history.append({"role": "user",
                            "content": "<output>\n[no valid action found]\n</output>",
                            "meta": {"kind": "observation"}})
            continue

        n_valid += 1
        if action.kind == "submit":
            n_submit += 1
            if turns_to_submit is None:
                turns_to_submit = turn
        elif action.kind == "bash":
            n_bash += 1
            cmd_norm = action.command.strip()
            if prev_bash_cmd is not None and cmd_norm == prev_bash_cmd:
                n_repeat += 1  # re-issued an identical command (thrash / stuck)
            prev_bash_cmd = cmd_norm

        # tools.execute may block (SWEEnv runs bash in a docker container).
        # Run it off the event loop so we don't stall the LLM client.
        observation = await asyncio.get_running_loop().run_in_executor(
            None, tools.execute, action)
        obs_is_error = _is_error_observation(observation.content)
        if obs_is_error:
            n_error_obs += 1
        error_flags.append(obs_is_error)
        history.append(_format_observation(observation))

        if observation.is_terminal or action.kind == "submit":
            completed = True
            break

    # Patch evaluation: only runs when the tools backend exposes one
    # (SWEEnvTools does; MockTools does not). Fails open — any error
    # leaves verified=None so the sweep continues.
    verified: Optional[bool] = None
    if completed:
        evaluator: Optional[Callable] = getattr(tools, "evaluate_patch", None)
        if callable(evaluator):
            try:
                verified = await asyncio.get_running_loop().run_in_executor(
                    None, evaluator)
            except Exception as e:
                print(f"[evaluate_patch failed for {task.task_id}: {e}]")
                verified = None

    wall_clock_s = asyncio.get_running_loop().time() - t_episode_start

    denom = turn  # turns actually executed (loop variable persists post-loop)

    # Error recovery rate: of the error turns that had a following turn, what
    # fraction were followed by a non-error turn (agent got unstuck). None if
    # there were no recoverable errors.
    eligible = 0
    recovered = 0
    for i in range(len(error_flags) - 1):
        if error_flags[i]:
            eligible += 1
            if not error_flags[i + 1]:
                recovered += 1
    error_recovery_rate = (recovered / eligible) if eligible else None

    trajectory = {
        "n_valid_actions": n_valid,
        "n_invalid_actions": n_invalid,
        "n_bash_actions": n_bash,
        "n_submit_actions": n_submit,
        "n_error_observations": n_error_obs,
        "n_repeated_actions": n_repeat,
        "action_validity_rate": (n_valid / denom) if denom else 0.0,
        "error_rate": (n_error_obs / denom) if denom else 0.0,
        "error_recovery_rate": error_recovery_rate,
        "turns_to_submit": turns_to_submit,      # None if never submitted
        "cache_hit_rate_by_turn": cache_hit_rate_by_turn,
        "reusable_prefix_by_turn": reusable_prefix_by_turn,
    }

    return EpisodeResult(
        task_id=task.task_id,
        n_turns=turn,
        completed=completed,
        final_history_tokens=messages_tokens(history),
        summary=metrics_logger.summary(),
        verified=verified,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        wall_clock_s=wall_clock_s,
        trajectory=trajectory,
    )


# Default mock task used by sweep runner in Phase 1.
DEFAULT_MOCK_TASK = Task(
    task_id="mock-jwt-fix",
    description=(
        "There is a bug in src/auth/jwt_handler.py: expired JWT tokens are "
        "still being accepted as valid. Find the bug, fix it, and verify "
        "the fix with the existing test suite under tests/auth/."
    ),
)
