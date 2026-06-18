"""
Context management strategies (the orchestration layer).

Each strategy is a callable that takes the full message history and config
params, and returns a transformed message list ready to send to the LLM,
plus accounting for how many tokens were saved.

Message format follows OpenAI chat completion conventions:
    {"role": "system" | "user" | "assistant", "content": str}

In this harness, we tag observation messages with role="user" (since that's
how SWE-agent's underlying chat API represents tool results when not using
the OpenAI tool-calls protocol). The strategy code distinguishes "task"
vs "observation" by message metadata stored under the "meta" key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


Message = dict[str, Any]


@dataclass
class AssembledContext:
    messages: list[Message]
    raw_history_tokens: int
    context_tokens: int
    strategy: str
    notes: dict[str, Any]


def estimate_tokens(text: str) -> int:
    """Cheap 4-chars-per-token approximation. Good enough for relative
    comparisons across strategies. Swap with tiktoken later if needed."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


# ---------- strategies -----------------------------------------------------


def _full_context(history: list[Message], params: dict) -> AssembledContext:
    tok = messages_tokens(history)
    return AssembledContext(
        messages=list(history),
        raw_history_tokens=tok,
        context_tokens=tok,
        strategy="full_context",
        notes={},
    )


def _sliding_window(history: list[Message], params: dict) -> AssembledContext:
    raw_tok = messages_tokens(history)
    window = int(params.get("window_turns", 6))
    keep_system = bool(params.get("always_keep_system", True))
    keep_task = bool(params.get("always_keep_task", True))

    system = [m for m in history if m.get("role") == "system"] if keep_system else []
    # The "task" is the first non-system message.
    task: list[Message] = []
    if keep_task:
        for m in history:
            if m.get("role") != "system":
                task = [m]
                break

    non_system = [m for m in history if m.get("role") != "system"]
    # Drop the task message from the tail consideration if we kept it.
    tail_pool = non_system[1:] if (keep_task and non_system) else non_system

    # window_turns means N (assistant, observation) pairs => 2N messages
    tail = tail_pool[-window * 2 :] if window > 0 else tail_pool

    assembled = system + task + tail
    return AssembledContext(
        messages=assembled,
        raw_history_tokens=raw_tok,
        context_tokens=messages_tokens(assembled),
        strategy="sliding_window",
        notes={"window_turns": window, "kept_messages": len(assembled)},
    )


def _cache_aware_ordering(history: list[Message], params: dict) -> AssembledContext:
    """
    Reorder so the prefix vLLM sees stays maximally stable across turns.

    Layout:
        [system] [task] [pinned_initial_observations] [stable file segments]
        [volatile tail (last N turn-pairs verbatim, original order)]

    'Stable' = messages containing file markers (file:, ``` fences, <file>)
    that are unlikely to change turn-to-turn. 'Volatile' = recent
    assistant actions and short observations.
    """
    raw_tok = messages_tokens(history)
    volatile_window = int(params.get("volatile_window_turns", 4))
    pin_initial = int(params.get("pin_initial_observations", 2))
    markers = params.get("stable_markers", ["file:", "```", "<file>"])
    marker_re = re.compile("|".join(re.escape(m) for m in markers))

    system = [m for m in history if m.get("role") == "system"]
    non_system = [m for m in history if m.get("role") != "system"]
    if not non_system:
        return AssembledContext(messages=list(history), raw_history_tokens=raw_tok,
                                context_tokens=raw_tok, strategy="cache_aware_ordering",
                                notes={})

    task = non_system[:1]
    rest = non_system[1:]
    tail_size = volatile_window * 2
    tail = rest[-tail_size:] if tail_size > 0 else []
    middle = rest[:-tail_size] if tail_size > 0 else rest

    # Pinned initial observations: first `pin_initial` messages with role=user
    # in `middle` (these are typically the first file reads / repo listings).
    pinned: list[Message] = []
    leftover: list[Message] = []
    pin_count = 0
    for m in middle:
        if pin_count < pin_initial and m.get("role") == "user":
            pinned.append(m)
            pin_count += 1
        else:
            leftover.append(m)

    stable, volatile_mid = [], []
    for m in leftover:
        content = str(m.get("content", ""))
        if marker_re.search(content):
            stable.append(m)
        else:
            volatile_mid.append(m)

    # Final order: system, task, pinned, stable, volatile_mid, tail.
    # Stable and pinned segments live at the front so prefix cache hits
    # extend further into the prompt.
    assembled = system + task + pinned + stable + volatile_mid + tail
    return AssembledContext(
        messages=assembled,
        raw_history_tokens=raw_tok,
        context_tokens=messages_tokens(assembled),
        strategy="cache_aware_ordering",
        notes={
            "pinned": len(pinned),
            "stable": len(stable),
            "volatile_mid": len(volatile_mid),
            "tail": len(tail),
        },
    )


def _tool_output_compression(history: list[Message], params: dict) -> AssembledContext:
    raw_tok = messages_tokens(history)
    max_chars = int(params.get("max_observation_chars", 2000))
    keep_first = int(params.get("keep_first_lines", 20))
    keep_last = int(params.get("keep_last_lines", 20))
    preserve_stderr = bool(params.get("preserve_stderr", True))

    out: list[Message] = []
    n_compressed = 0
    for m in history:
        if _is_observation(m):
            new_msg, compressed = _compress_observation(m, max_chars, keep_first,
                                                        keep_last, preserve_stderr)
            out.append(new_msg)
            if compressed:
                n_compressed += 1
        else:
            out.append(m)

    return AssembledContext(
        messages=out,
        raw_history_tokens=raw_tok,
        context_tokens=messages_tokens(out),
        strategy="tool_output_compression",
        notes={"observations_compressed": n_compressed},
    )


def _is_observation(msg: Message) -> bool:
    meta = msg.get("meta") or {}
    if meta.get("kind") == "observation":
        return True
    # Heuristic fallback: user-role messages that look like tool output blocks
    content = msg.get("content", "")
    return msg.get("role") == "user" and isinstance(content, str) and (
        content.startswith("<output>") or content.startswith("```")
    )


def _compress_observation(msg: Message, max_chars: int, keep_first: int,
                          keep_last: int, preserve_stderr: bool) -> tuple[Message, bool]:
    content = msg.get("content", "")
    if not isinstance(content, str) or len(content) <= max_chars:
        return msg, False

    lines = content.split("\n")
    if len(lines) <= keep_first + keep_last:
        return msg, False

    stderr_lines: list[str] = []
    if preserve_stderr:
        stderr_lines = [ln for ln in lines if re.search(r"(error|traceback|exception)",
                                                        ln, re.IGNORECASE)]

    head = "\n".join(lines[:keep_first])
    tail = "\n".join(lines[-keep_last:])
    dropped = len(lines) - keep_first - keep_last
    stderr_block = ("\n[preserved errors]\n" + "\n".join(stderr_lines[:20])
                    if stderr_lines else "")
    compressed = (
        f"{head}\n... [{dropped} lines omitted by tool_output_compression] ...\n"
        f"{tail}{stderr_block}"
    )
    return {**msg, "content": compressed, "meta": {**(msg.get("meta") or {}),
                                                   "compressed": True}}, True


def _summarization(history: list[Message], params: dict,
                   summarizer=None) -> AssembledContext:
    """
    If history exceeds trigger_token_threshold, collapse old turns into a
    single summary message produced by the LLM. `summarizer` is an injected
    callable so the agent loop can supply its LLM client (or a mock).

    summarizer signature: (messages_to_summarize: list[Message]) -> str
    """
    raw_tok = messages_tokens(history)
    threshold = int(params.get("trigger_token_threshold", 8000))
    keep_recent = int(params.get("keep_recent_turns", 4)) * 2

    if raw_tok < threshold:
        return AssembledContext(messages=list(history), raw_history_tokens=raw_tok,
                                context_tokens=raw_tok, strategy="summarization",
                                notes={"triggered": False})

    system = [m for m in history if m.get("role") == "system"]
    non_system = [m for m in history if m.get("role") != "system"]
    if len(non_system) <= keep_recent + 1:
        return AssembledContext(messages=list(history), raw_history_tokens=raw_tok,
                                context_tokens=raw_tok, strategy="summarization",
                                notes={"triggered": False, "reason": "too_short"})

    task = non_system[:1]
    middle = non_system[1:-keep_recent] if keep_recent > 0 else non_system[1:]
    recent = non_system[-keep_recent:] if keep_recent > 0 else []

    if not middle:
        return AssembledContext(messages=list(history), raw_history_tokens=raw_tok,
                                context_tokens=raw_tok, strategy="summarization",
                                notes={"triggered": False})

    summary_text = (summarizer or _mock_summarize)(middle)
    summary_msg: Message = {
        "role": "user",
        "content": f"[CONTEXT SUMMARY — {len(middle)} prior turns collapsed]\n{summary_text}",
        "meta": {"kind": "summary"},
    }
    assembled = system + task + [summary_msg] + recent
    return AssembledContext(
        messages=assembled,
        raw_history_tokens=raw_tok,
        context_tokens=messages_tokens(assembled),
        strategy="summarization",
        notes={"triggered": True, "summarized_turns": len(middle)},
    )


def _mock_summarize(messages: list[Message]) -> str:
    n = len(messages)
    actions = sum(1 for m in messages if m.get("role") == "assistant")
    return (f"Mock summary of {n} prior messages ({actions} agent actions). "
            f"Earlier work explored the repository, identified relevant files, "
            f"and made initial edits. Recent turns retained verbatim below.")


# ---------- registry & entry point -----------------------------------------


STRATEGIES = {
    "full_context": _full_context,
    "sliding_window": _sliding_window,
    "cache_aware_ordering": _cache_aware_ordering,
    "tool_output_compression": _tool_output_compression,
    "summarization": _summarization,
}


def assemble(strategy: str, history: list[Message], params: dict,
             summarizer=None) -> AssembledContext:
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown orchestration strategy: {strategy}. "
                         f"Available: {sorted(STRATEGIES)}")
    if strategy == "summarization":
        return _summarization(history, params, summarizer=summarizer)
    return fn(history, params)
