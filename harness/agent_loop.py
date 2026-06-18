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

from harness.context_manager import assemble, Message, messages_tokens
from harness.llm_client import CompletionResult
from harness.metrics_logger import MetricsLogger, TurnMetrics
from harness.tools import Action, Observation, parse_action


SYSTEM_PROMPT = (
    "You are an autonomous software engineering agent. You have access to a "
    "bash shell. Respond with a single bash command in a fenced ```bash``` "
    "block. When the task is complete, run `submit`. Think step by step, "
    "verify your changes with tests, and keep responses concise."
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
        {"role": "user", "content": f"<task>\n{task.description}\n</task>",
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

    for turn in range(1, max_turns + 1):
        assembled = assemble(strategy, history, params,
                             summarizer=_sync_summarizer if strategy == "summarization" else None)

        result: CompletionResult = await llm_client.chat(
            assembled.messages, max_tokens=512, temperature=0.0
        )

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
            finish_reason=result.finish_reason,
        ))

        last_finish_reason = result.finish_reason
        history.append(_format_assistant(result.content))

        action = parse_action(result.content)
        if action is None:
            history.append({"role": "user",
                            "content": "<output>\n[no valid action found]\n</output>",
                            "meta": {"kind": "observation"}})
            continue

        # tools.execute may block (SWEEnv runs bash in a docker container).
        # Run it off the event loop so we don't stall the LLM client.
        observation = await asyncio.get_running_loop().run_in_executor(
            None, tools.execute, action)
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

    return EpisodeResult(
        task_id=task.task_id,
        n_turns=turn,
        completed=completed,
        final_history_tokens=messages_tokens(history),
        summary=metrics_logger.summary(),
        verified=verified,
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
