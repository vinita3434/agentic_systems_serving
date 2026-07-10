"""
LLM client. Two backends:

  - MockLLMClient : no GPU needed. Simulates realistic TTFT growth with
                    prompt length (prefill is ~O(n) in token count, plus a
                    constant). Maintains a fake prefix cache so the
                    cache_aware_ordering strategy can produce measurably
                    lower TTFT than the others.

  - VLLMClient   : OpenAI-compatible HTTP client to a real vLLM server.
                    Streams the response so TTFT is measured at the actual
                    first token, not at the full response.

Both expose the same async interface:
    .chat(messages, **gen_kwargs) -> CompletionResult
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


Message = dict[str, Any]


@dataclass
class CompletionResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    finish_reason: str = "stop"
    cache_hit_tokens: Optional[int] = None
    cache_hit_rate: Optional[float] = None


# ---------- mock backend ---------------------------------------------------


class MockLLMClient:
    """
    Mock LLM that returns canned SWE-agent style actions.

    TTFT model:
        ttft_ms = base + (uncached_prefix_tokens / 1000) * per_1k_ms

    A fake prefix cache stores the longest prefix of (system+task+...) seen
    so far. Calls that reuse that prefix only pay TTFT on the *new* suffix.
    cache_aware_ordering should therefore win measurably here.
    """

    BASE_TTFT_MS = 50.0
    PREFILL_MS_PER_1K = 120.0
    DECODE_MS_PER_TOKEN = 2.0

    # Hard-coded canned action sequence for the mock SWE task.
    CANNED_ACTIONS = [
        "I'll start by exploring the repo.\n```bash\nls -la\n```",
        "Let me look at the auth module.\n```bash\ncat src/auth/jwt_handler.py\n```",
        "The bug is the verify_exp=False option. I'll fix it.\n"
        "```bash\nsed -i \"s/options={'verify_exp': False}//\" src/auth/jwt_handler.py\n```",
        "Now running the tests.\n```bash\npython -m pytest tests/auth/ -v\n```",
        "All tests pass. Submitting.\n```bash\nsubmit\n```",
    ]

    def __init__(self, model: str = "mock-qwen2.5-coder-7b", seed: int = 0):
        self.model = model
        self._rng = random.Random(seed)
        self._cache_prefix: str = ""
        self._call_idx = 0

    async def chat(self, messages: list[Message],
                   max_tokens: int = 512,
                   temperature: float = 0.0,
                   **_) -> CompletionResult:
        t0 = time.perf_counter()
        prompt_text = _serialize_messages(messages)
        prompt_tokens = _estimate_tokens(prompt_text)

        cache_hit_chars = _longest_common_prefix(self._cache_prefix, prompt_text)
        cache_hit_tokens = max(1, cache_hit_chars // 4) if cache_hit_chars else 0
        uncached_tokens = max(0, prompt_tokens - cache_hit_tokens)
        cache_hit_rate = cache_hit_tokens / prompt_tokens if prompt_tokens else 0.0

        ttft_ms = self.BASE_TTFT_MS + (uncached_tokens / 1000.0) * self.PREFILL_MS_PER_1K
        ttft_ms *= 1.0 + (self._rng.random() - 0.5) * 0.05  # small jitter
        await asyncio.sleep(ttft_ms / 1000.0)

        action = self.CANNED_ACTIONS[min(self._call_idx, len(self.CANNED_ACTIONS) - 1)]
        completion_tokens = _estimate_tokens(action)
        await asyncio.sleep(completion_tokens * self.DECODE_MS_PER_TOKEN / 1000.0)

        total_ms = (time.perf_counter() - t0) * 1000.0

        # Update cache: the prompt is now the longest prefix we've seen.
        if len(prompt_text) > len(self._cache_prefix):
            self._cache_prefix = prompt_text
        self._call_idx += 1

        finish = "stop" if "submit" in action else "stop"
        return CompletionResult(
            content=action,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            finish_reason=finish,
            cache_hit_tokens=cache_hit_tokens,
            cache_hit_rate=cache_hit_rate,
        )

    def reset_cache(self) -> None:
        self._cache_prefix = ""
        self._call_idx = 0


# ---------- real vLLM backend ---------------------------------------------


class VLLMClient:
    """OpenAI-compatible streaming client. Use this once a real vLLM server
    is running (Phase 2 on RunPod)."""

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                 api_key: str = "EMPTY",
                 timeout: float = 300.0,
                 engine: str = "vllm"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.engine = engine
        self.headers = {"Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"}
        self.timeout = timeout
        # Cumulative cache counters from previous /metrics scrape, used to
        # compute per-request deltas (the cumulative rate is uninformative
        # across many requests).
        self._prev_queries: Optional[float] = None
        self._prev_hits: Optional[float] = None

    async def chat(self, messages: list[Message],
                   max_tokens: int = 512,
                   temperature: float = 0.0,
                   **gen_kwargs) -> CompletionResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **gen_kwargs,
        }
        t0 = time.perf_counter()
        ttft_ms: Optional[float] = None
        chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions",
                                     json=payload, headers=self.headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (evt.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        chunks.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    usage = evt.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)

        total_ms = (time.perf_counter() - t0) * 1000.0
        if ttft_ms is None:
            ttft_ms = total_ms  # no streamed tokens

        cache_hit_tokens, cache_hit_rate = await self._fetch_cache_stats()

        return CompletionResult(
            content="".join(chunks),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            finish_reason=finish_reason,
            cache_hit_tokens=cache_hit_tokens,
            cache_hit_rate=cache_hit_rate,
        )

    async def _fetch_cache_stats(self) -> tuple[Optional[int], Optional[float]]:
        """Per-request prefix-cache hit stats, adapted per serving engine.

        Different engines expose cache metrics under different names (and
        some as a direct ratio gauge rather than counters). CACHE_METRIC_ADAPTERS
        maps engine -> how to read them:

          "counter" mode: two cumulative counters (queries, hits). We snapshot
            them around each request and return the per-call delta ratio, since
            the cumulative rate is uninformative across many requests.
          "gauge" mode: a single instantaneous hit-rate gauge; returned as-is
            (per-request token counts are not recoverable, so tokens = None).

        vLLM / continuum / infercept share vLLM's counters. SGLang exposes its
        own; the name below is a best guess (UNVERIFIED) — confirm against a
        live `curl /metrics` on first real run. Any miss falls back to
        (None, None) so the sweep still completes.
        """
        adapter = CACHE_METRIC_ADAPTERS.get(self.engine,
                                            CACHE_METRIC_ADAPTERS["vllm"])
        mode = adapter["mode"]
        try:
            metrics_url = self.base_url.replace("/v1", "") + "/metrics"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(metrics_url)
                resp.raise_for_status()
            text = resp.text

            if mode == "gauge":
                rate = _parse_prom(text, adapter["rate"])
                if rate is not None:
                    return None, max(0.0, min(1.0, rate))
                return None, None

            # counter mode
            queries = _parse_prom(text, adapter["queries"])
            hits = _parse_prom(text, adapter["hits"])
            if queries is None or hits is None:
                return None, None

            if self._prev_queries is None:
                delta_q, delta_h = queries, hits
            else:
                delta_q = max(0.0, queries - self._prev_queries)
                delta_h = max(0.0, hits - self._prev_hits)

            self._prev_queries, self._prev_hits = queries, hits

            if delta_q > 0:
                return int(delta_h), delta_h / delta_q
        except Exception:
            pass
        return None, None


async def preflight_serving_check(base_url: str, serving_cfg: dict,
                                  timeout: float = 5.0,
                                  api_key: str = "EMPTY") -> list[str]:
    """Sanity-check that the server at base_url plausibly matches the
    serving config we were asked to run. Returns a list of human-readable
    warnings (empty == everything checks out).

    What it can verify:
      * server is reachable                         (hard error if not)
      * served model id matches serving_cfg.model    (warn on mismatch)
      * engine family: sglang vs vllm-family         (warn on mismatch)

    What it CANNOT verify reliably: vanilla vLLM vs the vllm-continuum fork
    — both speak the identical OpenAI API and expose `vllm:`-prefixed
    metrics. For the continuum config we emit an explicit reminder to
    confirm the running env, since a silent fallback to vanilla vLLM is
    exactly the failure mode this check exists to surface.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-len("/v1")]
    v1 = root + "/v1"

    expected_engine = serving_cfg.get("engine", "vllm")
    expected_family = "sglang" if expected_engine == "sglang" else "vllm"
    expected_model = serving_cfg.get("model")
    warnings: list[str] = []

    # Auth header — vLLM servers can require an API key; match what the
    # VLLMClient sends so the preflight isn't rejected with 401.
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        # 1. Reachability + served model id.
        try:
            resp = await client.get(f"{v1}/models")
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"serving preflight: no reachable server at {v1} ({e}). "
                f"Start it with ./serving/start_vllm.sh "
                f"{serving_cfg.get('name', '<config>')} before running "
                f"--backend vllm.") from e

        try:
            served = [m.get("id") for m in (resp.json().get("data") or [])]
        except Exception:
            served = []
        if expected_model and served and expected_model not in served:
            warnings.append(
                f"served model {served} != config model '{expected_model}'.")

        # 2. Engine family fingerprint.
        running_family = "unknown"
        try:
            info = await client.get(f"{root}/get_model_info")
            if info.status_code == 200:
                running_family = "sglang"
        except Exception:
            pass
        if running_family == "unknown":
            try:
                metrics = await client.get(f"{root}/metrics")
                if metrics.status_code == 200 and "vllm:" in metrics.text:
                    running_family = "vllm"
            except Exception:
                pass

        if running_family != "unknown" and running_family != expected_family:
            warnings.append(
                f"config '{serving_cfg.get('name')}' expects a "
                f"{expected_family}-family engine but the running server "
                f"fingerprints as '{running_family}'. You are likely measuring "
                f"the wrong engine.")

        # 3. vLLM forks cannot be distinguished from vanilla vLLM via the API.
        if expected_engine in ("vllm-continuum", "infercept"):
            fork = "vllm-continuum" if expected_engine == "vllm-continuum" else "INFERCEPT"
            warnings.append(
                f"config expects the {fork} fork, which is "
                f"API-indistinguishable from vanilla vLLM. Confirm the server "
                f"was started from the {fork} env (`which vllm` / check the "
                f"active venv); otherwise this run silently measures vanilla "
                f"vLLM.")

    return warnings


# Per-engine /metrics adapters. "counter" reads two cumulative counters and
# returns their per-request delta ratio; "gauge" reads one instantaneous rate.
# vLLM and its forks (continuum, infercept) inherit vLLM's counters. The
# sglang entry is a best-guess metric name — verify with `curl :30000/metrics`.
CACHE_METRIC_ADAPTERS: dict[str, dict[str, str]] = {
    "vllm": {"mode": "counter",
             "queries": "vllm:prefix_cache_queries_total",
             "hits": "vllm:prefix_cache_hits_total"},
    "vllm-continuum": {"mode": "counter",
                       "queries": "vllm:prefix_cache_queries_total",
                       "hits": "vllm:prefix_cache_hits_total"},
    "infercept": {"mode": "counter",
                  "queries": "vllm:prefix_cache_queries_total",
                  "hits": "vllm:prefix_cache_hits_total"},
    # UNVERIFIED metric name — confirm on first real SGLang run.
    "sglang": {"mode": "gauge",
               "rate": "sglang:cache_hit_rate"},
}


def _parse_prom(text: str, metric: str) -> Optional[float]:
    for line in text.splitlines():
        if line.startswith(metric):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                return None
    return None


# ---------- helpers --------------------------------------------------------


def _serialize_messages(messages: list[Message]) -> str:
    return "\n".join(f"<{m.get('role','user')}>\n{m.get('content','')}" for m in messages)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def make_client(backend: str, **kwargs) -> Any:
    if backend == "mock":
        return MockLLMClient(**kwargs)
    if backend == "vllm":
        return VLLMClient(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
