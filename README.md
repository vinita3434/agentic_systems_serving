# agentic_systems_serving

A research harness for ablation experiments on agentic LLM serving.
Targets the question: **does workflow-aware serving compound with
orchestration that preserves prefixes, or do orchestration choices
invalidate serving-layer wins?**

Workload: SWE-agent over SWE-bench (full split) on
Qwen2.5-Coder-7B-Instruct, served via vLLM / SGLang / vllm-continuum
on a RunPod A100 / H100.

---

## 1. Scope

The orchestration layer (how an agent loop assembles its prompt across
turns) and the serving layer (how an inference engine manages KV cache,
batching, and attention) are typically studied independently. This
harness is built to measure their interaction under a realistic agentic
workload.

The thesis under test: **orchestration choices and serving choices are
not independent.** A given serving optimization (prefix caching, CPU
spill, RadixAttention, workflow-aware retention) has an assumed prompt
shape across turns; a given orchestration strategy (summarization,
sliding window, tool output compression, prefix-stable reordering)
rewrites that shape. The harness measures whether the assumed
serving-layer wins survive each orchestration choice — and whether
explicit prefix preservation in orchestration compounds with
workflow-aware serving.

The full study is structured as **three layers of isolation experiments
plus a targeted set of cross-layer interactions** (see §4).

---

## 2. Architecture

The harness is built on **Option C** of the SWE-agent integration design
space: a custom agent loop that owns every per-turn decision (system
prompt, context assembly, LLM call, action parsing, observation
formatting, history management), while delegating the Docker / bash /
file-editing environment to SWE-agent's `SWEEnv` and the task data to
the SWE-bench dataset. Patch evaluation shells out to
`swebench.harness.run_evaluation`.

```
┌──────────────── harness/ (this repo) ───────────────┐
│                                                     │
│  run_experiment.py  (sweep runner / design modes)   │
│           │                                         │
│           ▼                                         │
│  agent_loop.py    ──┬─ context_manager.py           │
│                     │   (5 orchestration strategies)│
│                     ├─ llm_client.py                │
│                     │   (mock + vLLM streaming +    │
│                     │    /metrics scraping)         │
│                     ├─ tools.py / sweenv_tools.py   │
│                     │   (action parser + SWEEnv     │
│                     │    adapter + patch evaluator) │
│                     └─ metrics_logger.py            │
│                         (per-turn JSONL +           │
│                          per-episode JSONL)         │
└───────────────────────│─────────────────────────────┘
                        │ OpenAI-compatible HTTP
                        ▼
              ┌─────────────────────┐
              │ vllm / sglang /     │
              │ vllm-continuum      │
              └─────────────────────┘
                        │
                        ▼   (SWEEnv-managed)
              ┌─────────────────────┐
              │ per-task Docker     │
              │ container w/ repo   │
              │ at base_commit      │
              └─────────────────────┘
```

What we own:
- The agent loop, system prompt, context assembly per turn.
- The orchestration strategies (5 of them, decomposed over 3 sub-axes:
  context management, prompt assembly, tool output methodology).
- The serving abstraction (engine-agnostic YAML configs, engine
  dispatch in `start_vllm.sh`).
- The LLM client (streams responses to capture true TTFT; scrapes
  vLLM's `/metrics` for prefix-cache counters).
- The metrics pipeline (per-turn and per-episode JSONL, plus a sweep
  summary JSON).

What we delegate:
- Docker container lifecycle, bash execution, file editing
  → `sweagent.environment.swe_env.SWEEnv`.
- Task data (repo URL, base commit, problem statement, hidden tests)
  → `princeton-nlp/SWE-bench` dataset via `harness/swebench_tasks.py`.
- Patch verification → `swebench.harness.run_evaluation`.

---

## 3. Components

### Orchestration (`configs/orchestration/` + `harness/context_manager.py`)

Five strategies, decomposed into three orthogonal sub-axes. The cube
origin is `full_context` (full history, chronological, raw observations);
each non-origin strategy is a one-axis variation from there.

| Strategy | Context mgmt | Prompt assembly | Tool output |
|---|---|---|---|
| `full_context` | full (origin) | chronological (origin) | raw (origin) |
| `sliding_window` | **drop turns older than window** | chronological | raw |
| `summarization` | **LLM-summarize old turns when over threshold** | chronological | raw |
| `cache_aware_ordering` | full | **stable prefix first, volatile tail last** | raw |
| `tool_output_compression` | full | chronological | **head + tail + drop marker for long outputs** |

### Serving (`configs/serving/`)

Five configurations, treating cache management as the independent
variable. All configs target `Qwen/Qwen2.5-Coder-7B-Instruct`; the
`engine` field dispatches to vanilla vLLM, the
[vllm-continuum](https://github.com/Hanchenli/vllm-continuum) fork, or
SGLang.

| Config | Engine | Caching mechanism | Role |
|---|---|---|---|
| `cache_off` | vllm | `enable_prefix_caching: false` | Empirical floor — what KV reuse is worth |
| `vllm_lru` | vllm | Standard prefix cache, LRU eviction | Production-quality baseline |
| `vllm_lmcache` | vllm | Prefix cache + CPU-tier spill via `LMCacheConnectorV1` | Tiered cache under pressure |
| `sglang` | sglang | RadixAttention prefix-tree sharing | Alternative engine, workflow-blind |
| `vllm_continuum` | vllm-continuum | Workflow-aware KV TTL (tool-output blocks retained longer) | Workflow-aware serving |

### Agent loop (`harness/agent_loop.py`)

```
history = [system_prompt, task]
for turn in 1..max_turns:
    assembled    = context_manager.assemble(strategy, history, params)
    result       = llm_client.chat(assembled.messages)            # streams; captures TTFT
    metrics_logger.log(TurnMetrics(...))                          # per-turn JSONL
    history.append(assistant(result.content))
    action       = parse_action(result.content)
    observation  = tools.execute(action)                          # via SWEEnv
    history.append(observation)
    break if observation.is_terminal
if completed and tools.evaluate_patch:                            # SWEEnvTools only
    verified = tools.evaluate_patch()                             # swebench harness
metrics_logger.log_episode(EpisodeResult(..., verified=verified)) # per-episode JSONL
```

Tool execution runs in an executor so the blocking `SWEEnv.communicate()`
does not stall the async LLM client.

### Sweep runner (`experiments/run_experiment.py`)

CLI flags:
- `--backend {mock, vllm}` — `mock` is GPU-free; `vllm` hits a real
  server (any OpenAI-compatible engine).
- `--task-source {mock, swebench} --swebench-split {full, lite, verified} --task-limit N --task-id ID`.
- `--gpu-class {A100, H100}` — filters serving configs via the
  `gpu_classes` field in each YAML.
- `--design <mode>` — see §4.

---

## 4. Experimental Design

Eight design modes total. Five are layered; three are legacy
(grid / ofat / ofat+interactions) preserved for backward compatibility.

### Layer 1 — `serving_sweep` (5 cells)

Vary serving; hold orchestration at the prefix-preserving anchor
`cache_aware_ordering`. Measures what each serving config buys when
orchestration is already trying to help.

### Layer 2 — three sub-axis orchestration sweeps (3 + 2 + 2 = 7 cells)

Vary one orchestration sub-axis at a time; hold serving at the standard
prefix-caching baseline `vllm_lru`.

- `context_mgmt_sweep`: `{full_context, sliding_window, summarization}` × `vllm_lru`
- `prompt_assembly_sweep`: `{full_context, cache_aware_ordering}` × `vllm_lru`
- `tool_output_sweep`: `{full_context, tool_output_compression}` × `vllm_lru`

The `(full_context, vllm_lru)` cell — the joint origin — is re-measured
in every L2 sweep as a noise-floor anchor.

### Layer 3 — `interactions` (4 cells, H1–H4)

Each cell is the one novel corner of a 2×2 whose other three corners
come for free from L1 and L2 sweeps.

| Hypothesis | Test cell | Mechanism under test | Prediction |
|---|---|---|---|
| **H1** `summarization_kills_continuum` | (summarization, vllm_continuum) | Summarization rewrites old turns → tool-output bytes no longer in prompt → Continuum's tool-aware TTL has nothing to retain | Continuum's L1 win evaporates |
| **H2** `sliding_window_kills_radix` | (sliding_window, sglang) | Sliding window drops the early-turn prefix that anchors RadixAttention's shared tree branches | SGLang's shared-prefix advantage collapses |
| **H3** `compression_breaks_lmcache` | (tool_output_compression, vllm_lmcache) | Compression rewrites observation bytes → CPU-restored KV blocks hash-mismatch the current prompt | LMCache CPU-tier hit rate drops to ~0 |
| **H4** `continuum_independent_of_orchestration` | (full_context, vllm_continuum) | Tests whether Continuum's win is independent of orchestration | Either Continuum still helps (independent) or its L1 win was orchestration-driven |

Total cost: **5 + 3 + 2 + 2 + 4 = 16 cells**, each answering a single
named question, vs 25 for a brute-force grid.

---

## 5. Status

| Component | State |
|---|---|
| Custom agent loop | ✅ End-to-end working |
| 5 orchestration strategies | ✅ Implemented + validated in mock |
| 5 serving configs (cache-focused) | ✅ YAMLs + engine-dispatch launcher |
| 5 layered design modes + 3 legacy modes | ✅ All 8 validated in mock |
| Per-turn metrics (TTFT, latency, tokens, cache hits) | ✅ JSONL |
| Per-episode metrics (`verified` field) | ✅ JSONL — populates `null` in mock; calls SWE-bench evaluator in real runs |
| vLLM `/metrics` cache-hit scraping | ✅ Wired in `VLLMClient` |
| RunPod bootstrap script | ✅ Installs vanilla vLLM, sweagent, swebench, lmcache. Notes-only for SGLang + vllm-continuum (conflicting installs). |
| RunPod deployment guide | ✅ `RUNPOD.md` |
| **Pod provisioned** | ⏳ Not yet |
| **Real vLLM / SGLang / Continuum runs** | ⏳ Not yet |
| **SWE-bench evaluation exercised end-to-end** | ⏳ Not yet (code wired, not executed) |

Mock-mode sweep across all five new design modes is reproducible from
`PYTHONPATH=. python experiments/run_experiment.py --design <mode>`.

---

## 6. Pending — Agent Evaluation

Patch evaluation is **wired but not yet exercised**. Specifically:

1. **`SWEEnvTools.evaluate_patch()`** shells out to
   `python -m swebench.harness.run_evaluation` with a one-row
   predictions JSON, then parses the harness's stdout for
   `"resolved"` + the instance ID. This heuristic was chosen because
   the harness's machine-readable output format has shifted across
   versions; the safer parse is to read the per-instance log file
   produced under `logs/run_evaluation/<run_id>/`, but that requires
   confirming the path layout for the swebench version installed on
   the pod. **Action item:** on first real run, capture an example
   output and harden the parser against it.

2. **Cost of evaluation.** Each patch evaluation spins up a fresh
   Docker container, applies the patch, installs deps, and runs the
   hidden test suite. Per-episode cost is **30–120 s typical, up to
   600 s for repos with heavy install steps**. With a `--task-limit 5`
   sweep and 16 cells, evaluation alone adds ~40–60 min on top of
   inference. Action item: decide whether to evaluate every episode
   or sample; current default is "always when `task_source=swebench`".

3. **Ground-truth comparison not yet computed.** The harness logs
   `verified: bool` but not the *diff* between the agent's patch and
   the gold patch in `SWEBenchTask.patch`. That comparison is useful
   for debugging cases where the agent's patch passes the tests but
   differs structurally from the gold. Action item: add an optional
   `patch_diff` field if needed.

4. **Run-id collision risk.** `evaluate_patch()` derives `run_id` from
   `instance_id` alone. Running the same instance in multiple cells
   within one sweep will collide. Action item: include the cell tag
   (`sweep<id>__<role>__<orch>__<serv>`) in `run_id`.

---

## 7. Evaluation Metrics

Per (orchestration, serving) cell, we report the following — split
into what is **already captured** (per-turn / per-episode JSONL) and
what will be **derived** during analysis.

### 7.1 Captured per turn (`<exp>.jsonl`)

| Metric | Definition | Layer this exposes |
|---|---|---|
| `ttft_ms` | Time to first token, measured at the streaming boundary | Serving (prefill) |
| `total_latency_ms` | End-to-end request latency | Combined |
| `prompt_tokens` | Tokens the model received | Combined |
| `completion_tokens` | Tokens the model produced | Combined |
| `context_tokens` | Same as `prompt_tokens`, named separately for orchestration accounting | Orchestration (effective) |
| `raw_history_tokens` | History size *before* orchestration ran | Orchestration (pre-effect) |
| `cache_hit_rate` | From vLLM `/metrics`: `prefix_cache_hits / prefix_cache_queries` | Serving |
| `cache_hit_tokens` | Tokens served from prefix cache this request | Serving |
| `finish_reason` | `stop` / `length` / `tool_calls` | Combined |

### 7.2 Captured per episode (`<exp>__episodes.jsonl`)

| Metric | Definition |
|---|---|
| `n_turns` | Turns until `submit` or `max_turns` |
| `completed` | Did the agent emit `submit`? |
| `final_history_tokens` | Total history size at episode end |
| `verified` | `True` if patch passes SWE-bench tests, `False` if not, `None` for mock / evaluation skip / parse failure |

### 7.3 Derived per cell (sweep summary)

Already computed by `MetricsLogger.summary()`:
- `avg_ttft_ms`, `p50_ttft_ms`, `p95_ttft_ms`
- `avg_total_latency_ms`
- `avg_prompt_tokens`, `avg_context_tokens`, `avg_raw_history_tokens`
- `compression_ratio` = `avg(context_tokens) / avg(raw_history_tokens)`
- `total_prompt_tokens`, `total_completion_tokens`
- `avg_cache_hit_rate`

To be added in the analysis pass (downstream of the JSONLs):
- **Task success rate**: `count(verified=True) / count(episodes)` per cell.
- **Cost per verified task**:
  `(total_prompt_tokens · $/Mtok_in + total_completion_tokens · $/Mtok_out + wall_clock · $/GPU·hr) / count(verified=True)`.
  This is the punchline economics metric — collapses serving and
  orchestration effects into a single number per cell, and answers
  the question a deployment team actually cares about.
- **TTFT-vs-context-length curves**: `ttft_ms` plotted against
  `prompt_tokens`, one line per cell. Reveals whether a serving config
  scales differently across the range of context sizes an orchestration
  strategy produces.
- **Cache-hit growth curves**: `cache_hit_rate` plotted against turn
  index within an episode. Tells you whether the orchestration's
  prefix stability is being exploited by the serving config (rising
  curve) or not (flat curve).
- **2×2 deltas for H1–H4**: for each hypothesis, compute the
  per-cell-vs-anchor TTFT and verified-rate deltas, then test whether
  the L3 cell's delta matches the predicted direction.

### 7.4 What "performance" means in this study

Three orthogonal axes of performance, all needed for the thesis:

1. **Serving efficiency** — TTFT, total latency, cache hit rate.
   What does each serving config achieve under a given orchestration?
2. **Orchestration efficiency** — `compression_ratio`, context tokens
   per turn, raw vs effective history size.
   What does each orchestration strategy save (or cost) before the
   model even sees the prompt?
3. **End-to-end task success** — `verified` rate per cell.
   Does the combined system actually solve real SWE-bench tasks?
   None of the above metrics matter if the agent doesn't ship a
   correct patch.

The cost-per-verified-task metric is the integration of all three —
it is what we will report as the headline number per cell, with the
underlying TTFT / cache-hit / compression numbers as the explanation
for *why* a given cell wins or loses.
