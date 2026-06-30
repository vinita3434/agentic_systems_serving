# RUNBOOK — running the harness on RunPod

Operational guide for running real (`--backend vllm`) sweeps on a RunPod
A100/H100. Strategy: **one serving engine up at a time**, each iteration
sweeping all 5 orchestration strategies against it. After 5 iterations you
have the full 5×5 (serving × orchestration) grid, each cell served by the
correct engine.

For *what* the experiment measures, see `README.md`. This file is just the
operational steps.

---

## 0. Concepts (30-second refresher)

- The harness talks **HTTP** to the engine. By default it derives the URL
  from the serving config's own `port` (each engine binds a distinct port),
  so `--serving <name>` automatically targets the right server. Override
  with `--vllm-base-url` only if you relocate a server. It does **not** start
  engines — `serving/start_vllm.sh` does that (and reads the same `port:`
  field, so server and harness always agree).
- `--serving <name>` only *labels* the metrics + selects launch flags. It
  does **not** switch the running engine. So you must start the matching
  engine yourself before each sweep.
- The startup **preflight check** confirms the running engine matches the
  `--serving` config and prints `ok` or a ⚠. It cannot tell vanilla vLLM
  from the continuum fork (identical API) — it reminds you to confirm that
  env manually. Disable with `--skip-serving-check`.
- vanilla vLLM, the continuum fork, and SGLang all collide on the `vllm` /
  server binary, so each lives in its **own venv**. Swap envs, don't
  reinstall.

---

## 1. One-time pod setup

Create a pod (PyTorch / CUDA 12.x template, A100 or H100, volume on
`/workspace`), SSH in, then:

```bash
# clone + base deps (vanilla vLLM, sweagent, swebench, lmcache, datasets)
REPO_URL=https://github.com/<you>/agentic_systems_serving.git bash setup_runpod.sh
huggingface-cli login            # Qwen2.5-Coder is a gated repo
```

Build the three engine environments (continuum and sglang *must* be
separate; keeping vanilla separate too avoids surprises):

```bash
# vanilla vLLM  → cache_off, vllm_lru, vllm_lmcache
python3 -m venv ~/envs/vllm
source ~/envs/vllm/bin/activate && pip install 'vllm>=0.6.0' lmcache && deactivate

# vllm-continuum fork → vllm_continuum
python3 -m venv ~/envs/continuum
source ~/envs/continuum/bin/activate
git clone https://github.com/Hanchenli/vllm-continuum ~/vllm-continuum
pip install -e ~/vllm-continuum
which vllm                        # MUST resolve into ~/envs/continuum
deactivate

# SGLang → sglang
python3 -m venv ~/envs/sglang
source ~/envs/sglang/bin/activate && pip install 'sglang[all]' && deactivate
```

---

## 2. Smoke test (do this first, before committing GPU time)

Real SWE-bench evaluation is slow (30–120 s/episode, Docker per task). Run a
single task end-to-end on the cheapest engine first:

```bash
# window 1: engine
tmux new -s engine
source ~/envs/vllm/bin/activate
cd /workspace/agentic_systems_serving
./serving/start_vllm.sh vllm_lru        # wait for "startup complete", then Ctrl-b d

# window 2: harness, ONE task, ONE orchestration
# (no --vllm-base-url needed: vllm_lru's port 8001 is read from its YAML)
cd /workspace/agentic_systems_serving
PYTHONPATH=. python experiments/run_experiment.py \
  --backend vllm \
  --layer custom --serving vllm_lru --orchestration full_context \
  --task-source swebench --task-limit 1 --gpu-class A100
```

If that produces a `verified` value and a summary file, the full pipeline
(serving + Docker + swebench eval) works. Then proceed.

---

## 3. The main cycle — repeat 5 times, once per serving config

For each row in the table below: **start → sweep → stop.**

```bash
# ── window 1: start the engine ──
tmux new -s engine
source ~/envs/<ENV>/bin/activate
cd /workspace/agentic_systems_serving
./serving/start_vllm.sh <SERVING>       # wait for startup, Ctrl-b d to detach

# ── window 2: sweep ALL orchestrations against it ──
# No --vllm-base-url: the port is read from the serving config's YAML.
cd /workspace/agentic_systems_serving
PYTHONPATH=. python experiments/run_experiment.py \
  --backend vllm \
  --layer orchestration --serving <SERVING> \
  --task-source swebench --task-limit 5 --gpu-class A100

# ── stop the engine before the next config (frees port + GPU) ──
tmux kill-session -t engine
```

| # | `<ENV>`    | `<SERVING>`      | Port (auto, from YAML) |
|---|------------|------------------|------------------------|
| 1 | vllm       | `cache_off`      | 8000                   |
| 2 | vllm       | `vllm_lru`       | 8001                   |
| 3 | vllm       | `vllm_lmcache`   | 8002                   |
| 4 | continuum  | `vllm_continuum` | 8003                   |
| 5 | sglang     | `sglang`         | 30000                  |

Each iteration = 5 orchestration cells × `--task-limit` tasks. With
`--task-limit 5` that's 25 episodes per serving config, 125 total.

> Each config now binds a **distinct** port, so they can't collide — and in
> principle you could run several at once (subject to GPU memory: each
> reserves ~90%, so concurrent servers need multiple GPUs or a lower
> `gpu_memory_utilization`). For the sequential workflow, still kill each
> server before starting the next to free the GPU.

---

## 4. CLI lever reference

| Flag | Meaning |
|---|---|
| `--backend {mock,vllm}` | `mock` = no GPU, simulated; `vllm` = real server. |
| `--vllm-base-url URL` | Override the server URL. Default: derived from the serving config's `port`. |
| `--layer {serving,orchestration,interactions,custom}` | Which compartment to run. Overrides `--design`. |
| `--serving NAME [NAME...]` | Restrict/override serving config(s). |
| `--orchestration NAME [NAME...]` | Restrict/override orchestration config(s). |
| `--axis {context_mgmt,prompt_assembly,tool_output}` | In `--layer orchestration`, sweep one sub-axis only. |
| `--interaction H1 [H2...]` | In `--layer interactions`, run named hypotheses only. |
| `--task-source {mock,swebench}` | Canned task vs real dataset instances. |
| `--task-limit N` | Tasks (episodes) per cell. Default 5. |
| `--task-id ID` | Single SWE-bench instance; overrides `--task-limit`. |
| `--max-turns N` | Max agent turns per episode. Default 20. |
| `--gpu-class {A100,H100}` | Skip serving configs incompatible with this GPU. |
| `--skip-serving-check` | Skip the engine-match preflight (vllm backend). |
| `--gpu-hourly-usd` / `--input-usd-per-mtok` / `--output-usd-per-mtok` | Cost-accounting rates. |

Common patterns:

```bash
# All orchestrations against the active engine (the main-cycle command)
... --layer orchestration --serving <S>

# One exact cell
... --layer custom --serving <S> --orchestration <O>

# Just one orchestration sub-axis
... --layer orchestration --serving <S> --axis context_mgmt

# Interaction hypotheses (need the matching engine up for each)
... --layer interactions --interaction H1 H3
```

---

## 5. Output

Per sweep, written to `experiments/results/`:

- `sweep<id>__<role>__<orch>__<serving>.jsonl` — per-turn metrics.
- `sweep<id>__<role>__<orch>__<serving>__episodes.jsonl` — per-episode
  (includes `verified`, cost fields).
- `sweep<id>_summary.json` — per-cell rollup (ttft percentiles, compression,
  cache hit rate, task success rate, cost-per-verified-task).

After all 5 iterations you have 5 summary files covering the full grid, each
cell served by the correct engine — the dataset the H1–H4 hypotheses need.

---

## 6. Gotchas

- **Wrong engine, silent.** `--serving vllm_continuum` against a vanilla
  vLLM server "works" but measures vanilla. The preflight ⚠ is your guard;
  also check `which vllm` in the continuum env.
- **Ports are per-config and automatic.** cache_off=8000, vllm_lru=8001,
  vllm_lmcache=8002, vllm_continuum=8003, sglang=30000. The harness derives
  the URL from the config, so you normally don't pass `--vllm-base-url` at
  all. If you change a `port:` in a YAML, both the server and the harness
  pick it up automatically.
- **Model gating.** If requests 401/403, re-run `huggingface-cli login`.
- **tmux.** Always start servers in tmux; an SSH drop otherwise kills the
  engine mid-sweep.
- **Eval cost.** Each verified episode spins a fresh Docker container; budget
  30–120 s/episode (up to 600 s for heavy repos).
