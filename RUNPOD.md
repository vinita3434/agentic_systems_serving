# RunPod Deployment Guide

End-to-end steps to take this harness from local mock to real SWE-bench
runs across the three-layer experimental design, using Qwen2.5-Coder-7B-Instruct
on a RunPod A100.

---

## 1. Provision the pod

In the RunPod UI:

- **GPU**: A100 80GB SXM (or PCIe). H100 also works — both fp8 paths
  remain available.
- **Template**: any official "PyTorch 2.x / CUDA 12.x" image. Confirm
  Python >= 3.10.
- **Volume**: at least 100 GB. SWE-bench Docker images + Qwen2.5-Coder
  weights (~15 GB) + per-task test containers add up.
- **Expose ports**:
  - `8000` (vLLM / Continuum default)
  - `30000` (SGLang default — only needed if you'll run the sglang serving config)
  - `22` (SSH) is automatic.
- **Container disk**: 40 GB+

## 2. Get the harness onto the pod

```bash
# from your laptop:
scp -r -P <PORT> agentic_systems_serving root@<POD-IP>:/workspace/
```

## 3. Run the bootstrap

```bash
ssh root@<POD-IP>
cd /workspace/agentic_systems_serving
bash ./serving/setup_runpod.sh
```

This installs: Docker (for SWEEnv), vanilla vLLM, sweagent, swebench,
datasets, lmcache. It prints — but does NOT auto-run — the install steps
for the optional engines (vllm-continuum fork and SGLang). See the
"Switching serving engines" section below.

## 4. Hugging Face login

Qwen2.5-Coder-7B-Instruct is gated.

```bash
huggingface-cli login
# paste a token with "Read access to gated repos" scope
```

## 5. Start a serving config

In a `tmux` session:

```bash
tmux new -s vllm
cd /workspace/agentic_systems_serving
./serving/start_vllm.sh vllm_lru
# Ctrl-b d to detach
```

First start downloads the model (~15 GB) and takes a few minutes.
Subsequent starts are seconds.

Confirm it's up:

```bash
curl -s http://localhost:8000/v1/models | jq
```

## 6. Pick a design mode and run the sweep

The harness now supports five layered design modes (plus three legacy
modes preserved for compatibility):

| Mode | Layer | What it varies | What it holds fixed |
|---|---|---|---|
| `serving_sweep` | L1 | all 5 serving configs | orchestration = `cache_aware_ordering` |
| `context_mgmt_sweep` | L2a | `full_context`, `sliding_window`, `summarization` | serving = `vllm_lru` |
| `prompt_assembly_sweep` | L2b | `full_context`, `cache_aware_ordering` | serving = `vllm_lru` |
| `tool_output_sweep` | L2c | `full_context`, `tool_output_compression` | serving = `vllm_lru` |
| `interactions` | L3 | only the 4 named H1–H4 hypothesis cells | — |
| `grid` / `ofat` / `ofat+interactions` | legacy | (preserved) | (preserved) |

Run examples:

```bash
cd /workspace/agentic_systems_serving

# Layer 2 sweeps — only need vllm_lru running:
PYTHONPATH=. python experiments/run_experiment.py \
    --backend vllm --design context_mgmt_sweep \
    --task-source swebench --task-limit 3 --gpu-class A100

PYTHONPATH=. python experiments/run_experiment.py \
    --backend vllm --design prompt_assembly_sweep \
    --task-source swebench --task-limit 3 --gpu-class A100

PYTHONPATH=. python experiments/run_experiment.py \
    --backend vllm --design tool_output_sweep \
    --task-source swebench --task-limit 3 --gpu-class A100

# Layer 1 (serving sweep) — requires restarting the server between cells
# with different serving configs (see "Switching serving engines" below).

# Layer 3 interactions — requires multiple serving engines available;
# the runner expects all 5 serving configs to be selectable, but the
# server you have running serves only one at a time. Run interactions in
# four passes, one per H, restarting the server in between.
```

## 7. Switching serving engines mid-sweep

There is no single command that restarts vLLM mid-sweep — the runner
calls one endpoint and trusts that whatever is running is the config
named in the cell. To compare serving configs:

```bash
for cfg in cache_off vllm_lru vllm_lmcache vllm_continuum sglang; do
    pkill -f 'vllm.entrypoints' || true
    pkill -f 'sglang.launch_server' || true
    pkill -f '^vllm serve' || true
    sleep 5
    nohup ./serving/start_vllm.sh "$cfg" > vllm-"$cfg".log 2>&1 &
    sleep 60   # model load
    PYTHONPATH=. python experiments/run_experiment.py \
        --backend vllm --design serving_sweep \
        --task-source swebench --task-limit 3 --gpu-class A100
done
```

Note: `serving_sweep` enumerates all 5 serving configs in `select_cells`,
but only the cell whose config matches the currently-running engine
produces meaningful numbers. The simplest analysis is to run the inner
sweep with `--design ofat` or directly target one cell at a time. (A
future refactor could gate cells by "what's running"; for now keep the
runs explicit so the metrics file is self-describing.)

### Required installs per engine

| Engine | Needed for serving configs | Install |
|---|---|---|
| vanilla vLLM | `cache_off`, `vllm_lru`, `vllm_lmcache` | `pip install 'vllm>=0.6.0'` (done by setup_runpod.sh) |
| vllm-continuum fork | `vllm_continuum` | Clone https://github.com/Hanchenli/vllm-continuum, `pip uninstall vllm && pip install -e <fork>`. **Conflicts with vanilla vLLM** — pick one. |
| SGLang | `sglang` | `pip install 'sglang[all]'` |
| LMCache | `vllm_lmcache` | `pip install lmcache` (done by setup_runpod.sh) |

`start_vllm.sh` checks for the right binary and exits with an install
hint if it's missing.

## 8. Retrieve results

```bash
# From your laptop:
scp -P <PORT> -r root@<POD-IP>:/workspace/agentic_systems_serving/experiments/results .
```

Per cell, two files are produced:
- `sweep<id>__<tag>__<orch>__<serv>.jsonl`           — per-turn metrics
- `sweep<id>__<tag>__<orch>__<serv>__episodes.jsonl` — per-episode rows with `verified` (True/False from swebench harness, or None for mock / failed evaluation)

The per-sweep summary JSON aggregates all cells in one run.

---

## Notes / gotchas

**SWE-bench patch evaluation.** When `--task-source swebench`, the
harness calls `swebench.harness.run_evaluation` after each terminal
submit to score the patch. Per-episode cost is 30–120 s. Set
`--task-limit` accordingly; with 5 tasks per cell and the eval overhead,
a single serving_sweep run is roughly 30 min on A100.

**A100 vs H100.** Both work. The `--gpu-class A100` flag is metadata
used by `gpu_classes` filtering in the YAMLs — currently all five
serving configs declare both, so filtering is a no-op. Tighten the
YAMLs if a particular config requires Hopper-only features.

**Cost discipline.** A pod left running with `--design grid
--task-limit 50` will burn credits fast. Start with `--design
serving_sweep --task-limit 3` for a directional read (~20–30 min on
A100), then expand once the signal is clear.

**Switching back to vanilla vLLM after continuum.** Reinstall:
`pip uninstall -y vllm && pip install 'vllm>=0.6.0'`.
