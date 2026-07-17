# RUNBOOK — AWS single-box L1 sweep

One EC2 GPU box runs **everything**: vLLM serving (GPU) + Docker task sandbox
+ the harness. The harness talks to vLLM over **localhost** — no proxy, no
network hop. This is the whole point of the AWS route vs the old RunPod split
(RunPod pods can't run Docker).

**Model:** `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` (AWQ 4-bit, ~20 GB). 32B only —
the 7B never submitted a patch. **Serving config:** `vllm_lru_32b` (LRU
prefix-caching baseline). **Goal today:** a complete L1 sweep of the **hard**
task set on baseline vLLM, orchestration held at `cache_aware_ordering`.

### Hard rules (do not violate)
- **Strictly sequential** — one episode at a time, one harness instance per
  server. The vLLM `/metrics` cache counter is server-global; concurrency
  corrupts it. Never start a second run against the same server.
- **Baseline vLLM only today.** No SGLang / Continuum / InferCept.
- **Budget:** ~$100 total (~50 GPU-hours for the whole project). Stop the box
  when idle; use `--calibrate` before committing to the full sweep.

---

## 1. Launch the EC2 instance

In the EC2 console (region **us-east-1**):

| Setting | Value |
|---|---|
| **AMI** | *AWS Deep Learning AMI (Ubuntu)* — e.g. "Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)". Ships NVIDIA driver + CUDA + Docker + nvidia-container-toolkit. |
| **Instance type** | `g6e.xlarge` (L40S 48 GB, 4 vCPU) |
| **Storage** | **200 GB gp3** root (SWE-bench Docker images + HF weights) |
| **Key pair** | your existing key (e.g. `serving-key.pem`) |
| **Security group** | **SSH only (port 22) from *my IP*** — inbound `tcp/22` source *My IP*. No other inbound; vLLM stays on localhost, never exposed. |

> **Why SSH-only:** vLLM binds `0.0.0.0:8000` inside the box, but nothing needs
> to reach it from outside — the harness is on the same box. Do **not** open 8000.

## 2. SSH in

```bash
ssh -i ~/.ssh/serving-key.pem ubuntu@<EC2_PUBLIC_IP>
```

(DLAMI Ubuntu user is `ubuntu`. If you used a bare Ubuntu AMI instead, the
setup script will stop at the first missing piece and tell you what to install.)

## 3. Clone + bootstrap

```bash
export HF_TOKEN=<hf read token>       # only needed if the AWQ repo is gated
REPO_URL=https://github.com/vinita3434/agentic_systems_serving.git \
  bash <(curl -fsSL https://raw.githubusercontent.com/vinita3434/agentic_systems_serving/main/serving/setup_aws.sh) \
  || true
# --- or, if you prefer to clone first (equivalent): ---
git clone https://github.com/vinita3434/agentic_systems_serving.git
cd agentic_systems_serving
REPO_URL=https://github.com/vinita3434/agentic_systems_serving.git bash serving/setup_aws.sh
```

`setup_aws.sh` (idempotent — safe to re-run; reuses the venv, weights cached in
`~/.cache/huggingface`, no 20 GB re-download):
- verifies GPU + Docker,
- creates a **dedicated venv** at `~/.venv-serving` (never the conda base),
- installs **pinned** `vllm==0.6.6` + `hf_transfer` + swebench/datasets/docker,
- exports `HF_HUB_ENABLE_HF_TRANSFER=1` for fast weight downloads,
- starts vLLM in a tmux session `vllm` with the AWQ flags below,
- runs an **AWQ Marlin kernel check** (warns if the slow AWQ path is active).

Activate the venv for subsequent commands:

```bash
cd ~/agentic_systems_serving
source ~/.venv-serving/bin/activate
export PYTHONPATH=.
```

## 4. Start the vLLM server (what setup_aws.sh runs)

The script starts this for you; here it is explicitly in case you restart it.
L40S 48 GB: ~20 GB weights, `gpu-memory-utilization 0.92` leaves ~24 GB for KV.

```bash
tmux new -s vllm
HF_HUB_ENABLE_HF_TRANSFER=1 vllm serve Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
  --host 0.0.0.0 --port 8000 \
  --quantization awq_marlin --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 --max-num-seqs 32 \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser hermes \
  2>&1 | tee ~/vllm.log
# detach: Ctrl-b then d
```

Wait until it's serving, and confirm the **fast** kernel:

```bash
until curl -s localhost:8000/v1/models | grep -q id; do sleep 5; done
grep -iE "marlin|awq|quantization" ~/vllm.log     # want a "Marlin" line
```

If you see AWQ active but **no** Marlin line, you're on the slow path — stop and
investigate (wrong vLLM version / GPU), because it poisons every serving metric.

## 5. Smoke test — one task end to end

```bash
PYTHONPATH=. python experiments/run_experiment.py \
  --layer custom --orchestration cache_aware_ordering --serving vllm_lru_32b \
  --backend vllm --task-source swebench --swebench-split verified \
  --task-id astropy__astropy-12907 --gpu-class A100 --max-turns 40 \
  --gpu-hourly-usd 1.86
```

Check the episode row in `experiments/results/*__episodes.jsonl` (and the turn
log `*.jsonl`). **It must show:**
- `ttft_ms` populated (non-null, a real number) — streaming TTFT is being measured,
- `cache_hit_rate` **nonzero** on turns after the first — `/metrics` scrape works
  and prefix caching is live,
- `gpu_cost_usd` / `total_cost_usd` computed (non-null) — cost accounting works.

`verified` may be `false` — that's a model-capability outcome, not a plumbing
failure. What matters here is that the metrics are populated.

> Docker note: the harness pulls a per-task SWE-bench image on first use. The
> first smoke task will spend a few minutes pulling; later tasks reuse the cache.

## 6. Create an AMI snapshot — MANDATORY before the sweep

Do this **now**, after the smoke test passes, so you never re-pay the driver /
venv / weights / Docker-image setup. Next session you relaunch from this AMI and
skip straight to the sweep.

```bash
# from your laptop (aws cli), or the EC2 console → Actions → Image → Create image
aws ec2 create-image \
  --instance-id <INSTANCE_ID> \
  --name "agentic-serving-32b-awq-$(date +%Y%m%d)" \
  --description "DLAMI + venv + vllm 0.6.6 + 32B-AWQ weights + swebench images" \
  --no-reboot
```

Wait for the AMI state to become `available` before relying on it.

## 7. Calibrate before the full sweep

Run the first 3 hard tasks and read the projection (time + cost + finish) for
the whole sweep. `--calibrate` implies `--no-resume`, so it times fresh
episodes. **Its 3 episodes are logged and will be skipped by the real sweep
below (resume) — not wasted.**

```bash
PYTHONPATH=. python experiments/run_experiment.py \
  --layer serving --serving vllm_lru_32b \
  --backend vllm --gpu-class A100 \
  --task-source swebench --swebench-split verified --hard-only \
  --gpu-hourly-usd 1.86 \
  --calibrate 3
```

Read the projected total cost against your remaining budget before continuing.
If the projection blows the budget, reduce scope (e.g. `--task-limit`) rather
than running blind.

## 8. Full L1 sweep (in tmux, with auto-stop)

```bash
tmux new -s sweep
cd ~/agentic_systems_serving && source ~/.venv-serving/bin/activate

PYTHONPATH=. python experiments/run_experiment.py \
  --layer serving --serving vllm_lru_32b \
  --backend vllm --gpu-class A100 \
  --task-source swebench --swebench-split verified --hard-only \
  --gpu-hourly-usd 1.86 \
  --shutdown-on-complete
# detach: Ctrl-b then d
```

- `--layer serving --serving vllm_lru_32b` → the single L1 cell
  (orchestration held at `cache_aware_ordering`).
- `--hard-only` auto-raises `--max-turns` to 60 so long trajectories aren't
  truncated.
- **Resume is on by default:** if the box or run dies at task 15/20, just
  re-run the exact same command — completed episodes are skipped, so you pay
  zero repeated GPU work. (Add `--no-resume` only to force a clean re-run.)
- `--shutdown-on-complete` runs `sudo shutdown -h +5` when the sweep finishes,
  so an overnight run doesn't idle-bill. Cancel within 5 min with
  `sudo shutdown -c`.

Never launch a second `run_experiment.py` against this server while one is
running — the cache counter is server-global.

## 9. Stop the box / relaunch next session

- **If `--shutdown-on-complete` fired:** the OS halted, but the instance may
  still bill until you **Stop** it. In the console: *Instances → Stop instance*
  (or `aws ec2 stop-instances --instance-ids <INSTANCE_ID>`). A **stopped**
  instance bills only for its EBS volume, not the GPU.
- **Relaunch next session from the AMI (§6):** launch a new instance from
  `agentic-serving-32b-awq-<date>` (same g6e.xlarge, SSH-only SG), SSH in,
  `source ~/.venv-serving/bin/activate`, restart vLLM (§4, or re-run
  `bash serving/setup_aws.sh`), then go straight to §7/§8. Resume skips whatever
  already finished, so you continue where you left off.

---

### Quick reference — results
- Turn metrics:    `experiments/results/<experiment_id>.jsonl`
- Episode rows:    `experiments/results/<experiment_id>__episodes.jsonl`
- Sweep summary:   `experiments/results/sweep<id>_summary.json`
- Consolidate:     `python experiments/consolidate_metrics.py`
