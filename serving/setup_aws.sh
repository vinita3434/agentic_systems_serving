#!/usr/bin/env bash
# One-shot bootstrap for a SINGLE AWS EC2 GPU instance that runs EVERYTHING:
# vLLM serving (on the GPU) + Docker task sandbox + the harness. No split, no
# network hop — the harness talks to vLLM over localhost. This is the whole
# point of the AWS route vs the old RunPod pod (which couldn't run Docker).
#
# Assumes: AWS Deep Learning AMI (Ubuntu) — ships NVIDIA driver + CUDA + Docker
# + nvidia-container-toolkit preinstalled. If you used a bare Ubuntu AMI, this
# script stops at the first missing piece and tells you what to install.
#
# Recommended instance: g6e.xlarge (L40S 48GB, 4 vCPU). Fits the default model
# (Qwen2.5-Coder-32B-Instruct-AWQ, ~20GB) with room for KV cache.
#
# Usage (after SSH into the box):
#   REPO_URL=https://github.com/vinita3434/agentic_systems_serving.git \
#     bash setup_aws.sh
#
# Everything is parameterized at the top so the model/instance is easy to swap.

set -euo pipefail

# ---- knobs -----------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
QUANTIZATION="${QUANTIZATION:-awq_marlin}"   # set "" for a non-quantized model
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PORT="${PORT:-8000}"
REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-$HOME/agentic_systems_serving}"
VENV="${VENV:-$HOME/.venv-serving}"
# Pin vLLM explicitly so re-runs and new boxes get the SAME engine (metrics
# comparability). Bump deliberately, not by accident. 0.6.6 serves
# Qwen2.5-Coder-32B-AWQ via the awq_marlin kernel on Ada (L40S, sm_89).
VLLM_VERSION="${VLLM_VERSION:-0.6.6}"
# Persistent HF cache so the ~20GB weights download ONCE and survive re-runs
# and AMI snapshots. Keep this on the big gp3 root (or a mounted volume).
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME
# Rust-accelerated HF downloads (needs the hf_transfer pkg, installed below).
export HF_HUB_ENABLE_HF_TRANSFER=1

say() { echo; echo "==[$1]== ${*:2}"; }

# ---- 1. verify the GPU is visible -----------------------------------------
say 1 "GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This is not a GPU/driver AMI."
    echo "Use the AWS Deep Learning AMI (Ubuntu), or install NVIDIA drivers."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---- 2. verify Docker works (native — the thing RunPod couldn't do) --------
say 2 "Docker check"
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found. Install: curl -fsSL https://get.docker.com | sh"
    exit 1
fi
if ! docker run --rm hello-world >/dev/null 2>&1; then
    echo "ERROR: 'docker run' failed. If it's a permissions error:"
    echo "  sudo usermod -aG docker \$USER   # then log out/in"
    exit 1
fi
echo "docker ok: $(docker --version)"

# ---- 3. Python venv + serving/harness deps ---------------------------------
# Dedicated venv — never install into the DLAMI conda base (it drifts and is
# shared). Creating a venv over an existing one is a no-op, so re-runs are safe.
say 3 "Python env @ $VENV"
# Check for pip, not just python: a venv whose ensurepip failed leaves a bin/python
# but no bin/pip, and reusing it breaks the install below. Rebuild if pip is missing.
if [[ ! -x "$VENV/bin/pip" ]]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
else
    echo "reusing existing venv (idempotent)"
fi
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools
# Pinned vLLM (bundles a matching CUDA torch), plus eval + dataset + docker SDK.
# hf_transfer gives the Rust-accelerated downloader used by the env var above.
# transformers <4.48: newer releases drop Qwen2Tokenizer.all_special_tokens_extended
# which pinned vLLM 0.6.6 calls at startup (serve crashes otherwise).
"$VENV/bin/pip" install -q "vllm==${VLLM_VERSION}" 'hf_transfer' 'swebench' \
    'datasets>=2.20' 'docker' 'httpx' 'pyyaml' 'transformers<4.48'

# ---- 4. get the repo -------------------------------------------------------
say 4 "Repo @ $REPO_DIR"
if [[ ! -d "$REPO_DIR" ]]; then
    if [[ -z "$REPO_URL" ]]; then
        echo "ERROR: $REPO_DIR missing and REPO_URL unset."
        echo "Re-run with:  REPO_URL=<git url> bash setup_aws.sh"
        exit 1
    fi
    git clone "$REPO_URL" "$REPO_DIR"
fi
"$VENV/bin/pip" install -q -r "$REPO_DIR/requirements.txt" || true

# ---- 5. HF auth + cache (Qwen weights) -------------------------------------
say 5 "HuggingFace token + cache"
echo "HF cache: $HF_HOME  (weights persist here across re-runs — no re-download)"
echo "hf_transfer: HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "NOTE: export HF_TOKEN=<read token> before starting vLLM if the model is gated."
    echo "      (Base Qwen2.5-Coder is gated; the AWQ mirror usually is not — check.)"
else
    echo "HF_TOKEN is set."
fi

# ---- 6. start vLLM in tmux -------------------------------------------------
say 6 "Start vLLM (tmux session 'vllm')"
QFLAG=""; [[ -n "$QUANTIZATION" ]] && QFLAG="--quantization $QUANTIZATION"
# Export the HF vars INTO the tmux command so a pre-existing tmux server (with
# a stale env) still gets accelerated downloads + the persistent cache.
ENV_PREFIX="HF_HOME=$HF_HOME HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
[[ -n "${HF_TOKEN:-}" ]] && ENV_PREFIX="$ENV_PREFIX HF_TOKEN=$HF_TOKEN"
START_CMD="$ENV_PREFIX $VENV/bin/vllm serve $MODEL \
  --host 0.0.0.0 --port $PORT $QFLAG \
  --dtype float16 --gpu-memory-utilization $GPU_MEM_UTIL \
  --max-model-len $MAX_MODEL_LEN --max-num-seqs 64 \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser hermes"
echo "CMD: $START_CMD"
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "$START_CMD 2>&1 | tee $HOME/vllm.log"

echo
echo "vLLM is booting in tmux 'vllm' (first run downloads ~20GB — a few min)."
echo "Watch it:      tmux attach -t vllm     (detach: Ctrl-b then d)"
echo "Wait for up:   until curl -s localhost:$PORT/v1/models | grep -q id; do sleep 5; done"

# ---- 7. AWQ Marlin kernel sanity check -------------------------------------
# On Ada (L40S, sm_89) AWQ should run through the FAST Marlin kernel. If vLLM
# silently falls back to the slow generic AWQ path, TTFT/throughput tank — and
# that would quietly poison every serving-layer metric. Best-effort + non-fatal:
# we tail the boot log for the kernel vLLM chose. Never blocks the run.
if [[ "$QUANTIZATION" == awq* ]]; then
    say 7 "AWQ Marlin kernel check (waiting for vLLM to log its kernel)"
    kernel="unknown"
    for _ in $(seq 1 180); do   # up to ~15 min to cover a cold 20GB download
        if grep -qiE "marlin" "$HOME/vllm.log" 2>/dev/null; then
            kernel="marlin"; break
        elif grep -qiE "error|traceback|out of memory" "$HOME/vllm.log" 2>/dev/null; then
            kernel="error"; break
        elif curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -q id; then
            kernel="up_no_marlin"; break   # server ready but no marlin line seen
        fi
        sleep 5
    done
    case "$kernel" in
        marlin)
            echo "OK: AWQ Marlin kernel active (fast path).";;
        up_no_marlin)
            echo "WARNING: vLLM is up but no 'marlin' line found in the log —";
            echo "  AWQ may be on the SLOW generic kernel. Confirm with:";
            echo "    grep -iE 'awq|marlin|quantization' $HOME/vllm.log";;
        error)
            echo "WARNING: vLLM logged an error during boot — check:";
            echo "    tmux attach -t vllm   /   tail -n 100 $HOME/vllm.log";;
        *)
            echo "NOTE: could not confirm the kernel yet (still booting). Re-check:";
            echo "    grep -iE 'awq|marlin|quantization' $HOME/vllm.log";;
    esac
fi

echo
echo "Then run ONE smoke task (single episode — never two at once):"
echo "  cd $REPO_DIR"
echo "  PYTHONPATH=. $VENV/bin/python experiments/run_experiment.py \\"
echo "    --layer custom --orchestration cache_aware_ordering --serving vllm_lru_32b \\"
echo "    --backend vllm --task-source swebench --swebench-split verified \\"
echo "    --task-id astropy__astropy-12907 --gpu-class A100 --max-turns 40"
echo
echo "(--serving vllm_lru_32b already targets localhost:$PORT and the 32B model,"
echo " so no --vllm-base-url is needed. Re-running this script is safe: the venv"
echo " is reused and weights are cached in $HF_HOME — no 20GB re-download.)"
