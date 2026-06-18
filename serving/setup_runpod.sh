#!/usr/bin/env bash
# Bootstrap a fresh RunPod A100 (or H100) pod for this harness.
#
# Pod template: any PyTorch / CUDA 12.x base image (RunPod "PyTorch 2.x" works).
# Typical mounted volume: /workspace
#
# Run AFTER SSH'ing into the pod, from anywhere — the script clones the
# harness into /workspace if REPO_URL is set, otherwise expects you to
# have scp'd the repo into ${WORKSPACE} already.
#
# What it does:
#   1. system deps (Docker for SWEEnv, git, build-essential)
#   2. base Python deps (harness requirements + vanilla vLLM + datasets +
#      sweagent + lmcache)
#   3. optional engine paths (vllm-continuum and SGLang) — instructions
#      only; the script does NOT auto-install these because they require
#      either a fork checkout or a conflicting vLLM install.
#   4. HuggingFace login prompt (needed for Qwen2.5-Coder gated repo)
#   5. quick smoke check that vllm imports and CUDA is visible.
#
# Model: Qwen/Qwen2.5-Coder-7B-Instruct. Update if you change the model
# referenced in configs/serving/*.yaml.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/agentic_systems_serving}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==[1/6]== System dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl build-essential ca-certificates jq tmux

# Docker (SWEEnv needs it to run SWE-bench task containers)
if ! command -v docker >/dev/null 2>&1; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker || true
fi
docker --version || true

echo "==[2/6]== Python toolchain"
"${PYTHON_BIN}" -m pip install --upgrade pip wheel setuptools

echo "==[3/6]== Clone / locate harness repo"
mkdir -p "${WORKSPACE}"
if [[ ! -d "${REPO_DIR}" ]]; then
    if [[ -n "${REPO_URL}" ]]; then
        git clone "${REPO_URL}" "${REPO_DIR}"
    else
        echo "  ${REPO_DIR} not found and REPO_URL not set."
        echo "  scp -r your-laptop:agentic_systems_serving ${WORKSPACE}/  before running this, or"
        echo "  REPO_URL=https://github.com/you/agentic_systems_serving.git bash setup_runpod.sh"
        exit 1
    fi
fi
cd "${REPO_DIR}"

echo "==[4/6]== Base Python deps (vanilla vLLM + harness + sweagent + lmcache)"
"${PYTHON_BIN}" -m pip install --upgrade \
    'vllm>=0.6.0' \
    'sweagent' \
    'swebench' \
    'datasets>=2.20' \
    'lmcache' \
    -r requirements.txt

cat <<'EOF'

==[5/6]== Optional engine installs (read before running)

The harness supports three serving engines:

  (a) vanilla vLLM           — installed above.
  (b) vllm-continuum fork    — workflow-aware KV TTL.
                               Install instructions:
                                 git clone https://github.com/Hanchenli/vllm-continuum /opt/vllm-continuum
                                 cd /opt/vllm-continuum
                                 pip uninstall -y vllm        # the fork replaces it
                                 pip install -e .
                                 which vllm                   # must resolve to the fork
                               You CANNOT have both vanilla vLLM and the
                               fork installed simultaneously; the `vllm`
                               CLI will collide. Pick one for the
                               duration of a sweep, then swap.
  (c) SGLang                 — alternative engine with RadixAttention.
                               pip install 'sglang[all]'
                               Default port is 30000 (not 8000) — the
                               sglang.yaml config reflects this.

Install (b) or (c) on demand before running serving sweeps that touch
those engines. start_vllm.sh will refuse with an install hint if the
engine is not present.

EOF

echo "==[6/6]== Smoke checks"
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
try:
    import vllm; print(f"vllm: {vllm.__version__}")
except ImportError:
    print("vllm: NOT INSTALLED (expected if you're swapping to the continuum fork)")
PY

echo
echo "Setup complete."
echo "Next steps:"
echo "  1. huggingface-cli login           # needed for gated Qwen2.5-Coder weights"
echo "  2. tmux new -s vllm                # so vLLM survives ssh disconnects"
echo "  3. cd ${REPO_DIR}"
echo "     ./serving/start_vllm.sh vllm_lru"
echo "  4. in another shell:"
echo "     cd ${REPO_DIR}"
echo "     PYTHONPATH=. python experiments/run_experiment.py \\"
echo "         --backend vllm --design serving_sweep --task-source swebench --task-limit 3 --gpu-class A100"
