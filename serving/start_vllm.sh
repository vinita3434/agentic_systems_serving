#!/usr/bin/env bash
# Start a serving engine using one of the configs in configs/serving/*.yaml.
#
# Dispatches on the `engine:` field in the YAML:
#   vllm            -> python -m vllm.entrypoints.openai.api_server
#   vllm-continuum  -> `vllm serve` from the vllm-continuum fork
#                      (https://github.com/Hanchenli/vllm-continuum)
#   sglang          -> python -m sglang.launch_server
#
# Usage:
#   ./serving/start_vllm.sh <config_name>
#
# Examples:
#   ./serving/start_vllm.sh vllm_lru
#   ./serving/start_vllm.sh cache_off
#   ./serving/start_vllm.sh vllm_lmcache
#   ./serving/start_vllm.sh vllm_continuum
#   ./serving/start_vllm.sh sglang
#
# Prereqs (install on the pod via setup_runpod.sh):
#   - vanilla vLLM    : pip install 'vllm>=0.6.0'
#   - vllm-continuum  : pip install -e <continuum-fork-checkout>
#                       (must replace vanilla vLLM; `vllm` CLI must resolve to fork)
#   - SGLang          : pip install 'sglang[all]'
#   - lmcache         : pip install lmcache  (needed only for vllm_lmcache)
#
# Note: this script does NOT auto-install missing dependencies. If a
# required engine binary is absent, it prints the install hint and exits 2.

set -euo pipefail

CONFIG_NAME="${1:-vllm_lru}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/configs/serving/${CONFIG_NAME}.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config not found: ${CONFIG_FILE}" >&2
    echo "Available:" >&2
    ls "${REPO_ROOT}/configs/serving" >&2
    exit 1
fi

# Render the launch command via Python (handles nested flags + engine dispatch).
LAUNCH_CMD="$("${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
import json, shlex, shutil, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
engine = cfg.get("engine", "vllm")
model = cfg["model"]
host = cfg.get("host", "0.0.0.0")
port = cfg.get("port", 8000)
flags = cfg.get("flags") or {}
extra = list(cfg.get("extra_args") or [])


def add_flag(cmd, key, val):
    flag = "--" + str(key).replace("_", "-")
    if isinstance(val, bool):
        if val:
            cmd.append(flag)
    else:
        cmd += [flag, str(val)]


if engine in ("vllm", "vllm-continuum", "infercept"):
    if engine == "vllm-continuum":
        if not shutil.which("vllm"):
            sys.exit("ERROR: engine=vllm-continuum requires the `vllm` CLI "
                     "from the Hanchenli/vllm-continuum fork to be on PATH. "
                     "Install per the fork's README, then retry.")
        cmd = ["vllm", "serve", model, "--host", host, "--port", str(port)]
    elif engine == "infercept":
        # INFERCEPT is a vLLM fork. ASSUMPTION, verify against the fork
        # README: it exposes the OpenAI-compatible api_server like upstream
        # vLLM. If the fork ships a different entrypoint, edit this branch.
        try:
            import vllm  # noqa: F401  -- the fork installs under the vllm package
        except ImportError:
            sys.exit("ERROR: engine=infercept requires the INFERCEPT vLLM "
                     "fork installed (replaces vanilla vLLM). Install it into "
                     "its own venv per the fork's README, then retry.")
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model, "--host", host, "--port", str(port)]
    else:
        try:
            import vllm  # noqa: F401
        except ImportError:
            sys.exit("ERROR: vllm not installed. pip install 'vllm>=0.6.0'")
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model, "--host", host, "--port", str(port)]
    for k, v in flags.items():
        add_flag(cmd, k, v)
    cmd += extra

elif engine == "sglang":
    try:
        import sglang  # noqa: F401
    except ImportError:
        sys.exit("ERROR: sglang not installed. pip install 'sglang[all]'")
    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", model, "--host", host, "--port", str(port)]
    for k, v in flags.items():
        add_flag(cmd, k, v)
    cmd += extra

else:
    sys.exit(f"ERROR: unknown engine '{engine}' in YAML. "
             f"Expected one of: vllm | vllm-continuum | sglang")

print(shlex.join(cmd))
PY
)"

echo "Starting serving config '${CONFIG_NAME}'"
echo "Command: ${LAUNCH_CMD}"
exec bash -c "${LAUNCH_CMD}"
