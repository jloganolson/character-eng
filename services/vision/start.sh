#!/usr/bin/env bash
# start.sh — Launch vLLM server + vlm-qa app
#
# Usage:
#   ./start.sh                          # defaults
#   ./start.sh --no-trackers            # skip auto-starting trackers
#   ./start.sh --no-camera              # disable local camera, use /inject_frame
#   VLLM_PORT=8001 ./start.sh           # custom vLLM port
#   VLLM_ARGS="--max-model-len 2048" ./start.sh  # extra vLLM args

set -euo pipefail

# --- Config (override via env vars) ---
VLLM_MODEL="${VLLM_MODEL:-LiquidAI/LFM2-VL-3B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "$PROJECT_ROOT" = "/app" ]; then
    VLLM_TIMEOUT="${VLLM_TIMEOUT:-240}"
else
    VLLM_TIMEOUT="${VLLM_TIMEOUT:-120}"
fi
VLLM_ARGS="${VLLM_ARGS:-}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-0}"
APP_PORT="${APP_PORT:-7860}"
APP_TIMEOUT="${APP_TIMEOUT:-30}"
VLLM_BIN="${VLLM_BIN:-}"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_ROOT/.env"
    set +a
fi

if [ -z "$VLLM_GENERATION_CONFIG" ] && [ "$PROJECT_ROOT" = "/app" ]; then
    # Remote/container boots should not depend on a late HF fetch for
    # generation_config.json after the model is already loaded and warmed.
    VLLM_GENERATION_CONFIG="vllm"
fi

BUNDLED_VISION_ENV="${CHARACTER_ENG_BUNDLED_VISION_ENV:-/opt/character-eng/vision-env}"
DEFAULT_RUNTIME_ROOT=""
if [ -n "${CHARACTER_ENG_RUNTIME_ROOT:-}" ]; then
    DEFAULT_RUNTIME_ROOT="$CHARACTER_ENG_RUNTIME_ROOT"
elif [ "$PROJECT_ROOT" = "/app" ] && [ -d "/workspace" ]; then
    DEFAULT_RUNTIME_ROOT="/workspace/character-eng-runtime"
fi
VISION_ENV_SOURCE="project"
if [ -n "${CHARACTER_ENG_VISION_ENV:-}" ]; then
    VISION_ENV="$CHARACTER_ENG_VISION_ENV"
    VISION_ENV_SOURCE="env"
elif [ -x "$BUNDLED_VISION_ENV/bin/python" ]; then
    VISION_ENV="$BUNDLED_VISION_ENV"
    VISION_ENV_SOURCE="bundled"
elif [ -n "$DEFAULT_RUNTIME_ROOT" ] && [ -x "$DEFAULT_RUNTIME_ROOT/vision/.venv/bin/python" ]; then
    VISION_ENV="$DEFAULT_RUNTIME_ROOT/vision/.venv"
    VISION_ENV_SOURCE="runtime"
elif [ -n "$DEFAULT_RUNTIME_ROOT" ]; then
    VISION_ENV="$DEFAULT_RUNTIME_ROOT/vision/.venv"
    VISION_ENV_SOURCE="runtime-sync"
else
    VISION_ENV="$SCRIPT_DIR/.venv"
fi
if [ -n "${CHARACTER_ENG_CACHE_ROOT:-}" ]; then
    CACHE_ROOT="$CHARACTER_ENG_CACHE_ROOT"
elif [ -n "$DEFAULT_RUNTIME_ROOT" ]; then
    CACHE_ROOT="$DEFAULT_RUNTIME_ROOT/.cache"
else
    CACHE_ROOT="$PROJECT_ROOT/.cache"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$VISION_ENV}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
# Transformers v5 deprecates TRANSFORMERS_CACHE in favor of HF_HOME.
unset TRANSFORMERS_CACHE 2>/dev/null || true

mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "=== Syncing vision runtime environment ==="
    uv sync --project "$SCRIPT_DIR" --frozen
    PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
    if [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: vision Python not found at $PYTHON_BIN after sync"
        exit 1
    fi
else
    echo "=== Reusing vision runtime environment (${VISION_ENV_SOURCE}) ==="
fi

if [ -z "$VLLM_BIN" ]; then
    if [ -x "$UV_PROJECT_ENVIRONMENT/bin/vllm" ]; then
        VLLM_BIN="$UV_PROJECT_ENVIRONMENT/bin/vllm"
    elif [ -x "$HOME/.local/share/uv/tools/vllm/bin/vllm" ]; then
        VLLM_BIN="$HOME/.local/share/uv/tools/vllm/bin/vllm"
    elif command -v vllm >/dev/null 2>&1; then
        VLLM_BIN="$(command -v vllm)"
    else
        echo "ERROR: vllm binary not found. Install it with 'uv tool install vllm' or set VLLM_BIN."
        exit 1
    fi
fi

AUTO_TRACKERS="--auto-start-trackers"
NO_CAMERA=""
CLEANED_UP=0
for arg in "$@"; do
    if [ "$arg" = "--no-trackers" ]; then
        AUTO_TRACKERS=""
    elif [ "$arg" = "--no-camera" ]; then
        NO_CAMERA="--no-camera"
    fi
done

cleanup() {
    if [ "$CLEANED_UP" = "1" ]; then
        return 0
    fi
    CLEANED_UP=1
    echo ""
    echo "Shutting down..."
    if [ -n "${APP_PID:-}" ]; then
        kill "$APP_PID" 2>/dev/null && echo "Stopped app (pid $APP_PID)"
        wait "$APP_PID" 2>/dev/null || true
        APP_PID=""
    fi
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null && echo "Stopped vLLM (pid $VLLM_PID)"
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    # Kill any leftover processes on our ports
    lsof -ti:"$VLLM_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:"$APP_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

print_log_tail() {
    local label="$1"
    local logfile="$2"
    if [ -f "$logfile" ]; then
        echo "----- ${label} (last 60 lines) -----"
        tail -n 60 "$logfile" || true
        echo "----- end ${label} -----"
    fi
}

# --- Apply vLLM patch ---
echo "=== Applying vLLM patch (if needed) ==="
"$PYTHON_BIN" "$SCRIPT_DIR/patch_vllm.py"

echo "=== Vision runtime environment ==="
echo "  Source: $VISION_ENV_SOURCE"
echo "  Env:   $UV_PROJECT_ENVIRONMENT"
echo "  Cache: $CACHE_ROOT"

# --- Kill existing processes on target ports ---
echo "=== Checking ports ==="
for port in "$VLLM_PORT" "$APP_PORT"; do
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing existing process(es) on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done

# --- Start vLLM ---
VLLM_ALREADY_RUNNING=false
if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "=== vLLM already running on port $VLLM_PORT ==="
    VLLM_ALREADY_RUNNING=true
    VLLM_PID=""
else
    echo "=== Starting vLLM server ==="
    echo "  Model: $VLLM_MODEL"
    echo "  Port:  $VLLM_PORT"
    echo "  GPU util: $VLLM_GPU_UTIL"
    echo "  Max model len: $VLLM_MAX_MODEL_LEN"

    VLLM_LOG="$SCRIPT_DIR/vllm.log"
    vllm_cmd=(
        "$VLLM_BIN" serve "$VLLM_MODEL"
        --port "$VLLM_PORT"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --gpu-memory-utilization "$VLLM_GPU_UTIL"
    )
    if [ "$VLLM_TRUST_REMOTE_CODE" = "1" ]; then
        vllm_cmd+=(--trust-remote-code)
    fi
    if [ -n "$VLLM_GENERATION_CONFIG" ]; then
        vllm_cmd+=(--generation-config "$VLLM_GENERATION_CONFIG")
    fi
    if [ -n "$VLLM_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_vllm_args=( $VLLM_ARGS )
        vllm_cmd+=("${extra_vllm_args[@]}")
    fi
    "${vllm_cmd[@]}" >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM pid: $VLLM_PID (logs: $VLLM_LOG)"

    # Wait for vLLM to be ready
    echo -n "  Waiting for vLLM"
    elapsed=0
    while ! curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
        # Check if vLLM process died
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: vLLM process died. Check $VLLM_LOG"
            print_log_tail "vLLM log" "$VLLM_LOG"
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
        if [ "$elapsed" -ge "$VLLM_TIMEOUT" ]; then
            echo ""
            echo "ERROR: vLLM did not start within ${VLLM_TIMEOUT}s"
            print_log_tail "vLLM log" "$VLLM_LOG"
            exit 1
        fi
    done
    echo " ready! (${elapsed}s)"
fi

# --- Start app ---
APP_LOG="$SCRIPT_DIR/app.log"
"$PYTHON_BIN" "$SCRIPT_DIR/app.py" \
    --port "$APP_PORT" \
    --vllm-url "http://localhost:$VLLM_PORT/v1" \
    $AUTO_TRACKERS \
    $NO_CAMERA \
    >"$APP_LOG" 2>&1 &
APP_PID=$!

# Wait for app to be ready
echo -n "  Starting app"
elapsed=0
while ! curl -sf "http://localhost:$APP_PORT/" >/dev/null 2>&1; do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: App process died. Check $APP_LOG"
        print_log_tail "vision app log" "$APP_LOG"
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    echo -n "."
    if [ "$elapsed" -ge "$APP_TIMEOUT" ]; then
        echo ""
        echo "ERROR: App did not start within ${APP_TIMEOUT}s. Check $APP_LOG"
        print_log_tail "vision app log" "$APP_LOG"
        exit 1
    fi
done
echo " ready! (${elapsed}s)"

# --- Status report ---
# Check vLLM model name
VLLM_MODEL_NAME=$(curl -sf "http://localhost:$VLLM_PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

# Check tracker status from app
STATUS=$(curl -sf "http://localhost:$APP_PORT/model_status" 2>/dev/null || echo "{}")
vllm_st=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('vllm','?'))" 2>/dev/null || echo "?")
sam3_st=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('sam3',{}).get('status','?'))" 2>/dev/null || echo "?")
face_st=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('face',{}).get('status','?'))" 2>/dev/null || echo "?")
person_st=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('person',{}).get('status','?'))" 2>/dev/null || echo "?")

echo ""
echo "========================================"
echo "  VLM-QA ready!"
echo "  http://localhost:$APP_PORT"
echo "========================================"
echo "  vLLM ($VLLM_MODEL_NAME)  : $vllm_st"
echo "  SAM3                     : $sam3_st"
echo "  InsightFace              : $face_st"
echo "  Person (ReID)            : $person_st"
echo "========================================"
echo "  Logs: $VLLM_LOG | $APP_LOG"
echo "  Ctrl+C to stop"
echo "========================================"
echo ""

# Keep running — wait for app process
wait $APP_PID
