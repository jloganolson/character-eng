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
VLLM_ARGS="${VLLM_ARGS:-}"
APP_PORT="${APP_PORT:-7860}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_BIN="${VLLM_BIN:-}"

if [ -z "$VLLM_BIN" ]; then
    if [ -x "$HOME/.local/share/uv/tools/vllm/bin/vllm" ]; then
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

# --- Apply vLLM patch ---
echo "=== Applying vLLM patch (if needed) ==="
uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/patch_vllm.py"

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
    "$VLLM_BIN" serve "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --trust-remote-code \
        $VLLM_ARGS \
        >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM pid: $VLLM_PID (logs: $VLLM_LOG)"

    # Wait for vLLM to be ready
    echo -n "  Waiting for vLLM"
    MAX_WAIT=120
    elapsed=0
    while ! curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
        # Check if vLLM process died
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: vLLM process died. Check $VLLM_LOG"
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
        if [ "$elapsed" -ge "$MAX_WAIT" ]; then
            echo ""
            echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
            exit 1
        fi
    done
    echo " ready! (${elapsed}s)"
fi

# --- Start app ---
APP_LOG="$SCRIPT_DIR/app.log"
uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/app.py" \
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
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    echo -n "."
    if [ "$elapsed" -ge 30 ]; then
        echo ""
        echo "ERROR: App did not start within 30s. Check $APP_LOG"
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
