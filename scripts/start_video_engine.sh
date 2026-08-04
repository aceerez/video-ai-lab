#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="/home/jupyter/asaf/video-ai-lab"
LOG_DIR="$ROOT/logs"
HF_HOME_DIR="$ROOT/models/huggingface"

SKYREELS_PYTHON="$ROOT/external/SkyReels-V3/.venv/bin/python"
QWEN_PYTHON="$ROOT/external/Qwen-Image-Edit-2511/.venv-qwen-image-edit-2511/bin/python"

A2V_SERVER="$ROOT/pipeline/servers/a2v_server.py"
R2V_SERVER="$ROOT/pipeline/servers/r2v_server.py"
IMAGE_SERVER="$ROOT/pipeline/servers/image_server.py"

mkdir -p \
  "$LOG_DIR" \
  "$HF_HOME_DIR" \
  "$ROOT/runtime/a2v/jobs" \
  "$ROOT/runtime/a2v/processing" \
  "$ROOT/runtime/a2v/done" \
  "$ROOT/runtime/a2v/failed" \
  "$ROOT/runtime/r2v/jobs" \
  "$ROOT/runtime/r2v/processing" \
  "$ROOT/runtime/r2v/done" \
  "$ROOT/runtime/r2v/failed" \
  "$ROOT/runtime/image/jobs" \
  "$ROOT/runtime/image/processing" \
  "$ROOT/runtime/image/done" \
  "$ROOT/runtime/image/failed"

export HF_HOME="$HF_HOME_DIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

is_running() {
  local pid_file="$1"

  [[ -f "$pid_file" ]] || return 1

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_server() {
  local name="$1"
  local gpu="$2"
  local python_bin="$3"
  local server_file="$4"
  local ready_text="$5"

  local pid_file="$LOG_DIR/${name}_server.pid"
  local log_file="$LOG_DIR/${name}_server.log"

  if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: Python executable not found: $python_bin"
    return 1
  fi

  if [[ ! -f "$server_file" ]]; then
    echo "ERROR: Server script not found: $server_file"
    return 1
  fi

  if is_running "$pid_file"; then
    echo "$name server already running. PID: $(cat "$pid_file")"
    return 0
  fi

  rm -f "$pid_file"
  : > "$log_file"

  echo "Starting $name server on physical GPU $gpu..."

  nohup env \
    HF_HOME="$HF_HOME_DIR" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "$python_bin" "$server_file" \
    > "$log_file" \
    2>&1 &

  local pid=$!
  echo "$pid" > "$pid_file"

  sleep 2

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: $name server exited during startup."
    tail -80 "$log_file" || true
    return 1
  fi

  echo "$name server started. PID: $pid"
}

wait_for_ready() {
  local name="$1"
  local ready_text="$2"
  local timeout_seconds="$3"

  local pid_file="$LOG_DIR/${name}_server.pid"
  local log_file="$LOG_DIR/${name}_server.log"

  local waited=0

  echo "Waiting for $name: $ready_text"

  while (( waited < timeout_seconds )); do
    if grep -Fq "$ready_text" "$log_file" 2>/dev/null; then
      echo "$name is READY."
      return 0
    fi

    if ! is_running "$pid_file"; then
      echo "ERROR: $name server stopped before becoming ready."
      tail -100 "$log_file" || true
      return 1
    fi

    sleep 5
    waited=$((waited + 5))

    if (( waited % 30 == 0 )); then
      echo "$name still loading... ${waited}s"
    fi
  done

  echo "ERROR: Timed out waiting for $name after ${timeout_seconds}s."
  tail -100 "$log_file" || true
  return 1
}

echo
echo "========================================================================"
echo " Video AI Engine"
echo "========================================================================"
echo "Root    : $ROOT"
echo "HF cache: $HF_HOME_DIR"
echo

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg..."

  export DEBIAN_FRONTEND=noninteractive

  apt-get update
  apt-get install -y ffmpeg

  hash -r
fi

echo
echo "Checking FlashAttention..."

if ! "$SKYREELS_PYTHON" - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func

q=torch.randn(1,32,16,128,device="cuda",dtype=torch.bfloat16)
k=torch.randn_like(q)
v=torch.randn_like(q)

flash_attn_func(q,k,v)

print("FlashAttention OK")
PY
then

    echo
    echo "FlashAttention missing or broken."
    echo "Installing local wheel..."

    WHEEL="$(ls -1 "$ROOT"/external/flash-attention-wheels/flash_attn-*.whl | head -1)"

    if [[ ! -f "$WHEEL" ]]; then
        echo "ERROR: FlashAttention wheel not found."
        exit 1
    fi

    "$SKYREELS_PYTHON" -m pip install \
        --force-reinstall \
        --no-deps \
        "$WHEEL"

    echo
    echo "Rechecking FlashAttention..."

    "$SKYREELS_PYTHON" - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func

q=torch.randn(1,32,16,128,device="cuda",dtype=torch.bfloat16)
k=torch.randn_like(q)
v=torch.randn_like(q)

flash_attn_func(q,k,v)

print("FlashAttention OK")
PY
fi

gpu_count="$("$SKYREELS_PYTHON" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [[ "$gpu_count" -lt 3 ]]; then
  echo "ERROR: This engine requires 3 visible GPUs, but found $gpu_count."
  exit 1
fi

start_server \
  "a2v" \
  "0" \
  "$SKYREELS_PYTHON" \
  "$A2V_SERVER" \
  "A2V SERVER READY"

start_server \
  "r2v" \
  "1" \
  "$SKYREELS_PYTHON" \
  "$R2V_SERVER" \
  "R2V SERVER READY"

start_server \
  "image" \
  "2" \
  "$QWEN_PYTHON" \
  "$IMAGE_SERVER" \
  "QWEN IMAGE SERVER READY"

echo
echo "All processes started. Waiting for models to become ready..."
echo

# Model initialization can take several minutes after a fresh pod start.
wait_for_ready "a2v" "A2V SERVER READY" 1200 &
wait_a2v=$!

wait_for_ready "r2v" "R2V SERVER READY" 1200 &
wait_r2v=$!

wait_for_ready "image" "QWEN IMAGE SERVER READY" 1200 &
wait_image=$!

failed=0

wait "$wait_a2v" || failed=1
wait "$wait_r2v" || failed=1
wait "$wait_image" || failed=1

if [[ "$failed" -ne 0 ]]; then
  echo
  echo "One or more servers failed to become ready."
  exit 1
fi

echo
echo "========================================================================"
echo " Video AI Engine READY"
echo "========================================================================"
echo "GPU 0 -> Persistent A2V"
echo "GPU 1 -> Persistent R2V"
echo "GPU 2 -> Persistent Qwen Image Edit"
echo
echo "PIDs:"
echo "A2V   : $(cat "$LOG_DIR/a2v_server.pid")"
echo "R2V   : $(cat "$LOG_DIR/r2v_server.pid")"
echo "Image : $(cat "$LOG_DIR/image_server.pid")"
echo

nvidia-smi \
  --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader
