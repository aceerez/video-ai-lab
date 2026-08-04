#!/usr/bin/env bash

# -------------------------------------------------------
# Video AI Lab Environment
# -------------------------------------------------------

export VIDEO_AI_ROOT="/home/jupyter/asaf/video-ai-lab"
export SKYREELS_ROOT="$VIDEO_AI_ROOT/external/SkyReels-V3"
export PATH="$VIDEO_AI_ROOT/.tools/bin:$PATH"

# Hugging Face cache
export HF_HOME="$VIDEO_AI_ROOT/models/huggingface"

# Better CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

########################################
# Install FFmpeg if missing
########################################

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg not found. Installing..."

    if [[ "$(id -u)" -eq 0 ]]; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
    else
        echo "ERROR: Root access is required to install FFmpeg."
        return 1 2>/dev/null || exit 1
    fi
fi

########################################
# Disable incompatible system FlashAttention
########################################

FLASH_STATUS=0

python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("flash_attn")

if spec is None:
    print("FlashAttention: not installed")
    raise SystemExit(0)

try:
    import flash_attn
    print(f"FlashAttention: OK ({flash_attn.__file__})")
    raise SystemExit(0)
except Exception as exc:
    print(f"FlashAttention: broken ({exc})")
    raise SystemExit(1)
PY

FLASH_STATUS=$?

if [[ "$FLASH_STATUS" -ne 0 ]]; then
    echo "Disabling incompatible system FlashAttention..."

    FLASH_SITE="/usr/local/lib/python3.12/dist-packages"

    if [[ -d "$FLASH_SITE/flash_attn" ]]; then
        mv \
          "$FLASH_SITE/flash_attn" \
          "$FLASH_SITE/flash_attn.disabled" \
          2>/dev/null || true
    fi

    for file in "$FLASH_SITE"/flash_attn_2_cuda*.so; do
        [[ -e "$file" ]] || continue
        mv "$file" "${file}.disabled" 2>/dev/null || true
    done

    for directory in "$FLASH_SITE"/flash_attn-*.dist-info; do
        [[ -e "$directory" ]] || continue
        mv "$directory" "${directory}.disabled" 2>/dev/null || true
    done

    python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("flash_attn")
print("FlashAttention after repair:", spec)

if spec is not None:
    raise SystemExit("ERROR: flash_attn is still visible to Python")
PY
fi

########################################
# Activate SkyReels environment
########################################

cd "$SKYREELS_ROOT" || {
    echo "ERROR: SkyReels directory not found: $SKYREELS_ROOT"
    return 1 2>/dev/null || exit 1
}

if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
else
    echo "ERROR: SkyReels venv not found"
    return 1 2>/dev/null || exit 1
fi

########################################
# Runtime verification
########################################

echo
echo "========================================"
echo " Video AI Lab"
echo "========================================"
echo "Project : $VIDEO_AI_ROOT"
echo "SkyReels: $SKYREELS_ROOT"
echo "Python  : $(which python)"
echo "UV      : $(command -v uv || echo 'not found')"
echo "FFmpeg  : $(command -v ffmpeg || echo 'not found')"
echo

ffmpeg -version | head -1

echo

python - <<'PY'
import importlib.util
import torch

print("Torch :", torch.__version__)
print("Path  :", torch.__file__)
print("CUDA  :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPUs  :", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024 ** 3)
    print(f"GPU {i}: {props.name} ({memory_gb:.1f} GB)")

print("FlashAttention:", importlib.util.find_spec("flash_attn"))
PY

echo
echo "Environment ready."