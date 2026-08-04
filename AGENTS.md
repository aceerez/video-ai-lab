# Video AI Lab – Agent Recovery Guide

## Purpose

This repository is a persistent multi-GPU AI video-generation engine.

Current proven setup:

- GPU 0: SkyReels V3 A2V 19B
- GPU 1: SkyReels V3 R2V 14B
- GPU 2: Qwen Image Edit 2511

Repository root:

`/home/jupyter/asaf/video-ai-lab`

Current working project:

`/home/jupyter/asaf/video-ai-lab/projects/itay`

Current approved final output:

`/home/jupyter/asaf/video-ai-lab/projects/itay/final/itay_final_v2.mp4`

Do not redesign the architecture unless explicitly requested.

## User Working Preferences

- Give complete commands ready to paste.
- Prefer complete file replacements with `cat > file <<'EOF'`.
- Do not ask the user to patch many small sections manually.
- Avoid unnecessary new scripts, schemas, or architecture layers.
- Move quickly and test the existing working pipeline.
- Stop after image generation for manual approval.
- Do not rerender all clips when only one clip is bad.
- Preserve approved images and clips.
- Give one practical next action at a time.

## Environments

SkyReels:

`/home/jupyter/asaf/video-ai-lab/external/SkyReels-V3/.venv/bin/python`

Qwen Image Edit:

`/home/jupyter/asaf/video-ai-lab/external/Qwen-Image-Edit-2511/.venv-qwen-image-edit-2511/bin/python`

## Persistent Servers

Server scripts:

- `pipeline/servers/a2v_server.py`
- `pipeline/servers/r2v_server.py`
- `pipeline/servers/image_server.py`

Runtime queues:

- `runtime/a2v/{jobs,processing,done,failed}`
- `runtime/r2v/{jobs,processing,done,failed}`
- `runtime/image/{jobs,processing,done,failed}`

Logs:

- `logs/a2v_server.log`
- `logs/r2v_server.log`
- `logs/image_server.log`

PID files:

- `logs/a2v_server.pid`
- `logs/r2v_server.pid`
- `logs/image_server.pid`

## Start the Engine

```bash
cd /home/jupyter/asaf/video-ai-lab
./scripts/start_video_engine.sh
```

The startup script should:

1. Create required directories.
2. Install `ffmpeg` automatically if missing.
3. Verify FlashAttention with a real CUDA operation.
4. Install the existing local FlashAttention wheel if the test fails.
5. Start the three persistent servers.
6. Avoid duplicate server processes.
7. Wait until all models report READY.

Expected result:

```text
Video AI Engine READY
GPU 0 -> Persistent A2V
GPU 1 -> Persistent R2V
GPU 2 -> Persistent Qwen Image Edit
```

## FlashAttention

Working setup:

```text
flash-attn: 2.8.3.post1
Torch: 2.13.0+cu130
CUDA runtime: 13.0
nvcc: 13.2
GPU: NVIDIA GB200
Compute capability: 10.0
Architecture: SM100
Python: 3.12
Platform: ARM64
CXX11 ABI: True
```

Local wheel:

`/home/jupyter/asaf/video-ai-lab/external/flash-attention-wheels/flash_attn-2.8.3.post1-cp312-cp312-linux_aarch64.whl`

Do not rebuild FlashAttention during normal startup.

Install existing wheel if required:

```bash
/home/jupyter/asaf/video-ai-lab/external/SkyReels-V3/.venv/bin/python   -m pip install   --force-reinstall   --no-deps   /home/jupyter/asaf/video-ai-lab/external/flash-attention-wheels/flash_attn-*.whl
```

## Pipeline

Compiler:

`pipeline/compile_project.py`

Persistent runner:

`pipeline/run_persistent_project.py`

Final assembly:

`pipeline/assemble_video.py`

Generate images and stop for approval:

```bash
/home/jupyter/asaf/video-ai-lab/external/SkyReels-V3/.venv/bin/python   /home/jupyter/asaf/video-ai-lab/pipeline/run_persistent_project.py   /home/jupyter/asaf/video-ai-lab/projects/<project>/render_manifest.json   --force-images   --images-only
```

After approval:

```bash
/home/jupyter/asaf/video-ai-lab/external/SkyReels-V3/.venv/bin/python   /home/jupyter/asaf/video-ai-lab/pipeline/run_persistent_project.py   /home/jupyter/asaf/video-ai-lab/projects/<project>/render_manifest.json
```

## Itay Project

Song:

`/home/jupyter/asaf/video-ai-lab/assets/itay.wav`

Duration:

`47.80 seconds`

References:

- `/home/jupyter/asaf/video-ai-lab/assets/itay_face.png`
- `/home/jupyter/asaf/video-ai-lab/assets/itay_full.jpeg`
- `/home/jupyter/asaf/video-ai-lab/assets/grandpa.jpg`

Approved Itay stage image:

`/home/jupyter/asaf/video-ai-lab/projects/itay/images/itay_1950s_mobster.png`

Approved Itay and Grandpa image:

`/home/jupyter/asaf/video-ai-lab/projects/itay/images/itay_grandpa_hug_1950s.png`

Heights:

- Itay: 188 cm
- Grandpa: 172 cm

Itay must appear approximately 16 cm taller.

Manifest:

`/home/jupyter/asaf/video-ai-lab/projects/itay/render_manifest.json`

Timeline:

- `shot_001_a2v`
- `shot_002_r2v`
- `shot_003_a2v`
- `shot_004_r2v`
- `shot_005_a2v`
- `shot_006_r2v`
- `shot_007_a2v`

## Identity Preservation

Use two references:

1. Close face image for identity.
2. Full-body image for proportions and posture.

The prompt must explicitly assign roles:

- First image is the highest-priority facial identity reference.
- Second image is the body and height reference.
- Preserve identity above styling and background.

Forbid cartoon, anime, illustration, painting, CGI, 3D render, waxy skin, plastic skin, beautification, and face redesign.

## Grandpa Scene Limitation

Dynamic two-person R2V caused identity drift, an invented woman, illustration style, extra people, and distorted bodies.

Working solution:

- Use the approved still image.
- Exactly two people.
- Keep the existing hug pose.
- Minimal motion only.
- Gentle breathing.
- One subtle blink each.
- Very small camera push-in.
- Closed mouths.
- No singing or speech.
- No identity changes.
- No added people.

Stable clip:

`/home/jupyter/asaf/video-ai-lab/projects/itay/clips/shot_004_r2v.mp4`

Do not replace it with a dynamic hug unless explicitly requested.

## Non-Singing R2V Rule

Include:

```text
The character is not singing.
Keep the mouth closed or naturally relaxed.
No speech.
No talking.
No lip sync.
No jaw movement.
Motion comes from the eyes, head, hands and body.
```

## Assembly Cache Problem

Normalized clips are cached under:

`projects/<project>/work/normalized/`

After replacing a source clip, delete the cache:

```bash
cd /home/jupyter/asaf/video-ai-lab

rm -rf   /home/jupyter/asaf/video-ai-lab/projects/itay/work/normalized

rm -f   /home/jupyter/asaf/video-ai-lab/projects/itay/work/concat.txt   /home/jupyter/asaf/video-ai-lab/projects/itay/work/video_timeline.mp4

mkdir -p   /home/jupyter/asaf/video-ai-lab/projects/itay/work/normalized
```

Use a new output filename to avoid player caching, such as:

`itay_final_v2.mp4`

Then assemble:

```bash
/home/jupyter/asaf/video-ai-lab/external/SkyReels-V3/.venv/bin/python   /home/jupyter/asaf/video-ai-lab/pipeline/assemble_video.py   /home/jupyter/asaf/video-ai-lab/projects/itay/render_manifest.json
```

Approved final file:

`/home/jupyter/asaf/video-ai-lab/projects/itay/final/itay_final_v2.mp4`

## Health Checks

```bash
nvidia-smi
tail -f /home/jupyter/asaf/video-ai-lab/logs/a2v_server.log
tail -f /home/jupyter/asaf/video-ai-lab/logs/r2v_server.log
tail -f /home/jupyter/asaf/video-ai-lab/logs/image_server.log
```

Queue state:

```bash
find /home/jupyter/asaf/video-ai-lab/runtime   -maxdepth 3   -type f   -printf '%P\n'   | sort
```

## Do Not

- Do not rerun the whole Itay video unless explicitly requested.
- Do not use `--force-videos` when only one shot needs replacement.
- Do not regenerate approved images.
- Do not overwrite the approved Grandpa clip.
- Do not trust normalized cache after replacing a source clip.
- Do not add another LLM to the pod.
- Do not create unnecessary architecture layers.
- Do not ask the user to patch many small code sections.
- Give full commands or full replacement files.

## Current Working State

```text
Fresh pod
  -> start_video_engine.sh
  -> ffmpeg auto-install
  -> FlashAttention auto-check/install
  -> three persistent GPU servers
  -> project compiler
  -> image generation
  -> image approval
  -> A2V and R2V generation
  -> manual single-shot rerender
  -> clean final assembly
  -> approved Itay final video
```

Preserve this working state before making further changes.
