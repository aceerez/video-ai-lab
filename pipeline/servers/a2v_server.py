#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Must be set before importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# The base container contains an incompatible system flash_attn build.
# Hide system dist-packages so SkyReels falls back to PyTorch attention.
import sys
sys.path = [
    entry
    for entry in sys.path
    if entry != "/usr/local/lib/python3.12/dist-packages"
]

import imageio
import torch

ROOT = Path("/home/jupyter/asaf/video-ai-lab")
SKYREELS_ROOT = ROOT / "external/SkyReels-V3"
RUNTIME_ROOT = ROOT / "runtime/a2v"

sys.path.insert(0, str(SKYREELS_ROOT))

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import TalkingAvatarPipeline
from skyreels_v3.utils.avatar_preprocess import preprocess_audio


DEFAULT_MODEL_ID = "Skywork/SkyReels-V3-A2V-19B"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_input_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    return path


class PersistentA2VServer:
    def __init__(
        self,
        model_id: str,
        jobs_dir: Path,
        processing_dir: Path,
        done_dir: Path,
        failed_dir: Path,
        work_dir: Path,
        poll_seconds: float,
    ) -> None:
        self.jobs_dir = jobs_dir
        self.processing_dir = processing_dir
        self.done_dir = done_dir
        self.failed_dir = failed_dir
        self.work_dir = work_dir
        self.poll_seconds = poll_seconds

        for directory in [
            jobs_dir,
            processing_dir,
            done_dir,
            failed_dir,
            work_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        print("=" * 72, flush=True)
        print("Video AI Lab - Persistent A2V Server", flush=True)
        print("=" * 72, flush=True)
        print("Visible GPU count:", torch.cuda.device_count(), flush=True)

        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "A2V server must see exactly one GPU. "
                "Start it with CUDA_VISIBLE_DEVICES=0."
            )

        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("Resolving model:", model_id, flush=True)

        self.model_path = download_model(model_id)
        print("Model path:", self.model_path, flush=True)

        config = WAN_CONFIGS["talking-avatar-19B"]

        print("Loading TalkingAvatarPipeline once...", flush=True)

        self.pipeline = TalkingAvatarPipeline(
            config=config,
            model_path=self.model_path,
            device_id=0,
            rank=0,
            use_usp=False,
            offload=False,
            low_vram=False,
        )

        print("", flush=True)
        print("A2V SERVER READY", flush=True)
        print("The model is resident on GPU 0.", flush=True)
        print(f"Watching: {self.jobs_dir}", flush=True)

    def claim_next_job(self) -> Path | None:
        job_files = sorted(
            self.jobs_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
        )

        for job_file in job_files:
            claimed = self.processing_dir / job_file.name

            try:
                job_file.replace(claimed)
                return claimed
            except FileNotFoundError:
                continue

        return None

    def process_job(self, job_path: Path) -> None:
        started_at = time.time()
        job = read_json(job_path)

        job_id = str(job.get("job_id") or job_path.stem)
        image_path = validate_input_file(job["image"], "Input image")
        audio_path = validate_input_file(job["audio"], "Input audio")
        output_path = Path(job["output"]).expanduser().resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = str(job["prompt"])
        seed = int(job.get("seed", 42))
        resolution = str(job.get("resolution", "720P"))
        force = bool(job.get("force", False))

        if output_path.is_file() and not force:
            print(f"[{job_id}] Existing output, skipping: {output_path}", flush=True)

            write_json(
                self.done_dir / f"{job_id}.json",
                {
                    **job,
                    "status": "ready",
                    "skipped": True,
                    "elapsed_seconds": 0,
                },
            )
            job_path.unlink(missing_ok=True)
            return

        job_work_dir = self.work_dir / job_id

        if job_work_dir.exists():
            shutil.rmtree(job_work_dir)

        job_work_dir.mkdir(parents=True, exist_ok=True)

        print("", flush=True)
        print("-" * 72, flush=True)
        print(f"[{job_id}] Generating A2V", flush=True)
        print(f"Image : {image_path}", flush=True)
        print(f"Audio : {audio_path}", flush=True)
        print(f"Output: {output_path}", flush=True)
        print(f"Seed  : {seed}", flush=True)
        print("-" * 72, flush=True)

        input_data = {
            "prompt": prompt,
            "cond_image": str(image_path),
            "cond_audio": {
                "person1": str(audio_path),
            },
        }

        processed_audio_dir = job_work_dir / "processed_audio"

        input_data, _ = preprocess_audio(
            self.model_path,
            input_data,
            str(processed_audio_dir),
        )

        kwargs = {
            "input_data": input_data,
            "size_buckget": resolution,
            "motion_frame": 5,
            "frame_num": 81,
            "drop_frame": 12,
            "shift": 11,
            "text_guide_scale": 1.0,
            "audio_guide_scale": 1.0,
            "seed": seed,
            "sampling_steps": 4,
            "max_frames_num": 5000,
        }

        # The same self.pipeline instance is reused for every job.
        with torch.inference_mode():
            video_frames = self.pipeline.generate(**kwargs)

        silent_video = job_work_dir / "video_without_audio.mp4"

        imageio.mimwrite(
            silent_video,
            video_frames,
            fps=25,
            quality=8,
            output_params=["-loglevel", "error"],
        )

        generated_audio = Path(input_data["video_audio"]).resolve()

        if not generated_audio.is_file():
            raise FileNotFoundError(
                f"Processed audio was not created: {generated_audio}"
            )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(silent_video),
                "-i",
                str(generated_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )

        elapsed = round(time.time() - started_at, 2)

        write_json(
            self.done_dir / f"{job_id}.json",
            {
                **job,
                "status": "ready",
                "elapsed_seconds": elapsed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        job_path.unlink(missing_ok=True)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"[{job_id}] READY in {elapsed}s: {output_path}", flush=True)
        print("Model remains loaded. Waiting for next job.", flush=True)

    def mark_failed(self, job_path: Path, exc: Exception) -> None:
        job_id = job_path.stem

        try:
            job = read_json(job_path)
            job_id = str(job.get("job_id") or job_id)
        except Exception:
            job = {}

        failure = {
            **job,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        write_json(
            self.failed_dir / f"{job_id}.json",
            failure,
        )

        job_path.unlink(missing_ok=True)

        print(f"[{job_id}] FAILED: {exc}", file=sys.stderr, flush=True)

    def serve_forever(self) -> None:
        while True:
            job_path = self.claim_next_job()

            if job_path is None:
                time.sleep(self.poll_seconds)
                continue

            try:
                self.process_job(job_path)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.mark_failed(job_path, exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Persistent SkyReels A2V worker."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=RUNTIME_ROOT,
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    runtime = args.runtime_dir.resolve()

    server = PersistentA2VServer(
        model_id=args.model_id,
        jobs_dir=runtime / "jobs",
        processing_dir=runtime / "processing",
        done_dir=runtime / "done",
        failed_dir=runtime / "failed",
        work_dir=runtime / "work",
        poll_seconds=args.poll_seconds,
    )

    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
