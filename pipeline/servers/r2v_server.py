#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import imageio
import torch
from diffusers.utils import load_image

ROOT = Path("/home/jupyter/asaf/video-ai-lab")
SKYREELS_ROOT = ROOT / "external/SkyReels-V3"
RUNTIME_ROOT = ROOT / "runtime/r2v"

sys.path.insert(0, str(SKYREELS_ROOT))

from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import ReferenceToVideoPipeline


DEFAULT_MODEL_ID = "Skywork/SkyReels-V3-R2V-14B"


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


class PersistentR2VServer:
    def __init__(
        self,
        model_id: str,
        runtime_dir: Path,
        poll_seconds: float,
    ) -> None:
        self.jobs_dir = runtime_dir / "jobs"
        self.processing_dir = runtime_dir / "processing"
        self.done_dir = runtime_dir / "done"
        self.failed_dir = runtime_dir / "failed"
        self.work_dir = runtime_dir / "work"
        self.poll_seconds = poll_seconds

        for directory in [
            self.jobs_dir,
            self.processing_dir,
            self.done_dir,
            self.failed_dir,
            self.work_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        print("=" * 72, flush=True)
        print("Video AI Lab - Persistent R2V Server", flush=True)
        print("=" * 72, flush=True)
        print("Visible GPU count:", torch.cuda.device_count(), flush=True)

        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "R2V server must see exactly one GPU. "
                "Start it with CUDA_VISIBLE_DEVICES=1."
            )

        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("Resolving model:", model_id, flush=True)

        self.model_path = download_model(model_id)

        print("Model path:", self.model_path, flush=True)
        print("Loading ReferenceToVideoPipeline once...", flush=True)

        self.pipeline = ReferenceToVideoPipeline(
            model_path=self.model_path,
            use_usp=False,
            offload=False,
            low_vram=False,
        )

        print("", flush=True)
        print("R2V SERVER READY", flush=True)
        print("The model is resident on physical GPU 1.", flush=True)
        print(f"Watching: {self.jobs_dir}", flush=True)

    def claim_next_job(self) -> Path | None:
        jobs = sorted(
            self.jobs_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
        )

        for job in jobs:
            claimed = self.processing_dir / job.name

            try:
                job.replace(claimed)
                return claimed
            except FileNotFoundError:
                continue

        return None

    def process_job(self, job_path: Path) -> None:
        started_at = time.time()
        job = read_json(job_path)

        job_id = str(job.get("job_id") or job_path.stem)
        output_path = Path(job["output"]).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        force = bool(job.get("force", False))

        if output_path.is_file() and not force:
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
            print(f"[{job_id}] Existing output skipped.", flush=True)
            return

        ref_values = job.get("ref_images") or [job["image"]]

        if isinstance(ref_values, str):
            ref_values = [ref_values]

        reference_images = []

        for value in ref_values:
            image_path = Path(value).expanduser().resolve()

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Reference image not found: {image_path}"
                )

            reference_images.append(load_image(str(image_path)))

        prompt = str(job["prompt"])
        duration = int(round(float(job.get("duration", 5))))
        seed = int(job.get("seed", 42))
        resolution = str(job.get("resolution", "720P"))

        print("", flush=True)
        print("-" * 72, flush=True)
        print(f"[{job_id}] Generating R2V", flush=True)
        print(f"References: {len(reference_images)}", flush=True)
        print(f"Duration  : {duration}", flush=True)
        print(f"Output    : {output_path}", flush=True)
        print(f"Seed      : {seed}", flush=True)
        print("-" * 72, flush=True)

        with torch.inference_mode():
            frames = self.pipeline.generate_video(
                ref_imgs=reference_images,
                prompt=prompt,
                duration=duration,
                seed=seed,
                resolution=resolution,
            )

        imageio.mimwrite(
            output_path,
            frames,
            fps=24,
            quality=8,
            output_params=["-loglevel", "error"],
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

        print(
            f"[{job_id}] READY in {elapsed}s: {output_path}",
            flush=True,
        )
        print(
            "Model remains loaded. Waiting for next job.",
            flush=True,
        )

    def mark_failed(self, job_path: Path, exc: Exception) -> None:
        try:
            job = read_json(job_path)
        except Exception:
            job = {}

        job_id = str(job.get("job_id") or job_path.stem)

        write_json(
            self.failed_dir / f"{job_id}.json",
            {
                **job,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        job_path.unlink(missing_ok=True)

        print(
            f"[{job_id}] FAILED: {exc}",
            file=sys.stderr,
            flush=True,
        )

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
    parser = argparse.ArgumentParser()
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

    server = PersistentR2VServer(
        model_id=args.model_id,
        runtime_dir=args.runtime_dir.resolve(),
        poll_seconds=args.poll_seconds,
    )

    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
