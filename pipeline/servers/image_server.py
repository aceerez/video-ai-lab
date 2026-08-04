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

# Physical GPU 2 becomes cuda:0 inside this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


ROOT = Path("/home/jupyter/asaf/video-ai-lab")
DEFAULT_RUNTIME = ROOT / "runtime/image"
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


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


class PersistentImageServer:
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
        self.poll_seconds = poll_seconds

        for directory in (
            self.jobs_dir,
            self.processing_dir,
            self.done_dir,
            self.failed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        print("=" * 72, flush=True)
        print("Video AI Lab - Persistent Qwen Image Server", flush=True)
        print("=" * 72, flush=True)
        print("Visible GPU count:", torch.cuda.device_count(), flush=True)

        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Image server must see exactly one GPU. "
                "Start it with CUDA_VISIBLE_DEVICES=2."
            )

        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("Loading model once:", model_id, flush=True)

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

        self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(disable=False)

        print("", flush=True)
        print("QWEN IMAGE SERVER READY", flush=True)
        print("The model is resident on physical GPU 2.", flush=True)
        print("Watching:", self.jobs_dir, flush=True)

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

        image_values = job.get("images")

        if not image_values:
            single_image = job.get("image")
            image_values = [single_image] if single_image else []

        if not image_values:
            raise ValueError(f"{job_id}: no input images were provided")

        images: list[Image.Image] = []

        for value in image_values:
            image_path = Path(value).expanduser().resolve()

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Input image not found: {image_path}"
                )

            images.append(Image.open(image_path).convert("RGB"))

        prompt = str(job["prompt"])
        negative_prompt = str(job.get("negative_prompt", ""))

        seed = int(job.get("seed", 42))
        steps = int(job.get("num_inference_steps", 50))
        true_cfg_scale = float(job.get("true_cfg_scale", 3.0))
        guidance_scale = float(job.get("guidance_scale", 1.0))

        print("", flush=True)
        print("-" * 72, flush=True)
        print(f"[{job_id}] Generating image", flush=True)
        print(f"Inputs : {len(images)}", flush=True)
        print(f"Output : {output_path}", flush=True)
        print(f"Seed   : {seed}", flush=True)
        print(f"Steps  : {steps}", flush=True)
        print("-" * 72, flush=True)

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)

        with torch.inference_mode():
            result = self.pipeline(
                image=images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                num_images_per_prompt=1,
            )

        result.images[0].save(output_path)

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

        for image in images:
            image.close()

        del result
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
    parser = argparse.ArgumentParser(
        description="Persistent Qwen image-edit worker."
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )

    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME,
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    server = PersistentImageServer(
        model_id=args.model_id,
        runtime_dir=args.runtime_dir.resolve(),
        poll_seconds=args.poll_seconds,
    )

    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
