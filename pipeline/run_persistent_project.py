#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/jupyter/asaf/video-ai-lab")

RUNTIMES = {
    "image": ROOT / "runtime/image",
    "a2v": ROOT / "runtime/a2v",
    "r2v": ROOT / "runtime/r2v",
}

SERVERS = {
    "image": {
        "pid": ROOT / "logs/image_server.pid",
        "log": ROOT / "logs/image_server.log",
        "ready": "QWEN IMAGE SERVER READY",
    },
    "a2v": {
        "pid": ROOT / "logs/a2v_server.pid",
        "log": ROOT / "logs/a2v_server.log",
        "ready": "A2V SERVER READY",
    },
    "r2v": {
        "pid": ROOT / "logs/r2v_server.pid",
        "log": ROOT / "logs/r2v_server.log",
        "ready": "R2V SERVER READY",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def process_is_running(pid_file: Path) -> bool:
    if not pid_file.is_file():
        return False

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False

    return Path(f"/proc/{pid}").exists()


def verify_servers() -> None:
    failures: list[str] = []

    for name, server in SERVERS.items():
        pid_file = server["pid"]
        log_file = server["log"]
        ready_text = server["ready"]

        if not process_is_running(pid_file):
            failures.append(f"{name}: process is not running")
            continue

        if not log_file.is_file():
            failures.append(f"{name}: log file is missing")
            continue

        if ready_text not in log_file.read_text(
            encoding="utf-8",
            errors="ignore",
        ):
            failures.append(f"{name}: model is not ready")

    if failures:
        details = "\n".join(f"  - {item}" for item in failures)
        raise RuntimeError(
            "Persistent engine is not ready:\n"
            f"{details}\n\n"
            "Run:\n"
            f"  {ROOT}/scripts/start_video_engine.sh"
        )

    print("Persistent engine: READY", flush=True)


def reset_job_state(runtime: Path, job_id: str) -> None:
    for folder in ("jobs", "processing", "done", "failed"):
        path = runtime / folder / f"{job_id}.json"
        path.unlink(missing_ok=True)


def submit_job(
    worker: str,
    job_id: str,
    payload: dict[str, Any],
) -> None:
    runtime = RUNTIMES[worker]

    for folder in ("jobs", "processing", "done", "failed"):
        (runtime / folder).mkdir(parents=True, exist_ok=True)

    reset_job_state(runtime, job_id)

    destination = runtime / "jobs" / f"{job_id}.json"
    atomic_write_json(destination, payload)

    print(f"Submitted {worker}: {job_id}", flush=True)


def wait_for_jobs(
    jobs: list[tuple[str, str]],
    poll_seconds: float = 5.0,
) -> None:
    remaining = set(jobs)

    while remaining:
        status_parts: list[str] = []

        for worker, job_id in list(remaining):
            runtime = RUNTIMES[worker]

            done_file = runtime / "done" / f"{job_id}.json"
            failed_file = runtime / "failed" / f"{job_id}.json"
            processing_file = runtime / "processing" / f"{job_id}.json"

            if done_file.is_file():
                result = read_json(done_file)
                elapsed = result.get("elapsed_seconds", "?")

                print(
                    f"READY {worker}: {job_id} "
                    f"({elapsed}s)",
                    flush=True,
                )

                remaining.remove((worker, job_id))
                continue

            if failed_file.is_file():
                failure = read_json(failed_file)

                raise RuntimeError(
                    f"{worker} job {job_id} failed:\n"
                    f"{failure.get('error')}\n\n"
                    f"{failure.get('traceback', '')}"
                )

            state = "RUNNING" if processing_file.is_file() else "QUEUED"
            status_parts.append(f"{job_id}={state}")

        if remaining:
            print(" | ".join(sorted(status_parts)), flush=True)
            time.sleep(poll_seconds)


def asset_prompt(asset: dict[str, Any]) -> str | None:
    if asset.get("prompt"):
        return str(asset["prompt"])

    operations = asset.get("operations", [])

    prompts = [
        str(operation["prompt"])
        for operation in operations
        if operation.get("prompt")
    ]

    if prompts:
        return "\n\n".join(prompts)

    return None


def generate_images(
    manifest: dict[str, Any],
    force: bool,
) -> None:
    image_jobs: list[tuple[str, str]] = []

    for asset_id, asset in manifest.get("assets", {}).items():
        if asset.get("type") != "image":
            continue

        output = Path(asset["output"])

        if output.is_file() and not force:
            asset["status"] = "ready"
            print(f"Image already ready: {asset_id}", flush=True)
            continue

        prompt = asset_prompt(asset)

        if not prompt:
            raise RuntimeError(
                f"Image asset '{asset_id}' needs generation, "
                "but it has no prompt or operations[].prompt"
            )

        sources = [
            str(Path(source).expanduser().resolve())
            for source in asset.get("sources", [])
        ]

        if not sources:
            raise RuntimeError(
                f"Image asset '{asset_id}' has no sources"
            )

        missing = [
            source
            for source in sources
            if not Path(source).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing sources for {asset_id}: {missing}"
            )

        job = {
            "job_id": asset_id,
            "images": sources,
            "prompt": prompt,
            "negative_prompt": asset.get("negative_prompt", ""),
            "output": str(output),
            "seed": int(asset.get("seed", 42)),
            "num_inference_steps": int(
                asset.get("num_inference_steps", 50)
            ),
            "true_cfg_scale": float(
                asset.get("true_cfg_scale", 3.0)
            ),
            "guidance_scale": float(
                asset.get("guidance_scale", 1.0)
            ),
            "force": force,
        }

        submit_job("image", asset_id, job)
        image_jobs.append(("image", asset_id))
        asset["status"] = "processing"

    if image_jobs:
        print("\nWaiting for image assets...", flush=True)
        wait_for_jobs(image_jobs)

        for _, asset_id in image_jobs:
            manifest["assets"][asset_id]["status"] = "ready"

    print("Image stage complete.", flush=True)


def submit_video_jobs(
    manifest: dict[str, Any],
    force: bool,
) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []

    for shot in sorted(
        manifest["shots"],
        key=lambda item: int(item["index"]),
    ):
        mode = shot["mode"]
        job_id = shot["shot_id"]

        image_file = Path(shot["image_file"])

        if not image_file.is_file():
            raise FileNotFoundError(
                f"Shot image missing for {job_id}: {image_file}"
            )

        common = {
            "job_id": job_id,
            "image": str(image_file),
            "prompt": shot["prompt"],
            "output": shot["video_file"],
            "seed": int(
                shot.get(
                    "seed",
                    42 + int(shot["index"]) - 1,
                )
            ),
            "resolution": shot.get("resolution", "720P"),
            "force": force,
        }

        if mode == "a2v":
            audio_file = Path(shot["audio_file"])

            if not audio_file.is_file():
                raise FileNotFoundError(
                    f"Shot audio missing for {job_id}: {audio_file}"
                )

            job = {
                **common,
                "audio": str(audio_file),
            }

            submit_job("a2v", job_id, job)
            jobs.append(("a2v", job_id))

        elif mode == "r2v":
            job = {
                **common,
                "duration": int(
                    round(float(shot["duration"]))
                ),
            }

            submit_job("r2v", job_id, job)
            jobs.append(("r2v", job_id))

        else:
            raise ValueError(
                f"Unsupported shot mode for {job_id}: {mode}"
            )

        shot["status"] = "processing"

    return jobs


def assemble_final(
    manifest_path: Path,
    force: bool,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "pipeline/assemble_video.py"),
        str(manifest_path),
    ]

    if force:
        command.append("--force")

    print("\nAssembling final video...", flush=True)
    print("$", " ".join(command), flush=True)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a complete project using persistent "
            "Qwen, A2V and R2V servers."
        )
    )

    parser.add_argument("manifest", type=Path)
    parser.add_argument("--force-images", action="store_true")
    parser.add_argument("--force-videos", action="store_true")
    parser.add_argument("--force-final", action="store_true")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Generate image assets and stop for manual approval.",
    )

    args = parser.parse_args()

    manifest_path = args.manifest.resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    print("=" * 72)
    print("Video AI Lab - Persistent Full Pipeline")
    print("=" * 72)
    print("Manifest:", manifest_path)
    print()

    verify_servers()

    manifest = read_json(manifest_path)

    print("\nStage 1: Images")
    generate_images(
        manifest=manifest,
        force=args.force_images,
    )

    atomic_write_json(manifest_path, manifest)

    if args.images_only:
        print()
        print("=" * 72)
        print("IMAGE APPROVAL REQUIRED")
        print("=" * 72)

        for asset_id, asset in manifest.get("assets", {}).items():
            if asset.get("type") == "image":
                print(f"{asset_id}: {asset['output']}")

        print()
        print("Inspect the images, then rerun without --images-only.")
        return 0

    print("\nStage 2: Video generation")
    video_jobs = submit_video_jobs(
        manifest=manifest,
        force=args.force_videos,
    )

    atomic_write_json(manifest_path, manifest)

    wait_for_jobs(video_jobs)

    for shot in manifest["shots"]:
        shot["status"] = "ready"
        shot["generated_file"] = shot["video_file"]

    atomic_write_json(manifest_path, manifest)

    print("\nStage 3: Final assembly")
    assemble_final(
        manifest_path=manifest_path,
        force=args.force_final,
    )

    manifest = read_json(manifest_path)
    manifest["final_status"] = "ready"
    atomic_write_json(manifest_path, manifest)

    print()
    print("=" * 72)
    print("PROJECT COMPLETE")
    print("=" * 72)
    print("Final video:", manifest["final_output"])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        raise
