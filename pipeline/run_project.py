#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/home/jupyter/asaf/video-ai-lab")

QWEN_PYTHON = (
    ROOT
    / "external/Qwen-Image-Edit-2511"
    / ".venv-qwen-image-edit-2511/bin/python"
)

SKYREELS_PYTHON = (
    ROOT
    / "external/SkyReels-V3"
    / ".venv/bin/python"
)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def run_blocking(
    command: list[str],
    env: dict[str, str] | None = None,
) -> None:
    print()
    print("=" * 80, flush=True)
    print("$", " ".join(command), flush=True)
    print("=" * 80, flush=True)

    subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
    )


def generate_pending_assets(
    manifest_path: Path,
    force: bool,
) -> None:
    manifest = load_manifest(manifest_path)

    pending_assets = [
        asset_id
        for asset_id, asset in manifest.get("assets", {}).items()
        if force
        or asset.get("status") != "ready"
        or not Path(asset["output"]).is_file()
    ]

    if not pending_assets:
        print("All image assets are ready.", flush=True)
        return

    for asset_id in pending_assets:
        command = [
            str(QWEN_PYTHON),
            "pipeline/workers/image_worker.py",
            str(manifest_path),
            "--asset",
            asset_id,
        ]

        if force:
            command.append("--force")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        run_blocking(command, env=env)


def validate_project(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)

    required = [
        Path(manifest["song_file"]),
        Path(manifest["character_image"]),
    ]

    for shot in manifest["shots"]:
        required.append(Path(shot["image_file"]))

        if shot["mode"] == "a2v":
            if not shot.get("audio_file"):
                raise RuntimeError(
                    f"A2V shot has no audio_file: {shot['shot_id']}"
                )

            required.append(Path(shot["audio_file"]))

    missing = sorted({
        str(path)
        for path in required
        if not path.is_file()
    })

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Required project files are missing:\n{formatted}"
        )


def create_worker_manifests(
    manifest_path: Path,
) -> tuple[Path, Path]:
    work_dir = manifest_path.parent / "work/manifests"
    work_dir.mkdir(parents=True, exist_ok=True)

    r2v_manifest = work_dir / "r2v_manifest.json"
    a2v_manifest = work_dir / "a2v_manifest.json"

    shutil.copy2(manifest_path, r2v_manifest)
    shutil.copy2(manifest_path, a2v_manifest)

    return r2v_manifest, a2v_manifest


def start_video_workers(
    r2v_manifest: Path,
    a2v_manifest: Path,
    force_videos: bool,
) -> None:
    r2v_command = [
        str(SKYREELS_PYTHON),
        "pipeline/workers/r2v_worker.py",
        str(r2v_manifest),
    ]

    a2v_command = [
        str(SKYREELS_PYTHON),
        "pipeline/workers/a2v_worker.py",
        str(a2v_manifest),
    ]

    if force_videos:
        r2v_command.append("--force")
        a2v_command.append("--force")

    print()
    print("=" * 80, flush=True)
    print("Starting parallel video generation", flush=True)
    print("=" * 80, flush=True)
    print("GPU 0: SkyReels A2V", flush=True)
    print("GPU 1: SkyReels R2V", flush=True)
    print()

    r2v_process = subprocess.Popen(
        r2v_command,
        cwd=ROOT,
        env=os.environ.copy(),
    )

    a2v_process = subprocess.Popen(
        a2v_command,
        cwd=ROOT,
        env=os.environ.copy(),
    )

    print(f"R2V PID: {r2v_process.pid}", flush=True)
    print(f"A2V PID: {a2v_process.pid}", flush=True)

    processes = {
        "R2V": r2v_process,
        "A2V": a2v_process,
    }

    failed_name: str | None = None
    failed_code: int | None = None

    while processes:
        for name, process in list(processes.items()):
            return_code = process.poll()

            if return_code is None:
                continue

            del processes[name]

            if return_code == 0:
                print(f"{name} worker completed successfully.", flush=True)
            else:
                failed_name = name
                failed_code = return_code
                print(
                    f"{name} worker failed with exit code {return_code}.",
                    file=sys.stderr,
                    flush=True,
                )
                break

        if failed_name is not None:
            for name, process in processes.items():
                if process.poll() is None:
                    print(f"Stopping {name} worker...", flush=True)
                    process.terminate()

            for process in processes.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()

            raise RuntimeError(
                f"{failed_name} generation failed "
                f"with exit code {failed_code}"
            )

        if processes:
            import time
            time.sleep(2)

    print()
    print("A2V and R2V generation completed.", flush=True)


def verify_generated_clips(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)

    missing = [
        Path(shot["video_file"])
        for shot in manifest["shots"]
        if not Path(shot["video_file"]).is_file()
    ]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Video generation finished, but clips are missing:\n"
            f"{formatted}"
        )

    print(f"Verified {len(manifest['shots'])} generated clips.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete Video AI Lab project."
    )

    parser.add_argument("manifest", type=Path)

    parser.add_argument(
        "--force-assets",
        action="store_true",
        help="Regenerate all image assets.",
    )

    parser.add_argument(
        "--force-videos",
        action="store_true",
        help="Regenerate all A2V and R2V clips.",
    )

    parser.add_argument(
        "--force-final",
        action="store_true",
        help="Regenerate normalized clips and final video.",
    )

    args = parser.parse_args()
    manifest_path = args.manifest.resolve()

    print()
    print("========================================")
    print(" Video AI Lab - Full Parallel Pipeline")
    print("========================================")
    print("Started :", datetime.now().isoformat(timespec="seconds"))
    print("Manifest:", manifest_path)
    print()

    generate_pending_assets(
        manifest_path=manifest_path,
        force=args.force_assets,
    )

    validate_project(manifest_path)

    r2v_manifest, a2v_manifest = create_worker_manifests(
        manifest_path
    )

    start_video_workers(
        r2v_manifest=r2v_manifest,
        a2v_manifest=a2v_manifest,
        force_videos=args.force_videos,
    )

    verify_generated_clips(manifest_path)

    assemble_command = [
        sys.executable,
        "pipeline/assemble_video.py",
        str(manifest_path),
    ]

    if args.force_final:
        assemble_command.append("--force")

    run_blocking(assemble_command)

    manifest = load_manifest(manifest_path)

    print()
    print("========================================")
    print(" Complete project finished")
    print("========================================")
    print("Finished:", datetime.now().isoformat(timespec="seconds"))
    print("Output  :", manifest["final_output"])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print()
        print(f"PIPELINE FAILED: {exc}", file=sys.stderr)
        raise
