#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SKYREELS_ROOT = Path(
    "/home/jupyter/asaf/video-ai-lab/external/SkyReels-V3"
)

SKYREELS_PYTHON = SKYREELS_ROOT / ".venv/bin/python"

DEFAULT_MODEL_ID = "Skywork/SkyReels-V3-A2V-19B"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def newest_output(directory: Path, before: set[Path]) -> Path:
    after = set(directory.glob("*with_audio.mp4"))

    created = sorted(
        after - before,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if created:
        return created[0]

    fallback = sorted(
        after,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not fallback:
        raise RuntimeError("No A2V output file was found")

    return fallback[0]


def run_shot(
    shot: dict[str, Any],
    model_id: str,
    resolution: str,
    seed: int,
    force: bool,
) -> None:
    image_file = Path(shot["image_file"])
    audio_file = Path(shot["audio_file"])
    output_file = Path(shot["video_file"])

    if not image_file.is_file():
        raise FileNotFoundError(
            f"Shot input image not found: {image_file}"
        )

    if not audio_file.is_file():
        raise FileNotFoundError(
            f"Shot audio file not found: {audio_file}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.is_file() and not force:
        print(f"Skipping existing clip: {output_file}")
        shot["status"] = "ready"
        shot["generated_file"] = str(output_file)
        return

    command = [
        str(SKYREELS_PYTHON),
        "generate_video.py",
        "--task_type",
        "talking_avatar",
        "--model_id",
        model_id,
        "--prompt",
        shot["prompt"],
        "--resolution",
        resolution,
        "--seed",
        str(seed),
        "--input_image",
        str(image_file),
        "--input_audio",
        str(audio_file),
    ]

    print()
    print("========================================")
    print(f"Generating: {shot['shot_id']}")
    print(f"Image     : {image_file}")
    print(f"Audio     : {audio_file}")
    print(f"Duration  : {shot['duration']} seconds")
    print(f"Output    : {output_file}")
    print("========================================")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HOME"] = (
        "/home/jupyter/asaf/video-ai-lab/models/huggingface"
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    result_dir = SKYREELS_ROOT / "result/talking_avatar"
    result_dir.mkdir(parents=True, exist_ok=True)

    before = set(result_dir.glob("*with_audio.mp4"))

    subprocess.run(
        command,
        cwd=SKYREELS_ROOT,
        env=env,
        check=True,
    )

    generated_file = newest_output(result_dir, before)
    generated_file.replace(output_file)

    shot["status"] = "ready"
    shot["generated_file"] = str(output_file)

    print(f"Saved: {output_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SkyReels A2V shots."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--shot",
        help="Generate only one shot ID.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate clips even when they already exist.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)

    model_config = manifest["models"]["singing_video"]
    model_id = model_config.get(
        "model_id",
        DEFAULT_MODEL_ID,
    )

    resolution = "720P"
    seed = 42

    shots = [
        shot
        for shot in manifest["shots"]
        if shot["mode"] == "a2v"
    ]

    if args.shot:
        shots = [
            shot
            for shot in shots
            if shot["shot_id"] == args.shot
        ]

        if not shots:
            raise KeyError(f"A2V shot not found: {args.shot}")

    print()
    print("========================================")
    print(" Video AI Lab - A2V Worker")
    print("========================================")
    print(f"Model : {model_id}")
    print("GPU   : 0")
    print(f"Shots : {len(shots)}")

    for index, shot in enumerate(shots):
        run_shot(
            shot=shot,
            model_id=model_id,
            resolution=resolution,
            seed=seed + index,
            force=args.force,
        )

        save_manifest(manifest_path, manifest)

    print()
    print("A2V generation complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
