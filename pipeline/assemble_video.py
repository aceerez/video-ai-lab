#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(command: list[str]) -> None:
    print()
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_clip(
    source: Path,
    output: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
    force: bool,
) -> None:
    if output.is_file() and not force:
        print(f"Skipping normalized clip: {output}")
        return

    if not source.is_file():
        raise FileNotFoundError(f"Video clip not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)

    # Creates a blurred widescreen background when the source is vertical,
    # while preserving the original video in the center.
    filter_complex = (
        f"[0:v]split=2[background][foreground];"
        f"[background]"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"boxblur=20:10[background_blurred];"
        f"[foreground]"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        f"[foreground_scaled];"
        f"[background_blurred][foreground_scaled]"
        f"overlay=(W-w)/2:(H-h)/2,"
        f"fps={fps},format=yuv420p[video]"
    )

    run([
        "ffmpeg",
        "-y",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-filter_complex", filter_complex,
        "-map", "[video]",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble all generated shots into the final movie."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate normalized clips and final output.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    project_dir = manifest_path.parent

    project_yaml = project_dir / "project.resolved.yaml"

    width = 1280
    height = 720
    fps = 25

    # Use values from the manifest/project defaults when available.
    shots = sorted(
        manifest["shots"],
        key=lambda shot: int(shot["index"]),
    )

    if not shots:
        raise RuntimeError("The manifest contains no shots")

    normalized_dir = project_dir / "work/normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    normalized_files: list[Path] = []

    print()
    print("========================================")
    print(" Video AI Lab - Final Assembly")
    print("========================================")
    print(f"Shots      : {len(shots)}")
    print(f"Resolution : {width}x{height}")
    print(f"FPS        : {fps}")

    for shot in shots:
        source = Path(shot["video_file"])
        normalized = normalized_dir / f"{shot['shot_id']}.mp4"

        normalize_clip(
            source=source,
            output=normalized,
            duration=float(shot["duration"]),
            width=width,
            height=height,
            fps=fps,
            force=args.force,
        )

        normalized_files.append(normalized)

    concat_file = project_dir / "work/concat.txt"
    concat_file.parent.mkdir(parents=True, exist_ok=True)

    concat_file.write_text(
        "\n".join(
            f"file '{path.resolve()}'"
            for path in normalized_files
        ) + "\n",
        encoding="utf-8",
    )

    video_only = project_dir / "work/video_timeline.mp4"

    run([
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(video_only),
    ])

    song_file = Path(manifest["song_file"])

    if not song_file.is_file():
        raise FileNotFoundError(f"Master song not found: {song_file}")

    final_output = Path(manifest["final_output"])
    final_output.parent.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg",
        "-y",
        "-i", str(video_only),
        "-i", str(song_file),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "256k",
        "-shortest",
        "-movflags", "+faststart",
        str(final_output),
    ])

    manifest["final_status"] = "ready"
    manifest["final_output"] = str(final_output)

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("========================================")
    print(" Final video ready")
    print("========================================")
    print(final_output)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
