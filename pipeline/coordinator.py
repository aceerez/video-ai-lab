#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Shot:
    index: int
    shot_id: str
    mode: str
    timeline_start: float
    duration: float
    audio_start: float | None
    audio_file: str | None
    image_file: str
    video_file: str
    prompt: str


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def audio_duration(path: Path) -> float:
    output = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(output)


def validate_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def build_prompt(
    mode: str,
    character_name: str,
    concept: str,
    visual_style: str,
    identity_rules: str,
    shot_number: int,
) -> str:
    if mode == "a2v":
        action = (
            f"{character_name} sings naturally to the camera with accurate "
            "lip sync, gentle facial expressions, subtle head movement, "
            "stable identity and a mostly static camera."
        )
    else:
        actions = [
            f"{character_name} gently sways and gives one friendly wave with a closed-mouth smile.",
            f"{character_name} takes two small rhythmic dance steps while keeping a closed-mouth smile.",
            f"{character_name} interacts gently with a friendly colorful puppet beside him.",
            f"{character_name} looks toward the audience and lightly claps to the rhythm.",
            f"A wide celebratory stage shot with {character_name}, balloons, warm lights and subtle confetti.",
        ]
        action = actions[(shot_number - 1) % len(actions)]

    return " ".join([
        concept.strip(),
        visual_style.strip(),
        identity_rules.strip(),
        action,
        "Natural realistic motion. No face distortion. No extra limbs.",
    ])


def create_audio_segment(
    source: Path,
    output: Path,
    start: float,
    duration: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output),
    ], check=True)


def create_plan(config: dict[str, Any], config_path: Path) -> list[Shot]:
    project = config["project"]
    inputs = config["inputs"]
    creative = config["creative"]
    planning = config["planning"]

    project_dir = Path(project["output_dir"])
    audio_source = Path(inputs["audio"]["file"])
    character_image = Path(inputs["character"]["image"])
    character_name = inputs["character"]["name"]

    validate_file(audio_source, "Audio")
    validate_file(character_image, "Character image")

    total_duration = audio_duration(audio_source)

    singing_duration = float(planning.get("singing_clip_seconds", 10))
    cinematic_duration = float(planning.get("cinematic_clip_seconds", 5))
    next_mode = str(planning.get("start_with", "a2v"))

    work_dir = project_dir / "work"
    audio_dir = work_dir / "audio"
    image_dir = project_dir / "images"
    clip_dir = project_dir / "clips"
    final_dir = project_dir / "final"

    for directory in [audio_dir, image_dir, clip_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, project_dir / "project.resolved.yaml")

    shots: list[Shot] = []
    cursor = 0.0
    shot_number = 1

    while cursor < total_duration - 0.01:
        mode = next_mode
        requested_duration = (
            singing_duration if mode == "a2v" else cinematic_duration
        )
        duration = min(requested_duration, total_duration - cursor)

        shot_id = f"shot_{shot_number:03d}_{mode}"
        audio_file: str | None = None
        audio_start: float | None = None

        if mode == "a2v":
            segment_path = audio_dir / f"{shot_id}.wav"
            create_audio_segment(
                source=audio_source,
                output=segment_path,
                start=cursor,
                duration=duration,
            )
            audio_file = str(segment_path)
            audio_start = cursor

        prompt = build_prompt(
            mode=mode,
            character_name=character_name,
            concept=creative["concept"],
            visual_style=creative["visual_style"],
            identity_rules=creative["identity_rules"],
            shot_number=shot_number,
        )

        shots.append(Shot(
            index=shot_number,
            shot_id=shot_id,
            mode=mode,
            timeline_start=round(cursor, 3),
            duration=round(duration, 3),
            audio_start=round(audio_start, 3) if audio_start is not None else None,
            audio_file=audio_file,
            image_file=str(image_dir / f"{shot_id}.png"),
            video_file=str(clip_dir / f"{shot_id}.mp4"),
            prompt=prompt,
        ))

        cursor += duration
        shot_number += 1
        next_mode = "r2v" if mode == "a2v" else "a2v"

    manifest = {
        "project_id": project["id"],
        "title": project["title"],
        "song_file": str(audio_source),
        "song_duration": round(total_duration, 3),
        "character_image": str(character_image),
        "models": config["models"],
        "shots": [asdict(shot) for shot in shots],
        "final_output": str(final_dir / config["export"]["filename"]),
    }

    manifest_path = project_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return shots


def main() -> int:
    parser = argparse.ArgumentParser(description="Video AI Lab coordinator")
    parser.add_argument("project_yaml", type=Path)
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ERROR: ffmpeg and ffprobe are required.", file=sys.stderr)
        return 1

    validate_file(args.project_yaml, "Project YAML")

    with args.project_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    shots = create_plan(config, args.project_yaml)
    project_dir = Path(config["project"]["output_dir"])

    print()
    print("========================================")
    print(" Video AI Lab - Project Plan")
    print("========================================")
    print(f"Project : {config['project']['title']}")
    print(f"Audio   : {config['inputs']['audio']['file']}")
    print(f"Shots   : {len(shots)}")
    print()

    for shot in shots:
        print(
            f"{shot.index:02d}  "
            f"{shot.mode.upper():3s}  "
            f"start={shot.timeline_start:6.2f}s  "
            f"duration={shot.duration:5.2f}s  "
            f"audio={shot.audio_file or '-'}"
        )

    print()
    print(f"Manifest: {project_dir / 'render_manifest.json'}")
    print("Audio splitting complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
