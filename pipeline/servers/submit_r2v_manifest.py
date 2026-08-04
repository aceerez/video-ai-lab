#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/jupyter/asaf/video-ai-lab")
DEFAULT_RUNTIME = ROOT / "runtime/r2v"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest.resolve())
    runtime = args.runtime_dir.resolve()

    shots = [
        shot
        for shot in manifest["shots"]
        if shot["mode"] == "r2v"
    ]

    submitted: list[str] = []

    for shot in shots:
        job_id = shot["shot_id"]

        job = {
            "job_id": job_id,
            "image": shot["image_file"],
            "prompt": shot["prompt"],
            "duration": int(round(float(shot["duration"]))),
            "output": shot["video_file"],
            "seed": 42 + int(shot["index"]) - 1,
            "resolution": "720P",
            "force": args.force,
        }

        atomic_write(
            runtime / "jobs" / f"{job_id}.json",
            job,
        )

        submitted.append(job_id)
        print("Submitted:", job_id)

    if not args.wait:
        return 0

    remaining = set(submitted)

    while remaining:
        states: list[str] = []

        for job_id in sorted(remaining):
            done = runtime / "done" / f"{job_id}.json"
            failed = runtime / "failed" / f"{job_id}.json"
            processing = runtime / "processing" / f"{job_id}.json"

            if done.is_file():
                print("READY:", job_id)
                remaining.remove(job_id)
                continue

            if failed.is_file():
                result = read_json(failed)
                raise RuntimeError(
                    f"{job_id} failed: {result.get('error')}"
                )

            if processing.is_file():
                states.append(f"{job_id}=RUNNING")
            else:
                states.append(f"{job_id}=QUEUED")

        if remaining:
            print(" | ".join(states))
            time.sleep(5)

    print("All R2V jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
