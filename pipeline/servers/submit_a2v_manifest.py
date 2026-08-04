#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/jupyter/asaf/video-ai-lab")
DEFAULT_RUNTIME = ROOT / "runtime/a2v"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
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
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    runtime = args.runtime_dir.resolve()

    manifest = load_json(manifest_path)

    a2v_shots = [
        shot
        for shot in manifest["shots"]
        if shot["mode"] == "a2v"
    ]

    submitted: list[str] = []

    for shot in a2v_shots:
        job_id = shot["shot_id"]

        job = {
            "job_id": job_id,
            "image": shot["image_file"],
            "audio": shot["audio_file"],
            "prompt": shot["prompt"],
            "output": shot["video_file"],
            "seed": 42 + int(shot["index"]) - 1,
            "resolution": "720P",
            "force": args.force,
        }

        destination = runtime / "jobs" / f"{job_id}.json"
        atomic_write_json(destination, job)

        submitted.append(job_id)
        print("Submitted:", job_id)

    if not args.wait:
        return 0

    print()
    print("Waiting for A2V jobs...")

    pending = set(submitted)

    while pending:
        for job_id in list(pending):
            done_file = runtime / "done" / f"{job_id}.json"
            failed_file = runtime / "failed" / f"{job_id}.json"

            if done_file.is_file():
                print("READY:", job_id)
                pending.remove(job_id)
                continue

            if failed_file.is_file():
                failure = load_json(failed_file)
                raise RuntimeError(
                    f"{job_id} failed: {failure.get('error')}"
                )

        if pending:
            print("Pending:", ", ".join(sorted(pending)))
            time.sleep(5)

    print()
    print("All A2V jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
