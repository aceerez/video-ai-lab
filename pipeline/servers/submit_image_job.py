#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/jupyter/asaf/video-ai-lab")
DEFAULT_RUNTIME = ROOT / "runtime/image"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--job-id", required=True)

    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Input image. Repeat for multiple reference images.",
    )

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=3.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait", action="store_true")

    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME,
    )

    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()

    job = {
        "job_id": args.job_id,
        "images": [
            str(Path(value).expanduser().resolve())
            for value in args.image
        ],
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "output": str(Path(args.output).expanduser().resolve()),
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "force": args.force,
    }

    destination = runtime / "jobs" / f"{args.job_id}.json"
    atomic_write(destination, job)

    print("Submitted:", args.job_id)

    if not args.wait:
        return 0

    done = runtime / "done" / f"{args.job_id}.json"
    failed = runtime / "failed" / f"{args.job_id}.json"
    processing = runtime / "processing" / f"{args.job_id}.json"

    while True:
        if done.is_file():
            result = read_json(done)
            print("READY:", result["output"])
            return 0

        if failed.is_file():
            result = read_json(failed)
            raise RuntimeError(result.get("error", "Image job failed"))

        state = "RUNNING" if processing.is_file() else "QUEUED"
        print(f"{args.job_id}: {state}")
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
