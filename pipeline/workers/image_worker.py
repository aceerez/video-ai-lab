#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from diffusers import QwenImageEditPlusPipeline
from PIL import Image


DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_config_path(config: dict[str, Any], dotted_path: str) -> Path:
    value: Any = config

    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Invalid source reference: {dotted_path}")
        value = value[part]

    if isinstance(value, dict):
        if "image" not in value:
            raise KeyError(
                f"Source reference does not contain an image: {dotted_path}"
            )
        value = value["image"]

    return Path(str(value)).resolve()


def open_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Source image not found: {path}")

        images.append(Image.open(path).convert("RGB"))

    return images


def build_negative_prompt(operation_type: str) -> str:
    common = """
different person, changed identity, changed ethnicity, changed age,
changed facial structure, changed eyes, changed nose, changed mouth,
changed cheeks, changed hairstyle, changed skin tone, deformed face,
asymmetrical face, duplicate person, extra limbs, extra fingers,
malformed hands, blurry face, cartoon, anime, illustration,
artificial skin, distorted body
"""

    if operation_type == "replace_background":
        return common + """
changed clothes, changed pose, changed body, changed camera angle,
changed framing, full body, additional person
"""

    if operation_type == "replace_clothing":
        return common + """
changed background, changed stage, changed lighting, changed pose,
changed camera angle, changed framing, hat, glasses, jewelry
"""

    if operation_type == "compose_people":
        return common + """
missing person, merged faces, fused bodies, duplicate child,
duplicate grandfather, extra person, open mouth, visible teeth,
aggressive movement
"""

    return common


def run_edit(
    pipe: QwenImageEditPlusPipeline,
    input_paths: list[Path],
    output_path: Path,
    prompt: str,
    operation_type: str,
    seed: int,
    steps: int,
    true_cfg_scale: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = open_images(input_paths)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    print()
    print("----------------------------------------")
    print(f"Operation : {operation_type}")
    print(f"Inputs    : {', '.join(str(path) for path in input_paths)}")
    print(f"Output    : {output_path}")
    print(f"Seed      : {seed}")
    print(f"Steps     : {steps}")
    print("----------------------------------------")

    with torch.inference_mode():
        result = pipe(
            image=images,
            prompt=prompt,
            negative_prompt=build_negative_prompt(operation_type),
            true_cfg_scale=true_cfg_scale,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=generator,
            num_images_per_prompt=1,
        )

    result.images[0].save(output_path)
    print(f"Saved: {output_path}")


def generate_asset(
    pipe: QwenImageEditPlusPipeline,
    config: dict[str, Any],
    manifest: dict[str, Any],
    asset_id: str,
    force: bool,
) -> None:
    project_assets = config.get("assets", {})
    manifest_assets = manifest.get("assets", {})

    if asset_id not in project_assets:
        raise KeyError(f"Asset not found in project YAML: {asset_id}")

    if asset_id not in manifest_assets:
        raise KeyError(f"Asset not found in manifest: {asset_id}")

    asset_config = project_assets[asset_id]
    asset_state = manifest_assets[asset_id]

    output_path = Path(asset_config["output"]).resolve()

    if output_path.is_file() and not force:
        print(f"Asset already exists, skipping: {output_path}")
        asset_state["status"] = "ready"
        asset_state["output"] = str(output_path)
        return

    source_refs = asset_config.get("sources", [])

    if not source_refs:
        raise ValueError(f"Asset has no sources: {asset_id}")

    original_sources = [
        resolve_config_path(config, source_ref)
        for source_ref in source_refs
    ]

    operations = asset_config.get("operations", [])

    if not operations:
        raise ValueError(f"Asset has no operations: {asset_id}")

    base_seed = int(config["project"].get("seed", 42))
    current_inputs = original_sources

    for index, operation in enumerate(operations):
        operation_type = operation["type"]
        prompt = operation["prompt"].strip()

        is_last_operation = index == len(operations) - 1

        if is_last_operation:
            operation_output = output_path
        else:
            intermediate = operation.get("intermediate_output")
            if not intermediate:
                raise ValueError(
                    f"Operation {operation_type} requires intermediate_output"
                )
            operation_output = Path(intermediate).resolve()

        if operation_output.is_file() and not force:
            print(f"Intermediate already exists, skipping: {operation_output}")
        else:
            run_edit(
                pipe=pipe,
                input_paths=current_inputs,
                output_path=operation_output,
                prompt=prompt,
                operation_type=operation_type,
                seed=base_seed + index,
                steps=50,
                true_cfg_scale=2.0,
            )

        current_inputs = [operation_output]

    if not output_path.is_file():
        raise RuntimeError(f"Asset generation failed: {output_path}")

    asset_state["status"] = "ready"
    asset_state["output"] = str(output_path)

    for shot in manifest.get("shots", []):
        if shot.get("asset_id") == asset_id:
            shot["image_file"] = str(output_path)

    print()
    print(f"Asset ready: {asset_id}")
    print(f"Output     : {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate image assets from the render manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--asset",
        required=True,
        help="Asset ID to generate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the asset even if it already exists.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    project_dir = manifest_path.parent
    project_yaml = project_dir / "project.resolved.yaml"

    manifest = load_json(manifest_path)
    config = load_yaml(project_yaml)

    image_model = config.get("models", {}).get("image", {})
    model_id = image_model.get("model_id", DEFAULT_MODEL_ID)

    print()
    print("========================================")
    print(" Video AI Lab - Image Asset Worker")
    print("========================================")
    print(f"Project : {manifest.get('title')}")
    print(f"Asset   : {args.asset}")
    print(f"Model   : {model_id}")
    print("GPU     : 0")
    print()

    print("Loading Qwen Image Edit...")

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)

    generate_asset(
        pipe=pipe,
        config=config,
        manifest=manifest,
        asset_id=args.asset,
        force=args.force,
    )

    save_json(manifest_path, manifest)

    print()
    print(f"Manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
