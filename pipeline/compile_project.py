from pathlib import Path
import argparse
import subprocess
import yaml
import json

ROOT = Path("/home/jupyter/asaf/video-ai-lab")


def audio_duration(audio: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio),
            ],
            text=True,
        ).strip()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    args = parser.parse_args()

    project_yaml = Path(args.project)
    cfg = yaml.safe_load(project_yaml.read_text())

    project_id = cfg["project"]["id"]
    title = cfg["project"]["title"]
    project_dir = ROOT / "projects" / project_id

    clips_dir = project_dir / "clips"
    images_dir = project_dir / "images"
    audio_dir = project_dir / "work" / "audio"
    final_dir = project_dir / "final"

    for directory in [clips_dir, images_dir, audio_dir, final_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    song = Path(cfg["inputs"]["song"]["file"])
    duration = audio_duration(song)

    cast = cfg["cast"]

    main_name = next(
        name
        for name, info in cast.items()
        if info["role"] == "main"
    )

    main_image = Path(cast[main_name]["image"])

    assets = {
        f"stage_{main_name}": {
            "asset_id": f"stage_{main_name}",
            "type": "image",
            "sources": [str(main_image)],
            "output": str(images_dir / "base_stage_final.png"),
            "intermediate_outputs": {
                "background_only": str(
                    images_dir / "base_stage_background.png"
                )
            },
            "prompt": (
                f"Create a photorealistic 1950s American television variety-show stage "
                f"featuring {main_name.capitalize()} from the reference image. "
                f"Bright pastel colors, checkerboard floor, chrome vintage microphone, "
                f"classic jukebox, warm studio lighting, authentic 1950s clothing, "
                f"happy audience atmosphere. Preserve the person's exact identity, age, "
                f"face, hairstyle, skin tone and body proportions."
            ),
            "status": "pending",
        }
    }

    multi_actor_asset_id = None

    for scene in cfg.get("story", []):
        actors = scene.get("actors", [])

        if len(actors) < 2:
            continue

        asset_id = "_".join(actors)

        if asset_id in assets:
            continue

        sources = [str(Path(cast[actor]["image"])) for actor in actors]

        assets[asset_id] = {
            "asset_id": asset_id,
            "type": "image",
            "sources": sources,
            "output": str(images_dir / f"{asset_id}.png"),
            "prompt": (
                f"Create a photorealistic 1950s television-stage scene with "
                f"{' and '.join(actor.capitalize() for actor in actors)} together. "
                f"Preserve each person's exact facial identity, age, hairstyle, skin tone "
                f"and body proportions. They share a warm natural family interaction. "
                f"Bright pastel studio, checkerboard floor, vintage microphone, warm lighting."
            ),
            "status": "pending",
        }

        if multi_actor_asset_id is None:
            multi_actor_asset_id = asset_id

    shots = []
    t = 0.0
    index = 1

    while t < duration:
        remaining = duration - t

        if index % 2 == 1:
            clip = min(10.0, remaining)
            shot_id = f"shot_{index:03d}_a2v"

            shots.append(
                {
                    "index": index,
                    "shot_id": shot_id,
                    "mode": "a2v",
                    "timeline_start": round(t, 2),
                    "duration": round(clip, 2),
                    "audio_start": round(t, 2),
                    "audio_file": str(audio_dir / f"{shot_id}.wav"),
                    "image_file": str(images_dir / "base_stage_final.png"),
                    "video_file": str(clips_dir / f"{shot_id}.mp4"),
                    "prompt": (
                        f"{main_name.capitalize()} performs naturally on a bright "
                        f"photorealistic 1950s American television variety-show stage. "
                        f"Accurate lip sync, stable identity, natural expression, "
                        f"subtle head and body movement, vintage microphone, pastel studio, "
                        f"warm lighting. No face distortion. No extra limbs."
                    ),
                    "asset_id": f"stage_{main_name}",
                    "status": "pending",
                }
            )

        else:
            clip = min(5.0, remaining)
            shot_id = f"shot_{index:03d}_r2v"

            use_multi_actor = (
                multi_actor_asset_id is not None
                and index == 4
            )

            if use_multi_actor:
                image_file = assets[multi_actor_asset_id]["output"]
                asset_id = multi_actor_asset_id
                prompt = (
                    f"{main_name.capitalize()} and Grandpa share a warm natural family moment "
                    f"on the bright 1950s television stage. Preserve both identities exactly. "
                    f"They are not singing. Keep mouths closed or naturally relaxed. "
                    f"No speech, no lip sync, natural body movement only."
                )
            else:
                image_file = str(images_dir / "base_stage_final.png")
                asset_id = f"stage_{main_name}"
                prompt = (
                    f"{main_name.capitalize()} moves naturally through the bright "
                    f"1950s television studio. The character is not singing. "
                    f"Keep the mouth closed or naturally relaxed. No speech, no lip sync. "
                    f"Motion comes from the eyes, head, hands and body. "
                    f"Preserve exact identity. No face distortion. No extra limbs."
                )

            shots.append(
                {
                    "index": index,
                    "shot_id": shot_id,
                    "mode": "r2v",
                    "timeline_start": round(t, 2),
                    "duration": round(clip, 2),
                    "audio_start": None,
                    "audio_file": None,
                    "image_file": image_file,
                    "video_file": str(clips_dir / f"{shot_id}.mp4"),
                    "prompt": prompt,
                    "asset_id": asset_id,
                    "status": "pending",
                }
            )

        t += clip
        index += 1

    manifest = {
        "project_id": project_id,
        "title": title,
        "song_file": str(song),
        "song_duration": round(duration, 2),
        "character_image": str(main_image),
        "models": {
            "image": {
                "engine": "qwen_image_edit_2511",
                "gpu": 2,
            },
            "cinematic_video": {
                "engine": "skyreels_r2v",
                "gpu": 1,
            },
            "singing_video": {
                "engine": "skyreels_a2v",
                "gpu": 0,
            },
        },
        "shots": shots,
        "final_output": str(final_dir / cfg["output"]["filename"]),
        "references": {
            name: str(Path(info["image"]))
            for name, info in cast.items()
        },
        "assets": assets,
        "final_status": "pending",
    }

    out = project_dir / "render_manifest.json"
    out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    for shot in shots:
        if shot["mode"] != "a2v":
            continue

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(shot["audio_start"]),
                "-i",
                str(song),
                "-t",
                str(shot["duration"]),
                "-vn",
                "-acodec",
                "pcm_s16le",
                shot["audio_file"],
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print()
    print("Generated:", out)
    print("Duration:", round(duration, 2))
    print("Shots:", len(shots))
    print("Assets:", ", ".join(assets.keys()))


if __name__ == "__main__":
    main()
