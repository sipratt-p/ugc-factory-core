"""Video generation engine using ID-LoRA inference_two_stage_hq.py.

This is the proven pipeline that produced the best quality clips (a369df).
Uses identity guidance + audio guidance + CFG rescaling for stable faces
and natural lip-sync. Runs as a subprocess with CUDA_VISIBLE_DEVICES isolation.

NOT ComfyUI — the ID-LoRA script has custom identity/audio guidance that
ComfyUI's native nodes don't replicate.
"""

import json
import shutil
import subprocess
import logging
import uuid
from pathlib import Path

from . import config
from .models import Persona, Script

logger = logging.getLogger(__name__)


def extract_last_frame(video_path: Path, output_path: Path) -> Path:
    subprocess.run([
        "ffmpeg", "-y", "-sseof", "-0.042",
        "-i", str(video_path), "-frames:v", "1", "-q:v", "1",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path


def generate_single_clip(
    prompt: str,
    first_frame_path: Path,
    output_dir: Path,
    reference_audio: Path = None,
    lora_weights: Path = None,
    height: int = None,
    width: int = None,
    num_frames: int = None,
    seed: int = None,
    filename_prefix: str = "output",
) -> Path:
    """Generate a single clip using ID-LoRA two-stage HQ pipeline.

    Uses V13 proven settings from config. All guidance parameters are
    centralized in config.py after 88+ experiments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    height = height or config.GENERATION_HEIGHT
    width = width or config.GENERATION_WIDTH
    num_frames = num_frames or config.NUM_FRAMES
    seed = seed or config.SEED
    lora = lora_weights or config.IDLORA_DEFAULT_WEIGHTS

    # Materialize a generation-ready first frame without repeatedly degrading
    # already-good portrait references.
    from .audio_utils import prepare_reference_image
    processed_frame = output_dir / "first_frame_processed.jpg"
    prepare_reference_image(first_frame_path, processed_frame)
    first_frame_for_gen = processed_frame

    cmd = [
        str(config.ML_ENV_PYTHON),
        str(config.IDLORA_INFERENCE_SCRIPT),
        "--lora-path", str(lora),
        "--first-frame", str(first_frame_for_gen),
        "--prompt", prompt,
        "--checkpoint", str(config.LTX_CHECKPOINT),
        "--text-encoder-path", str(config.LTX_TEXT_ENCODER),
        "--upsampler-path", str(config.LTX_UPSAMPLER),
        "--distilled-lora-path", str(config.LTX_DISTILLED_LORA),
        "--output-dir", str(output_dir),
        "--no-auto-resolution",
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--seed", str(seed),
        # V13 proven settings — locked after 88+ experiments
        "--video-guidance-scale", str(config.VIDEO_GUIDANCE),
        "--audio-guidance-scale", str(config.AUDIO_GUIDANCE),
        "--video-rescale", str(config.VIDEO_RESCALE),
        "--identity-guidance-scale", str(config.IDENTITY_GUIDANCE),
        "--distilled-strength-s1", str(config.DISTILLED_S1),
        "--distilled-strength-s2", str(config.DISTILLED_S2),
    ]

    if reference_audio and reference_audio.exists():
        cmd.extend(["--reference-audio", str(reference_audio)])

    env = {
        "CUDA_VISIBLE_DEVICES": str(config.VIDEO_GPU),
        "PATH": str(config.ML_ENV_PYTHON.parent) + ":/usr/bin:/bin",
        "HOME": str(Path.home()),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    logger.info(f"ID-LoRA generating: {height}x{width} → {height*2}x{width*2}, {num_frames} frames")
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=1800,  # 30 min per clip
        cwd=str(config.IDLORA_REPO),
    )

    if result.returncode != 0:
        logger.error(f"ID-LoRA failed:\n{result.stderr[-500:]}")
        raise RuntimeError(f"ID-LoRA inference failed: {result.stderr[-300:]}")

    # Find output video
    mp4_files = sorted(output_dir.glob("output_*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No output video in {output_dir}")

    output = mp4_files[-1]
    logger.info(f"Clip generated: {output}")
    return output


def reencode_highbitrate(input_path: Path, output_path: Path, bitrate: str = "15M") -> Path:
    """Re-encode at proper bitrate (ComfyUI/ID-LoRA default is ~2mbps)."""
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264", "-b:v", bitrate, "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path


def generate_clips(
    script: Script,
    persona: Persona,
    job_dir: Path,
    scene_bible=None,
    on_progress: callable = None,
    **kwargs,
) -> list[Path]:
    """Generate video clips using ID-LoRA with V13 proven settings.

    V13 settings (locked after 88+ experiments):
    - Identity guidance 1.5, Audio guidance 3.5
    - Always-original reference (no last-frame chaining — prevents drift)
    - 129 frames per clip, -preset slow
    - Scene bible drives [VISUAL] and [SOUNDS] prompts
    """
    video_dir = job_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Load or auto-generate scene bible
    if scene_bible is None:
        bible_path = config.PERSONAS_DIR / persona.slug / "scene_bible.json"
        if bible_path.exists():
            from .scene_bible import SceneBible
            scene_bible = SceneBible.load(bible_path)
            scene_bible.reference_image = str(persona.face_ref_path)
        else:
            from .scene_bible import auto_bible_from_image
            scene_bible = auto_bible_from_image(
                image_path=persona.face_ref_path,
                reference_audio=str(persona.voice_ref_path) if persona.voice_ref_path else "",
                persona_name=persona.name,
                cache_dir=config.PERSONAS_DIR / persona.slug,
            )

    # Voice reference — prefer persona-specific audio over generic placeholders
    from .audio_utils import (
        resolve_reference_audio_path,
        validate_reference_audio,
        normalize_reference_audio,
    )
    voice_ref = resolve_reference_audio_path(
        persona.face_ref_path.parent,
        persona.voice_ref_path,
    )
    if voice_ref and voice_ref.exists():
        check = validate_reference_audio(voice_ref)
        if check.get("needs_normalization"):
            logger.warning(f"Reference audio needs normalization: {check['warnings']}")
            normalized = job_dir / "reference_normalized.wav"
            voice_ref = normalize_reference_audio(voice_ref, normalized)
    else:
        logger.error("No usable voice reference available")

    if scene_bible is not None and voice_ref is not None:
        scene_bible.reference_audio = str(voice_ref)

    # Determine LoRA weights (per-persona override or default)
    lora = persona.lora_path or config.IDLORA_DEFAULT_WEIGHTS

    # Build sounds prompt from bible
    sounds = scene_bible.build_sounds_prompt() if scene_bible else None

    video_clips = []
    for i, scene in enumerate(script.scenes):
        clip_dir = video_dir / f"scene_{scene.index:03d}"

        # Always-original reference — V13 proven strategy (no drift)
        first_frame = persona.face_ref_path

        logger.info(f"Generating scene {i + 1}/{script.num_scenes}...")

        try:
            # Build prompt with scene bible visual + frozen [SOUNDS]
            if scene_bible:
                visual = scene_bible.build_visual_prompt()
                prompt = (
                    f"[VISUAL]: {visual} "
                    f"[SPEECH]: {scene.speech} "
                    f"[SOUNDS]: {sounds}"
                )
            else:
                prompt = scene.to_prompt(sounds)

            raw_clip = generate_single_clip(
                prompt=prompt,
                first_frame_path=first_frame,
                output_dir=clip_dir,
                reference_audio=voice_ref,
                lora_weights=lora,
                seed=config.SEED + i * 10,
            )

            # Re-encode at proper bitrate
            hq_clip = clip_dir / f"clip_hq_{uuid.uuid4().hex[:8]}.mp4"
            reencode_highbitrate(raw_clip, hq_clip)

            video_clips.append(hq_clip)
            logger.info(f"Scene {i + 1} done: {hq_clip}")

            if on_progress:
                on_progress(i + 1, script.num_scenes)

        except Exception as e:
            logger.error(f"Scene {i} failed: {e}")
            raise

    return video_clips
