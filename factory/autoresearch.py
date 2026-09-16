"""Autoresearch loop — automated parameter optimization for UGC video quality.

Inspired by Karpathy's autoresearch pattern. Generates clips across a parameter
grid, scores each on multiple quality metrics using Claude vision + audio analysis,
and produces a ranked leaderboard of the best settings.

Runs overnight, produces a knowledge base of findings.
"""

import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

RESULTS_DIR = config.AUTORESEARCH_DIR  # env UGC_RESULTS_DIR, default <repo>/outputs/autoresearch


@dataclass
class TestConfig:
    """A single test configuration to evaluate."""
    test_id: str = ""
    name: str = ""
    description: str = ""
    # Video generation
    height: int = 864
    width: int = 480
    num_frames: int = 161
    seed: int = 42
    # Guidance
    identity_guidance: float = 2.0
    video_guidance: float = 3.0
    audio_guidance: float = 7.0
    video_rescale: float = 0.45
    # LoRA
    distilled_s1: float = 0.25
    distilled_s2: float = 0.5
    # Content
    prompt_visual: str = ""
    prompt_speech: str = ""
    prompt_sounds: str = "close-mic natural speech, quiet room, no music"
    # Reference
    first_frame: str = ""
    reference_audio: str = ""
    # Multi-clip
    num_clips: int = 1
    clip_assembly: str = "single"  # single | hardcut
    # Extra LoRAs (path:strength format)
    extra_loras: list = field(default_factory=list)

    def __post_init__(self):
        if not self.test_id:
            self.test_id = uuid.uuid4().hex[:8]


@dataclass
class TestResult:
    """Scores for a single test run."""
    test_id: str = ""
    config: dict = field(default_factory=dict)
    # Visual quality scores (1-10)
    teeth_clarity: float = 0
    skin_realism: float = 0
    face_stability: float = 0
    lip_sync: float = 0
    hand_body_quality: float = 0
    image_sharpness: float = 0
    motion_naturalness: float = 0
    # Audio scores (1-10)
    audio_clarity: float = 0
    speech_intelligibility: float = 0
    voice_naturalness: float = 0
    # Content scores (1-10)
    topic_coverage: float = 0
    content_density: float = 0
    pacing_quality: float = 0
    # Multi-clip scores
    seam_quality: float = 0  # 10 = invisible, 1 = jarring
    identity_consistency: float = 0  # across clips
    # Overall
    overall_score: float = 0
    # Metadata
    output_path: str = ""
    duration_s: float = 0
    resolution: str = ""
    generation_time_s: float = 0
    notes: str = ""


def generate_test_clip(cfg: TestConfig, gpu_id: int = None) -> Path:
    """Generate a single clip using ID-LoRA pipeline.

    Args:
        gpu_id: Which GPU to use (0 or 1). Defaults to config.VIDEO_GPU.
    """
    output_dir = RESULTS_DIR / cfg.test_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    prompt = (
        f"[VISUAL]: {cfg.prompt_visual} "
        f"[SPEECH]: {cfg.prompt_speech} "
        f"[SOUNDS]: {cfg.prompt_sounds}"
    )

    from .audio_utils import prepare_reference_image, resolve_reference_audio_path

    first_frame_path = Path(cfg.first_frame)
    prepared_first_frame = output_dir / "first_frame_processed.jpg"
    prepare_reference_image(first_frame_path, prepared_first_frame)
    first_frame_parent = first_frame_path.parent if first_frame_path.parent.exists() else output_dir
    reference_audio = resolve_reference_audio_path(
        first_frame_parent,
        Path(cfg.reference_audio) if cfg.reference_audio else None,
    )

    cmd = [
        str(config.ML_ENV_PYTHON),
        str(config.IDLORA_INFERENCE_SCRIPT),
        "--lora-path", str(config.IDLORA_DEFAULT_WEIGHTS),
        "--first-frame", str(prepared_first_frame),
        "--prompt", prompt,
        "--checkpoint", str(config.LTX_CHECKPOINT),
        "--text-encoder-path", str(config.LTX_TEXT_ENCODER),
        "--upsampler-path", str(config.LTX_UPSAMPLER),
        "--distilled-lora-path", str(config.LTX_DISTILLED_LORA),
        "--output-dir", str(output_dir),
        "--no-auto-resolution",
        "--height", str(cfg.height),
        "--width", str(cfg.width),
        "--num-frames", str(cfg.num_frames),
        "--seed", str(cfg.seed),
        # Full precision — no quantize
        "--video-guidance-scale", str(cfg.video_guidance),
        "--audio-guidance-scale", str(cfg.audio_guidance),
        "--video-rescale", str(cfg.video_rescale),
        "--identity-guidance-scale", str(cfg.identity_guidance),
        "--distilled-strength-s1", str(cfg.distilled_s1),
        "--distilled-strength-s2", str(cfg.distilled_s2),
    ]

    # Extra LoRAs (e.g., VBVR for temporal consistency)
    for lora_spec in cfg.extra_loras:
        cmd.extend(["--extra-lora", lora_spec])

    # Only pass reference audio if provided (pipeline works without it)
    if reference_audio and Path(reference_audio).exists():
        cmd.extend(["--reference-audio", str(reference_audio)])

    gpu = gpu_id if gpu_id is not None else config.VIDEO_GPU
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PATH": str(config.ML_ENV_PYTHON.parent) + ":/usr/bin:/bin",
        "HOME": str(Path.home()),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    start_time = time.time()
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=1800,
        cwd=str(config.IDLORA_REPO),
    )
    gen_time = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"[{cfg.test_id}] Generation failed: {result.stderr[-300:]}")
        with open(output_dir / "error.txt", "w") as f:
            f.write(result.stderr)
        raise RuntimeError(f"Generation failed for {cfg.test_id}")

    # Find and re-encode output
    mp4_files = sorted(output_dir.glob("output_*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No output for {cfg.test_id}")

    raw = mp4_files[-1]
    hq = output_dir / f"{cfg.test_id}_hq.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw),
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k", str(hq),
    ], check=True, capture_output=True)

    # Save generation metadata
    meta = {
        "generation_time_s": round(gen_time, 1),
        "raw_path": str(raw),
        "hq_path": str(hq),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[{cfg.test_id}] Generated in {gen_time:.0f}s: {hq}")
    return hq


def generate_multi_clip(cfg: TestConfig, scenes: list[dict]) -> Path:
    """Generate multi-clip video with hard cuts."""
    clips = []
    output_dir = RESULTS_DIR / cfg.test_id
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(scenes):
        clip_cfg = TestConfig(
            test_id=f"{cfg.test_id}_clip{i}",
            height=cfg.height, width=cfg.width,
            num_frames=cfg.num_frames, seed=cfg.seed,
            identity_guidance=cfg.identity_guidance,
            video_guidance=cfg.video_guidance,
            audio_guidance=cfg.audio_guidance,
            video_rescale=cfg.video_rescale,
            distilled_s1=cfg.distilled_s1, distilled_s2=cfg.distilled_s2,
            prompt_visual=scene["visual"],
            prompt_speech=scene["speech"],
            prompt_sounds=cfg.prompt_sounds,
            reference_audio=cfg.reference_audio,
        )

        # Last-frame conditioning
        if i == 0:
            clip_cfg.first_frame = cfg.first_frame
        else:
            clip_cfg.first_frame = str(frames_dir / f"last_frame_{i-1}.png")

        clip_path = generate_test_clip(clip_cfg)
        clips.append(clip_path)

        # Extract last frame
        subprocess.run([
            "ffmpeg", "-y", "-sseof", "-0.042",
            "-i", str(clip_path), "-frames:v", "1", "-q:v", "1",
            str(frames_dir / f"last_frame_{i}.png"),
        ], check=True, capture_output=True)

    # Hard-cut assembly
    list_file = output_dir / "concat.txt"
    with open(list_file, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")

    final = output_dir / f"{cfg.test_id}_final.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(final),
    ], check=True, capture_output=True)

    list_file.unlink(missing_ok=True)
    logger.info(f"[{cfg.test_id}] Multi-clip assembled: {final}")
    return final


def extract_evaluation_frames(video_path: Path, output_dir: Path, count: int = 6) -> list[Path]:
    """Extract evenly-spaced frames for visual evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())

    frames = []
    for i in range(count):
        t = (i / (count - 1)) * duration if count > 1 else 0
        out = output_dir / f"frame_{i:03d}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t),
            "-i", str(video_path), "-frames:v", "1", "-q:v", "1", str(out),
        ], check=True, capture_output=True)
        frames.append(out)
    return frames


def analyze_audio(video_path: Path, output_dir: Path) -> dict:
    """Extract audio metrics for scoring."""
    audio_path = output_dir / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ], check=True, capture_output=True)

    # Basic audio analysis
    try:
        import numpy as np
        import soundfile as sf
        data, sr = sf.read(str(audio_path))
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
        silence_ratio = float(np.mean(np.abs(data) < 0.01))
        # Spectral analysis for speech detection
        from scipy.fft import fft
        spectrum = np.abs(fft(data[:sr]))[:sr//2]
        speech_band = spectrum[100:4000]  # 100Hz-4kHz = speech range
        total_energy = np.sum(spectrum**2)
        speech_energy = np.sum(speech_band**2)
        speech_ratio = float(speech_energy / total_energy) if total_energy > 0 else 0

        return {
            "rms_level": round(rms, 4),
            "peak_level": round(peak, 4),
            "silence_ratio": round(silence_ratio, 3),
            "speech_band_ratio": round(speech_ratio, 3),
            "duration_s": round(len(data) / sr, 2),
            "has_speech": speech_ratio > 0.3 and rms > 0.01,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        audio_path.unlink(missing_ok=True)


def save_leaderboard(results: list[TestResult]):
    """Save ranked results to a leaderboard file."""
    ranked = sorted(results, key=lambda r: r.overall_score, reverse=True)

    leaderboard = {
        "generated": datetime.now().isoformat(),
        "total_tests": len(ranked),
        "rankings": [],
    }

    for i, r in enumerate(ranked):
        leaderboard["rankings"].append({
            "rank": i + 1,
            "test_id": r.test_id,
            "overall_score": r.overall_score,
            "resolution": r.resolution,
            "duration": r.duration_s,
            "generation_time": r.generation_time_s,
            "scores": {
                "teeth_clarity": r.teeth_clarity,
                "skin_realism": r.skin_realism,
                "face_stability": r.face_stability,
                "lip_sync": r.lip_sync,
                "image_sharpness": r.image_sharpness,
                "motion_naturalness": r.motion_naturalness,
                "audio_clarity": r.audio_clarity,
                "speech_intelligibility": r.speech_intelligibility,
                "topic_coverage": r.topic_coverage,
                "content_density": r.content_density,
                "pacing_quality": r.pacing_quality,
                "seam_quality": r.seam_quality,
                "identity_consistency": r.identity_consistency,
            },
            "config": r.config,
            "output": r.output_path,
            "notes": r.notes,
        })

    path = RESULTS_DIR / "leaderboard.json"
    with open(path, "w") as f:
        json.dump(leaderboard, f, indent=2)

    # Also save as readable markdown
    md_path = RESULTS_DIR / "leaderboard.md"
    with open(md_path, "w") as f:
        f.write("# UGC Autoresearch Leaderboard\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("| Rank | ID | Overall | Teeth | Skin | Sharpness | Lip-sync | Audio | Resolution | Duration | Notes |\n")
        f.write("|------|-----|---------|-------|------|-----------|----------|-------|------------|----------|-------|\n")
        for r in leaderboard["rankings"]:
            s = r["scores"]
            f.write(f"| {r['rank']} | {r['test_id']} | {r['overall_score']:.1f} | "
                    f"{s['teeth_clarity']:.1f} | {s['skin_realism']:.1f} | "
                    f"{s['image_sharpness']:.1f} | {s['lip_sync']:.1f} | "
                    f"{s['audio_clarity']:.1f} | {r['resolution']} | "
                    f"{r['duration']:.1f}s | {r['notes'][:30]} |\n")

    logger.info(f"Leaderboard saved: {path}")
    return path
