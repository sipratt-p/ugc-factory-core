"""Video clip assembly — hard cut at speech pauses.

Uses hard cuts (jump cuts) between clips instead of crossfade blending.
This is how real YouTubers edit and it's more forgiving for AI-generated
clip boundaries. Each clip has its own natively generated audio, so
hard concat keeps lip-sync intact within each segment.
"""

import shutil
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def assemble_av(
    video_clips: list[Path],
    output_path: Path,
    use_handles: bool = False,
    handle_frames: int = 15,
    **kwargs,
) -> Path:
    """Assemble clips — with optional handle-based splice optimization.

    If use_handles=True, searches for the best splice point in the overlap
    region between clips (face/pose matching). Otherwise, hard cuts.
    """
    if use_handles and len(video_clips) > 1:
        from .handle_gen import assemble_with_handles
        result = assemble_with_handles(video_clips, output_path, handle_frames)
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(video_clips) == 1:
        shutil.copy2(video_clips[0], output_path)
        logger.info(f"Single clip, copied to {output_path}")
        return output_path

    logger.info(f"Assembling {len(video_clips)} clips with hard cuts...")

    # Write concat list
    list_file = output_path.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for clip in video_clips:
            f.write(f"file '{clip}'\n")

    # Hard concat — no re-encoding, just mux together
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # If copy fails (codec mismatch), re-encode
        logger.warning("Concat copy failed, re-encoding...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
            "-c:a", "aac", "-b:a", "256k",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    list_file.unlink(missing_ok=True)

    total_dur = get_video_duration(output_path)
    logger.info(f"Final video: {output_path} ({total_dur:.1f}s)")
    return output_path


def add_background_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    music_volume_db: float = -20,
    fade_in: float = 1.0,
    fade_out: float = 2.0,
) -> Path:
    """Mix background music under existing video audio."""
    duration = get_video_duration(video_path)
    fade_out_start = max(0, duration - fade_out)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-stream_loop", "-1", "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={music_volume_db}dB,"
        f"afade=t=in:d={fade_in},"
        f"afade=t=out:st={fade_out_start}:d={fade_out}[music];"
        f"[0:a][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Music overlay failed: {result.stderr[-200:]}")
    logger.info(f"Music added: {output_path}")
    return output_path


def list_music_library() -> list[Path]:
    from . import config
    music_dir = config.MUSIC_DIR
    if not music_dir.exists():
        return []
    return sorted(
        p for p in music_dir.iterdir()
        if p.suffix.lower() in (".mp3", ".wav", ".ogg", ".m4a", ".flac")
    )
