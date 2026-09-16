"""Animated subtitle generation for UGC videos.

Uses faster-whisper for word-level timestamps from the video's audio,
then generates ASS subtitle files with TikTok-style karaoke effects.
Burns subtitles into video via ffmpeg.
"""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from video as WAV for transcription."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_word_timestamps(audio_path: Path, device: str = "cpu") -> list[dict]:
    """Get word-level timestamps using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster-whisper not installed, falling back to script-based timing")
        return []

    model = WhisperModel("base", device=device, compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })
    return words


def estimate_word_timestamps(scenes: list, clip_duration: float = 5.0) -> list[dict]:
    """Estimate word timestamps from script scenes when whisper isn't available."""
    words = []
    for scene in scenes:
        scene_start = scene.index * clip_duration
        scene_words = scene.speech.split()
        if not scene_words:
            continue
        total_chars = sum(len(w) for w in scene_words)
        current = scene_start + 0.3  # small offset
        for w in scene_words:
            word_dur = max(0.15, (len(w) / total_chars) * (clip_duration - 0.6))
            words.append({
                "word": w,
                "start": round(current, 3),
                "end": round(current + word_dur, 3),
            })
            current += word_dur
    return words


def generate_ass_subtitles(
    word_timings: list[dict],
    output_path: Path,
    style: str = "karaoke",
    words_per_line: int = 3,
    font: str = "Montserrat",
    font_size: int = 56,
    canvas_width: int = 544,
    canvas_height: int = 960,
) -> Path:
    """Generate ASS subtitle file with TikTok-style animated text.

    Styles:
    - karaoke: words fill with highlight color as spoken (ASS \\k tags)
    - pop: active word scales up and changes color
    """
    try:
        import pysubs2
    except ImportError:
        logger.warning("pysubs2 not installed, skipping subtitle generation")
        return None

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(canvas_width)
    subs.info["PlayResY"] = str(canvas_height)

    if style == "karaoke":
        sub_style = pysubs2.SSAStyle(
            fontname=font,
            fontsize=font_size,
            bold=True,
            primarycolor=pysubs2.Color(255, 255, 255, 0),       # white
            secondarycolor=pysubs2.Color(0, 255, 255, 0),        # yellow fill
            outlinecolor=pysubs2.Color(0, 0, 0, 0),              # black outline
            backcolor=pysubs2.Color(0, 0, 0, 128),               # semi-transparent bg
            outline=3,
            shadow=0,
            alignment=2,  # bottom center
            marginv=120,
        )
    else:  # pop style
        sub_style = pysubs2.SSAStyle(
            fontname=font,
            fontsize=font_size,
            bold=True,
            primarycolor=pysubs2.Color(255, 255, 255, 0),
            outlinecolor=pysubs2.Color(0, 0, 0, 0),
            outline=3,
            shadow=0,
            alignment=2,
            marginv=120,
        )

    subs.styles["Default"] = sub_style

    # Group words into lines
    for i in range(0, len(word_timings), words_per_line):
        chunk = word_timings[i:i + words_per_line]
        if not chunk:
            continue

        start_ms = int(chunk[0]["start"] * 1000)
        end_ms = int(chunk[-1]["end"] * 1000)

        if style == "karaoke":
            text = ""
            for w in chunk:
                dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                text += f"{{\\k{dur_cs}}}{w['word']} "

            event = pysubs2.SSAEvent(
                start=start_ms, end=end_ms + 200,
                style="Default", text=text.strip(),
            )
            subs.events.append(event)
        else:
            # Pop style — one event per active word
            for j, w in enumerate(chunk):
                parts = []
                for k, cw in enumerate(chunk):
                    if k == j:
                        parts.append(f"{{\\c&H00FFFF&\\fscx115\\fscy115}}{cw['word']}{{\\r}}")
                    else:
                        parts.append(cw["word"])

                event = pysubs2.SSAEvent(
                    start=int(w["start"] * 1000),
                    end=int(w["end"] * 1000),
                    style="Default",
                    text=" ".join(parts),
                )
                subs.events.append(event)

    subs.save(str(output_path))
    logger.info(f"Subtitles saved: {output_path} ({len(subs.events)} events)")
    return output_path


def burn_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
) -> Path:
    """Burn ASS subtitles into video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Subtitle burn failed: {result.stderr[-300:]}")
        raise RuntimeError(f"Subtitle burn failed: {result.stderr[-200:]}")

    logger.info(f"Subtitles burned: {output_path}")
    return output_path


def add_subtitles_to_video(
    video_path: Path,
    output_path: Path,
    scenes: list = None,
    style: str = "karaoke",
    words_per_line: int = 3,
    use_whisper: bool = True,
) -> Path:
    """Full subtitle pipeline: extract audio → timestamps → ASS → burn.

    If scenes are provided and whisper fails, falls back to estimated timing.
    """
    work_dir = output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # Get word timestamps
    word_timings = []
    if use_whisper:
        audio_path = work_dir / "subtitle_audio.wav"
        extract_audio(video_path, audio_path)
        word_timings = get_word_timestamps(audio_path)
        audio_path.unlink(missing_ok=True)

    if not word_timings and scenes:
        logger.info("Using estimated timing from script")
        word_timings = estimate_word_timestamps(scenes)

    if not word_timings:
        logger.warning("No word timings available, skipping subtitles")
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path

    # Generate ASS file
    ass_path = work_dir / "subtitles.ass"
    generate_ass_subtitles(word_timings, ass_path, style=style, words_per_line=words_per_line)

    if ass_path and ass_path.exists():
        burn_subtitles(video_path, ass_path, output_path)
        ass_path.unlink(missing_ok=True)
    else:
        import shutil
        shutil.copy2(video_path, output_path)

    return output_path
