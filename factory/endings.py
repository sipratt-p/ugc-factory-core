"""Clip ending quality — detect and fix abrupt endings.

Problems:
- Mouth open at last frame (mid-word)
- Audio cuts mid-sentence
- No visual closure (hard stop vs fade)

Fixes:
- Fade to black on last N frames
- Audio fade-out matching video
- Score ending quality for beam rejection
"""

import logging
import subprocess
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def add_fade_out(video_path: Path, output_path: Path,
                  video_fade_frames: int = 18, audio_fade_ms: int = 750) -> Path:
    """Add fade-to-black + audio fade-out to end of clip.

    Args:
        video_fade_frames: frames to fade (18 = 0.75s at 24fps)
        audio_fade_ms: audio fade duration in ms
    """
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True).stdout.strip())

    fade_start = dur - (video_fade_frames / 24.0)
    audio_fade_start = dur - (audio_fade_ms / 1000.0)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fade=t=out:st={fade_start}:d={video_fade_frames/24.0}",
        "-af", f"afade=t=out:st={audio_fade_start}:d={audio_fade_ms/1000.0}",
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        str(output_path),
    ], check=True, capture_output=True)

    return output_path


def score_ending(video_path: Path) -> dict:
    """Score how cleanly a clip ends.

    Checks:
    - Audio energy in last 200ms (high = still speaking = bad)
    - Mouth openness in last frame (open = mid-word = bad)
    """
    try:
        import soundfile as sf

        dur = float(subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True).stdout.strip())

        # Check audio energy in last 200ms
        audio_path = video_path.parent / "_ending_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", str(max(0, dur - 0.2)),
            "-vn", "-ar", "16000", "-ac", "1", str(audio_path),
        ], check=True, capture_output=True)

        data, sr = sf.read(str(audio_path))
        audio_path.unlink(missing_ok=True)

        tail_rms = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0
        ends_in_silence = tail_rms < 0.015

        # Check mouth in last frame
        mouth_open = None
        try:
            from .score_visual import _get_face_analyzer
            import cv2

            last_frame = video_path.parent / "_ending_frame.png"
            subprocess.run([
                "ffmpeg", "-y", "-sseof", "-0.042",
                "-i", str(video_path), "-frames:v", "1", "-q:v", "1",
                str(last_frame),
            ], check=True, capture_output=True)

            analyzer = _get_face_analyzer()
            if analyzer:
                img = cv2.imread(str(last_frame))
                faces = analyzer.get(img)
                if faces and hasattr(faces[0], 'kps') and faces[0].kps is not None:
                    kps = faces[0].kps
                    left_mouth = kps[3]
                    right_mouth = kps[4]
                    nose = kps[2]
                    mouth_vert = abs(nose[1] - (left_mouth[1] + right_mouth[1]) / 2)
                    mouth_horiz = abs(left_mouth[0] - right_mouth[0])
                    mouth_open = round(mouth_vert / max(mouth_horiz, 1e-6), 4)

            last_frame.unlink(missing_ok=True)
        except Exception:
            pass

        # Ending score: 0-1, higher = cleaner ending
        score = 1.0
        if not ends_in_silence:
            score -= 0.4  # still speaking at end
        if mouth_open is not None and mouth_open > 0.7:
            score -= 0.3  # mouth wide open

        return {
            "ending_tail_rms": round(tail_rms, 4),
            "ending_in_silence": ends_in_silence,
            "ending_mouth_open": mouth_open,
            "ending_score": round(max(0, score), 3),
        }

    except Exception as e:
        logger.warning(f"Ending scoring failed: {e}")
        return {"ending_score": None, "ending_error": str(e)}
