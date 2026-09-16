"""Handle generation — generate extra frames for optimal cut placement.

Instead of cutting at arbitrary clip boundaries, generate 10-15 extra frames
on the tail of each clip and head of the next. Then search the overlap region
for the frame pair where face/pose/lighting match best across the two clips.

This is standard film editing practice — editors always shoot with handles
and find the natural splice point in post.
"""

import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# How many extra frames to generate on each side
HANDLE_FRAMES = 15  # ~0.6s at 24fps


def extract_tail_frames(video_path: Path, output_dir: Path, count: int = HANDLE_FRAMES) -> list[Path]:
    """Extract the last N frames from a video clip."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration and fps
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True).stdout.strip())

    fps = 24.0
    total_frames = int(dur * fps)
    start_frame = max(0, total_frames - count)

    frames = []
    for i in range(count):
        frame_num = start_frame + i
        t = frame_num / fps
        out = output_dir / f"tail_{i:03d}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t),
            "-i", str(video_path), "-frames:v", "1", "-q:v", "1", str(out),
        ], check=True, capture_output=True)
        frames.append(out)

    return frames


def extract_head_frames(video_path: Path, output_dir: Path, count: int = HANDLE_FRAMES) -> list[Path]:
    """Extract the first N frames from a video clip."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = 24.0
    frames = []
    for i in range(count):
        t = i / fps
        out = output_dir / f"head_{i:03d}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t),
            "-i", str(video_path), "-frames:v", "1", "-q:v", "1", str(out),
        ], check=True, capture_output=True)
        frames.append(out)

    return frames


def _get_audio_safe_frames(video_path: Path, region: str, count: int) -> set[int]:
    """Get frame indices that are safe to cut at (silence or word boundaries).

    Uses WhisperX for word timestamps and energy analysis for silence detection.
    Returns set of frame indices where cutting won't chop a word.
    """
    import soundfile as sf

    # Extract audio
    audio_path = video_path.parent / f"_splice_audio_{region}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ], check=True, capture_output=True)

    data, sr = sf.read(str(audio_path))
    duration = len(data) / sr
    fps = 24.0

    safe_frames = set()
    frame_rms = {}  # store RMS per frame for ranking

    # Method 1 (PRIMARY): Energy-based silence detection
    # Works reliably on short segments where WhisperX may fail
    frame_duration = 1.0 / fps
    for i in range(count):
        if region == "tail":
            t = duration - (count - i) / fps
        else:
            t = i / fps

        if t < 0 or t >= duration:
            continue

        # Check RMS energy in a 100ms window around this frame
        center_sample = int(t * sr)
        window = int(0.05 * sr)  # 50ms each side
        start_s = max(0, center_sample - window)
        end_s = min(len(data), center_sample + window)
        chunk = data[start_s:end_s]

        if len(chunk) > 0:
            rms = float(np.sqrt(np.mean(chunk**2)))
            frame_rms[i] = rms
            if rms < 0.02:  # silence threshold (slightly relaxed)
                safe_frames.add(i)

    # Method 2: WhisperX word boundaries (if available)
    try:
        import whisperx
        model = whisperx.load_model("base", device="cpu", compute_type="int8")
        audio_data = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio_data, language="en")

        # Find gaps between words
        for seg in result.get("segments", []):
            words = seg.get("words", [])
            for j in range(len(words) - 1):
                gap_start = words[j].get("end", 0)
                gap_end = words[j + 1].get("start", 0)
                gap_duration = gap_end - gap_start

                if gap_duration > 0.05:  # >50ms gap = word boundary
                    gap_mid = (gap_start + gap_end) / 2
                    # Convert to frame index
                    for i in range(count):
                        if region == "tail":
                            t = duration - (count - i) / fps
                        else:
                            t = i / fps
                        if abs(t - gap_mid) < frame_duration:
                            safe_frames.add(i)
    except Exception as e:
        logger.warning(f"WhisperX word boundary detection failed: {e}")

    audio_path.unlink(missing_ok=True)

    # If no silence found, pick the 5 quietest frames as "safe"
    if not safe_frames and frame_rms:
        sorted_by_rms = sorted(frame_rms.items(), key=lambda x: x[1])
        for idx, rms in sorted_by_rms[:5]:
            safe_frames.add(idx)
        logger.info(f"    No silence found — using 5 quietest frames (min RMS={sorted_by_rms[0][1]:.4f})")

    # Ultimate fallback
    if not safe_frames:
        safe_frames = set(range(count))

    return safe_frames


def find_best_splice(
    tail_frames: list[Path],
    head_frames: list[Path],
    tail_video: Path = None,
    head_video: Path = None,
) -> dict:
    """Find the optimal splice point between two clips.

    Audio-aware: only considers frames at silence or word boundaries.
    Then picks the best visual match from those safe frames.

    Priority: audio timing primary, visual match secondary.
    """
    from .score_visual import _load_image, _get_face_embedding, _cosine_similarity
    import cv2

    # Get audio-safe frame indices
    tail_safe = set(range(len(tail_frames)))
    head_safe = set(range(len(head_frames)))

    if tail_video and tail_video.exists():
        logger.info("    Analyzing tail audio for safe cut points...")
        tail_safe = _get_audio_safe_frames(tail_video, "tail", len(tail_frames))
        logger.info(f"    {len(tail_safe)}/{len(tail_frames)} tail frames are audio-safe")

    if head_video and head_video.exists():
        logger.info("    Analyzing head audio for safe cut points...")
        head_safe = _get_audio_safe_frames(head_video, "head", len(head_frames))
        logger.info(f"    {len(head_safe)}/{len(head_frames)} head frames are audio-safe")

    # Get face embeddings for all frames
    tail_embs = []
    tail_grays = []
    for fp in tail_frames:
        try:
            img = _load_image(fp)
            emb = _get_face_embedding(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            gray = cv2.resize(gray, (160, 288))
            tail_embs.append(emb)
            tail_grays.append(gray)
        except Exception:
            tail_embs.append(None)
            tail_grays.append(None)

    head_embs = []
    head_grays = []
    for fp in head_frames:
        try:
            img = _load_image(fp)
            emb = _get_face_embedding(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            gray = cv2.resize(gray, (160, 288))
            head_embs.append(emb)
            head_grays.append(gray)
        except Exception:
            head_embs.append(None)
            head_grays.append(None)

    best_score = -1
    best_tail_idx = len(tail_frames) - 1
    best_head_idx = 0
    best_breakdown = {}

    # ONLY search audio-safe frame pairs
    for ti in tail_safe:
        for hi in head_safe:
            if ti >= len(tail_embs) or hi >= len(head_embs):
                continue
            if tail_embs[ti] is None or head_embs[hi] is None:
                continue

            face_sim = _cosine_similarity(tail_embs[ti], head_embs[hi])

            pixel_sim = 0
            if tail_grays[ti] is not None and head_grays[hi] is not None:
                t_norm = (tail_grays[ti] - tail_grays[ti].mean()) / (tail_grays[ti].std() + 1e-8)
                h_norm = (head_grays[hi] - head_grays[hi].mean()) / (head_grays[hi].std() + 1e-8)
                pixel_sim = float(np.mean(t_norm * h_norm))

            # Combined: 60% face, 20% pixel, 20% bonus for being in silence
            combined = face_sim * 0.6 + max(0, pixel_sim) * 0.2

            # Bonus if both frames are in silence
            if ti in tail_safe and hi in head_safe:
                combined += 0.2

            if combined > best_score:
                best_score = combined
                best_tail_idx = ti
                best_head_idx = hi
                best_breakdown = {
                    "face_similarity": round(face_sim, 4),
                    "pixel_similarity": round(pixel_sim, 4),
                    "audio_safe_tail": ti in tail_safe,
                    "audio_safe_head": hi in head_safe,
                }

    return {
        "tail_cut_frame": best_tail_idx,
        "head_cut_frame": best_head_idx,
        "splice_score": round(best_score, 4),
        "breakdown": best_breakdown,
        "tail_cut_time_offset": round(best_tail_idx / 24.0, 3),
        "head_cut_time_offset": round(best_head_idx / 24.0, 3),
        "audio_safe_tail_count": len(tail_safe),
        "audio_safe_head_count": len(head_safe),
    }


def trim_clip_tail(video_path: Path, output_path: Path, frames_to_keep: int) -> Path:
    """Trim a clip to keep only the first N frames (cut the tail)."""
    duration = frames_to_keep / 24.0
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path


def trim_clip_head(video_path: Path, output_path: Path, skip_frames: int) -> Path:
    """Trim a clip by skipping the first N frames (cut the head)."""
    skip_time = skip_frames / 24.0
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", str(skip_time),
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path


def assemble_with_handles(
    clips: list[Path],
    output_path: Path,
    handle_frames: int = 24,
    crossfade_frames: int = 4,
) -> dict:
    """Assemble multi-clip video with optimal splice points + cross-dissolve.

    For each pair of adjacent clips:
    1. Extract tail frames from clip N (using handle_frames search window)
    2. Extract head frames from clip N+1
    3. Find best splice point (audio-safe + face/pixel similarity)
    4. Trim both clips at the optimal point
    5. Apply short cross-dissolve (crossfade_frames) at each splice

    Cross-dissolve blends the last few frames of clip N with the first
    few frames of clip N+1, smoothing face/lighting jumps at the cut.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / "handles"
    work_dir.mkdir(parents=True, exist_ok=True)

    if len(clips) == 1:
        import shutil
        shutil.copy2(clips[0], output_path)
        return {"splices": [], "method": "single_clip"}

    logger.info(f"Assembling {len(clips)} clips with handle splice + "
                f"{crossfade_frames}-frame cross-dissolve...")

    splice_results = []
    trimmed_clips = []

    for i in range(len(clips)):
        clip = clips[i]
        trimmed = clip  # default: no trim

        if i < len(clips) - 1:
            tail_dir = work_dir / f"tail_{i}"
            head_dir = work_dir / f"head_{i+1}"

            logger.info(f"  Searching splice point {i}→{i+1} ({handle_frames} handle frames)...")

            tail_frames = extract_tail_frames(clip, tail_dir, handle_frames)
            head_frames = extract_head_frames(clips[i+1], head_dir, handle_frames)

            splice = find_best_splice(tail_frames, head_frames,
                                       tail_video=clip, head_video=clips[i+1])
            splice["cut_id"] = f"s{i:02d}_s{i+1:02d}"
            splice_results.append(splice)

            logger.info(f"    Best splice: tail_frame={splice['tail_cut_frame']}, "
                       f"head_frame={splice['head_cut_frame']}, "
                       f"score={splice['splice_score']}")

            # Trim this clip's tail
            if splice["tail_cut_frame"] < handle_frames - 1:
                dur = float(subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
                    capture_output=True, text=True, check=True).stdout.strip())
                total_frames = int(dur * 24)
                keep_frames = total_frames - (handle_frames - 1 - splice["tail_cut_frame"])
                trimmed = work_dir / f"trimmed_tail_{i}.mp4"
                trim_clip_tail(clip, trimmed, keep_frames)

            # Clean up extracted frames
            for fp in tail_frames:
                fp.unlink(missing_ok=True)
            tail_dir.rmdir()
            for fp in head_frames:
                fp.unlink(missing_ok=True)
            head_dir.rmdir()

        # Trim next clip's head if needed
        if i > 0 and splice_results:
            prev_splice = splice_results[-1] if i == len(clips) - 1 else (splice_results[-2] if len(splice_results) >= 2 else None)
            if prev_splice and prev_splice.get("head_cut_frame", 0) > 0:
                head_trimmed = work_dir / f"trimmed_head_{i}.mp4"
                trim_clip_head(trimmed, head_trimmed, prev_splice["head_cut_frame"])
                trimmed = head_trimmed

        trimmed_clips.append(trimmed)

    # Assemble with cross-dissolve between each pair
    if crossfade_frames > 0 and len(trimmed_clips) > 1:
        crossfade_dur = crossfade_frames / 24.0
        final = _assemble_with_crossfade(trimmed_clips, output_path, crossfade_dur, work_dir)
    else:
        # Fallback: hard concat
        list_file = work_dir / "concat.txt"
        with open(list_file, "w") as f:
            for c in trimmed_clips:
                f.write(f"file '{c}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
            "-c:a", "aac", "-b:a", "256k",
            str(output_path),
        ], check=True, capture_output=True)
        list_file.unlink(missing_ok=True)

    # Get final duration
    final_dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
        capture_output=True, text=True, check=True).stdout.strip())

    logger.info(f"  Assembled with handles + crossfade: {output_path} ({final_dur:.1f}s)")

    return {
        "splices": splice_results,
        "method": "handle_splice_crossfade",
        "handle_frames": handle_frames,
        "crossfade_frames": crossfade_frames,
        "final_duration": round(final_dur, 1),
        "mean_splice_score": round(np.mean([s["splice_score"] for s in splice_results]), 4) if splice_results else 0,
    }


def _assemble_with_crossfade(
    clips: list[Path],
    output_path: Path,
    crossfade_dur: float,
    work_dir: Path,
) -> Path:
    """Assemble clips with ffmpeg xfade filter for cross-dissolve transitions.

    Each transition is a short dissolve (crossfade_dur seconds) that blends
    the tail of clip N with the head of clip N+1, smoothing face/lighting jumps.
    Audio uses acrossfade for smooth audio transition.
    """
    if len(clips) < 2:
        return clips[0]

    # Get durations for offset calculation
    durations = []
    for c in clips:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(c)],
            capture_output=True, text=True, check=True).stdout.strip())
        durations.append(dur)

    # Build ffmpeg xfade filter chain
    # For N clips, we need N-1 xfade operations chained together
    n = len(clips)

    # Build input args
    inputs = []
    for c in clips:
        inputs.extend(["-i", str(c)])

    # Calculate offsets: each xfade starts at (cumulative_dur - crossfade_dur)
    # First xfade: offset = dur[0] - crossfade_dur
    # Second xfade: offset = dur[0] + dur[1] - 2*crossfade_dur
    # etc.
    offsets = []
    cumulative = 0
    for i in range(n - 1):
        cumulative += durations[i]
        offset = cumulative - (i + 1) * crossfade_dur
        offsets.append(max(0, offset))

    # Build video filter
    if n == 2:
        vfilter = f"[0:v][1:v]xfade=transition=fade:duration={crossfade_dur}:offset={offsets[0]}[outv]"
        afilter = f"[0:a][1:a]acrossfade=d={crossfade_dur}:c1=tri:c2=tri[outa]"
    else:
        # Chain xfades for 3+ clips
        vparts = []
        apart = []
        prev_v = "[0:v]"
        prev_a = "[0:a]"

        for i in range(n - 1):
            next_v = f"[{i+1}:v]"
            next_a = f"[{i+1}:a]"
            out_v = f"[v{i}]" if i < n - 2 else "[outv]"
            out_a = f"[a{i}]" if i < n - 2 else "[outa]"

            vparts.append(f"{prev_v}{next_v}xfade=transition=fade:duration={crossfade_dur}:offset={offsets[i]}{out_v}")
            apart.append(f"{prev_a}{next_a}acrossfade=d={crossfade_dur}:c1=tri:c2=tri{out_a}")

            prev_v = out_v
            prev_a = out_a

        vfilter = ";".join(vparts)
        afilter = ";".join(apart)

    filter_complex = f"{vfilter};{afilter}"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        str(output_path),
    ]

    logger.info(f"  Cross-fade assembly: {n} clips, {crossfade_dur}s dissolve")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
