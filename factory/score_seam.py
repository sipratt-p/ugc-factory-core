"""Seam, within-clip, and lip sync scoring for multi-clip sequences.

Two concerns:
1. SEAM quality — what happens at every hard cut between clips
2. WITHIN-CLIP consistency — does quality degrade within a single clip

Seam metrics (between clip N and clip N+1):
  - Face embedding delta (ArcFace: last frame N vs first frame N+1)
  - NISQA quality delta (overall MOS difference between clips)
  - Loudness delta (RMS energy at the boundary)

Within-clip metrics:
  - Identity match: first frame vs last frame (ArcFace cosine)
  - Sharpness slope: Laplacian variance first frames vs last frames
"""

import logging
import subprocess
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_frame_at(video_path: Path, time_s: float, output_path: Path) -> Path:
    """Extract a single frame at the given time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(time_s),
        "-i", str(video_path), "-frames:v", "1", "-q:v", "1", str(output_path),
    ], check=True, capture_output=True)
    return output_path


def _get_duration(video_path: Path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True).stdout.strip())


def _get_rms(video_path: Path, start_s: float, duration_s: float = 0.25) -> float:
    """Get RMS energy of audio at a specific time window."""
    import soundfile as sf

    tmp = video_path.parent / "_seam_audio.wav"
    # Use output seeking (-ss after -i) for reliability on short clips
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", str(max(0, start_s)),
        "-t", str(duration_s),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", str(tmp),
    ], check=True, capture_output=True)

    data, _ = sf.read(str(tmp))
    tmp.unlink(missing_ok=True)

    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(data ** 2)))


def _laplacian_sharpness(frame_path: Path) -> float:
    """Compute Laplacian variance (sharpness) of a frame."""
    import cv2
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def score_seam(
    clip_before: Path,
    clip_after: Path,
    nisqa_before: dict = None,
    nisqa_after: dict = None,
) -> dict:
    """Score the quality of a hard cut between two clips.

    Returns seam metrics and a composite seam_score (0-1, higher = better seam).
    """
    from .score_visual import _load_image, _get_face_embedding, _cosine_similarity

    result = {}
    work_dir = clip_before.parent / "_seam_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        dur_before = _get_duration(clip_before)
        dur_after = _get_duration(clip_after)

        # --- Face embedding delta across cut ---
        last_frame = work_dir / "last_frame.png"
        first_frame = work_dir / "first_frame.png"

        _extract_frame_at(clip_before, max(0, dur_before - 0.042), last_frame)
        _extract_frame_at(clip_after, 0.0, first_frame)

        img_before = _load_image(last_frame)
        img_after = _load_image(first_frame)
        emb_before = _get_face_embedding(img_before)
        emb_after = _get_face_embedding(img_after)

        if emb_before is not None and emb_after is not None:
            face_sim = _cosine_similarity(emb_before, emb_after)
            result["face_delta"] = round(1.0 - face_sim, 4)
        else:
            result["face_delta"] = None

        # --- LPIPS perceptual distance at boundary ---
        try:
            import lpips
            import torch
            import cv2

            _lpips_fn = lpips.LPIPS(net='alex', verbose=False)

            def _img_to_tensor(img_bgr):
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (256, 256))
                t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
                return t.unsqueeze(0) * 2 - 1  # normalize to [-1, 1]

            t_before = _img_to_tensor(img_before)
            t_after = _img_to_tensor(img_after)
            lpips_dist = float(_lpips_fn(t_before, t_after).item())
            result["lpips_delta"] = round(lpips_dist, 4)
        except Exception as e:
            logger.warning(f"LPIPS scoring failed: {e}")
            result["lpips_delta"] = None

        # --- Head pose delta via InsightFace landmarks ---
        try:
            from .score_visual import _get_face_analyzer

            def _get_pose_vector(img_bgr):
                """Extract head pose from InsightFace landmarks (already loaded)."""
                analyzer = _get_face_analyzer()
                if analyzer is None:
                    return None
                faces = analyzer.get(img_bgr)
                if not faces:
                    return None
                face = faces[0]
                # Use 5 key landmarks: left eye, right eye, nose, left mouth, right mouth
                if hasattr(face, 'kps') and face.kps is not None:
                    return face.kps.flatten()
                return None

            pose_before = _get_pose_vector(img_before)
            pose_after = _get_pose_vector(img_after)
            if pose_before is not None and pose_after is not None:
                pose_delta = float(np.linalg.norm(pose_before - pose_after))
                # Normalize by image dimensions (landmarks are in pixel space)
                h = img_before.shape[0]
                result["head_pose_delta"] = round(pose_delta / h, 4)
            else:
                result["head_pose_delta"] = None
        except Exception as e:
            logger.warning(f"Head pose scoring failed: {e}")
            result["head_pose_delta"] = None

        last_frame.unlink(missing_ok=True)
        first_frame.unlink(missing_ok=True)

        # --- Loudness delta at boundary ---
        rms_end = _get_rms(clip_before, max(0, dur_before - 0.25), 0.25)
        rms_start = _get_rms(clip_after, 0.0, 0.25)
        result["loudness_end"] = round(rms_end, 4)
        result["loudness_start"] = round(rms_start, 4)
        result["loudness_delta"] = round(abs(rms_end - rms_start), 4)
        result["lands_in_pause"] = bool(rms_end < 0.02 or rms_start < 0.02)

        # --- NISQA quality delta between clips ---
        if nisqa_before and nisqa_after:
            mos_before = nisqa_before.get("nisqa_mos") or 0
            mos_after = nisqa_after.get("nisqa_mos") or 0
            result["nisqa_delta"] = round(abs(mos_before - mos_after), 3)

            col_before = nisqa_before.get("nisqa_coloration") or 0
            col_after = nisqa_after.get("nisqa_coloration") or 0
            result["coloration_delta"] = round(abs(col_before - col_after), 3)
        else:
            result["nisqa_delta"] = None
            result["coloration_delta"] = None

        # --- Composite seam score (0-1, higher = better) ---
        score = 1.0

        # Face identity jump (30% weight)
        face_d = result.get("face_delta")
        if face_d is not None:
            score -= min(face_d * 1.5, 0.30)

        # LPIPS perceptual distance (25% weight) — highest signal for visual jarring
        lpips_d = result.get("lpips_delta")
        if lpips_d is not None:
            # LPIPS 0.0 = identical, 0.5+ = very different
            score -= min(lpips_d * 0.5, 0.25)

        # Head pose snap (15% weight)
        pose_d = result.get("head_pose_delta")
        if pose_d is not None:
            # pose_delta 0.0 = same pose, 0.1+ = noticeable snap
            score -= min(pose_d * 1.5, 0.15)

        # Audio landing (15% weight)
        if not result.get("lands_in_pause", False):
            score -= 0.10
        if result.get("loudness_delta", 0) > 0.05:
            score -= min(result["loudness_delta"] * 1.0, 0.05)

        # Audio quality consistency (15% weight)
        if result.get("nisqa_delta") is not None and result["nisqa_delta"] > 0.8:
            score -= min(result["nisqa_delta"] * 0.1, 0.15)

        result["seam_score"] = round(max(0, min(1, score)), 3)

    except Exception as e:
        logger.warning(f"Seam scoring failed: {e}")
        result["seam_score"] = 0.5
        result["seam_error"] = str(e)

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    return result


def score_within_clip(video_path: Path) -> dict:
    """Score consistency within a single clip.

    Compares first frames vs last frames to detect:
    - Identity drift (face changes over the clip)
    - Sharpness degradation (gets blurry/melty toward the end)
    """
    from .score_visual import _load_image, _get_face_embedding, _cosine_similarity

    result = {}
    work_dir = video_path.parent / "_within_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        dur = _get_duration(video_path)

        # Extract frames at 10%, 25%, 75%, 90% of duration
        time_early_1 = dur * 0.10
        time_early_2 = dur * 0.25
        time_late_1 = dur * 0.75
        time_late_2 = dur * 0.90

        frames = {}
        for label, t in [("e1", time_early_1), ("e2", time_early_2),
                         ("l1", time_late_1), ("l2", time_late_2)]:
            fp = work_dir / f"{label}.png"
            _extract_frame_at(video_path, t, fp)
            frames[label] = fp

        # Face identity: early vs late
        emb_early = []
        emb_late = []
        for label in ["e1", "e2"]:
            try:
                img = _load_image(frames[label])
                emb = _get_face_embedding(img)
                if emb is not None:
                    emb_early.append(emb)
            except Exception:
                pass
        for label in ["l1", "l2"]:
            try:
                img = _load_image(frames[label])
                emb = _get_face_embedding(img)
                if emb is not None:
                    emb_late.append(emb)
            except Exception:
                pass

        if emb_early and emb_late:
            # Average similarity between early and late embeddings
            sims = []
            for e in emb_early:
                for l in emb_late:
                    sims.append(_cosine_similarity(e, l))
            identity_sim = float(np.mean(sims))
            result["identity_early_late"] = round(identity_sim, 4)
            result["identity_drop"] = round(1.0 - identity_sim, 4)
        else:
            result["identity_early_late"] = None
            result["identity_drop"] = 0.0

        # Sharpness: early vs late
        sharp_early = []
        sharp_late = []
        for label in ["e1", "e2"]:
            s = _laplacian_sharpness(frames[label])
            if s > 0:
                sharp_early.append(s)
        for label in ["l1", "l2"]:
            s = _laplacian_sharpness(frames[label])
            if s > 0:
                sharp_late.append(s)

        if sharp_early and sharp_late:
            mean_early = float(np.mean(sharp_early))
            mean_late = float(np.mean(sharp_late))
            result["sharpness_early"] = round(mean_early, 1)
            result["sharpness_late"] = round(mean_late, 1)
            if mean_early > 0:
                drop_pct = (mean_early - mean_late) / mean_early * 100
                result["sharpness_drop_pct"] = round(drop_pct, 1)
            else:
                result["sharpness_drop_pct"] = 0.0
        else:
            result["sharpness_drop_pct"] = 0.0

    except Exception as e:
        logger.warning(f"Within-clip scoring failed: {e}")
        result["identity_drop"] = 0.0
        result["sharpness_drop_pct"] = 0.0
        result["within_clip_error"] = str(e)

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    return result


def score_lip_sync(video_path: Path) -> dict:
    """Score lip sync quality using MediaPipe landmark-audio correlation.

    Measures correlation between mouth aperture (from facial landmarks)
    and audio energy envelope. High correlation = good lip sync.
    No GPU required — runs entirely on CPU via MediaPipe.

    Returns:
      - lip_sync_corr: Pearson correlation at zero lag (-1 to 1, higher = better)
      - lip_sync_peak_corr: max cross-correlation at any offset
      - lip_sync_offset_ms: temporal offset at peak (0 = perfect sync)
      - mouth_movement_range: std of mouth aperture (higher = more expressive)
    """
    try:
        import cv2
        import soundfile as sf
        from scipy.signal import correlate

        audio_path = video_path.parent / "_lipsync_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "16000", "-ac", "1", str(audio_path),
        ], check=True, capture_output=True)

        audio_data, sr = sf.read(str(audio_path))
        audio_path.unlink(missing_ok=True)

        dur = _get_duration(video_path)
        fps = 24.0
        n_frames = int(dur * fps)

        # Per-frame audio RMS envelope
        samples_per_frame = int(sr / fps)
        audio_envelope = []
        for i in range(n_frames):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(audio_data))
            if start < len(audio_data):
                audio_envelope.append(float(np.sqrt(np.mean(audio_data[start:end] ** 2))))
            else:
                audio_envelope.append(0.0)

        # Mouth aperture per frame via InsightFace 106-point landmarks
        from .score_visual import _get_face_analyzer
        analyzer = _get_face_analyzer()

        cap = cv2.VideoCapture(str(video_path))
        mouth_apertures = []
        frame_skip = 2  # process every 2nd frame for speed, interpolate

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0 and analyzer is not None:
                faces = analyzer.get(frame)
                if faces:
                    kps = faces[0].kps  # 5-point: left_eye, right_eye, nose, left_mouth, right_mouth
                    if kps is not None and len(kps) >= 5:
                        # Mouth aperture from 5-point landmarks:
                        # vertical: distance from nose to midpoint of mouth corners
                        # horizontal: distance between mouth corners
                        left_mouth = kps[3]
                        right_mouth = kps[4]
                        nose = kps[2]
                        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2
                        vert = abs(nose[1] - mouth_mid_y)
                        horiz = abs(left_mouth[0] - right_mouth[0])
                        mouth_apertures.append(vert / max(horiz, 1e-6))
                    else:
                        mouth_apertures.append(mouth_apertures[-1] if mouth_apertures else 0.0)
                else:
                    mouth_apertures.append(mouth_apertures[-1] if mouth_apertures else 0.0)
            else:
                # Interpolate skipped frames
                mouth_apertures.append(mouth_apertures[-1] if mouth_apertures else 0.0)
            frame_idx += 1

        cap.release()

        min_len = min(len(audio_envelope), len(mouth_apertures))
        if min_len < 10:
            return {"lip_sync_corr": None, "error": "too few frames"}

        audio_env = np.array(audio_envelope[:min_len])
        mouth_env = np.array(mouth_apertures[:min_len])

        # Normalize
        audio_env = (audio_env - audio_env.mean()) / max(audio_env.std(), 1e-8)
        mouth_env = (mouth_env - mouth_env.mean()) / max(mouth_env.std(), 1e-8)

        # Cross-correlation
        xcorr = correlate(mouth_env, audio_env, mode='full')
        xcorr = xcorr / max(np.abs(xcorr).max(), 1e-8)
        lags = np.arange(-min_len + 1, min_len)

        peak_idx = int(np.argmax(xcorr))
        peak_corr = float(xcorr[peak_idx])
        offset_frames = int(lags[peak_idx])

        zero_lag_corr = float(np.corrcoef(audio_env, mouth_env)[0, 1])
        mouth_range = float(np.std(mouth_apertures[:min_len]))

        return {
            "lip_sync_corr": round(zero_lag_corr, 4),
            "lip_sync_peak_corr": round(peak_corr, 4),
            "lip_sync_offset_frames": offset_frames,
            "lip_sync_offset_ms": round(offset_frames / fps * 1000, 1),
            "mouth_movement_range": round(mouth_range, 4),
        }

    except Exception as e:
        logger.warning(f"Lip sync scoring failed: {e}")
        return {"lip_sync_corr": None, "lip_sync_error": str(e)}
