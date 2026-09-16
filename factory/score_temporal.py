"""Temporal quality scoring — detects intra-clip degradation.

From V6 human review: quality drops from beginning to end of a clip,
but our scoring averages over the whole clip, hiding the degradation.

This module compares early vs late quality within a single clip:
- Visual: face consistency early→late, sharpness early→late
- Audio: naturalness early→late, speech clarity early→late
- Returns a degradation score (0 = no degradation, 1 = severe)
"""

import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_temporal_frames(video_path: Path, output_dir: Path) -> dict:
    """Extract frames from early, middle, and late portions of a clip.

    Returns dict with paths to early frames (first 20%), mid frames, and late frames (last 20%).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True).stdout.strip())

    frames = {"early": [], "mid": [], "late": []}

    # Early: 3 frames from first 20%
    for i, t in enumerate([dur * 0.05, dur * 0.10, dur * 0.15]):
        path = output_dir / f"early_{i}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t),
            "-i", str(video_path), "-frames:v", "1", "-q:v", "1", str(path),
        ], check=True, capture_output=True)
        frames["early"].append(path)

    # Mid: 3 frames from middle 20%
    for i, t in enumerate([dur * 0.40, dur * 0.50, dur * 0.60]):
        path = output_dir / f"mid_{i}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t),
            "-i", str(video_path), "-frames:v", "1", "-q:v", "1", str(path),
        ], check=True, capture_output=True)
        frames["mid"].append(path)

    # Late: 3 frames from last 20%
    for i, t in enumerate([dur * 0.80, dur * 0.87, dur * 0.95]):
        path = output_dir / f"late_{i}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(t),
            "-i", str(video_path), "-frames:v", "1", "-q:v", "1", str(path),
        ], check=True, capture_output=True)
        frames["late"].append(path)

    return frames


def score_visual_degradation(video_path: Path, reference_path: Path = None) -> dict:
    """Measure visual quality degradation from start to end of clip."""
    from .score_visual import (
        _load_image, _get_face_embedding, _cosine_similarity,
    )
    import cv2

    temp_dir = video_path.parent / "temporal_eval"
    temporal_frames = extract_temporal_frames(video_path, temp_dir)

    results = {}

    # Face embedding comparison: early vs late vs reference
    early_embs = []
    late_embs = []
    ref_emb = None

    if reference_path and Path(reference_path).exists():
        try:
            ref_img = _load_image(Path(reference_path))
            ref_emb = _get_face_embedding(ref_img)
        except Exception:
            pass

    for fp in temporal_frames["early"]:
        try:
            emb = _get_face_embedding(_load_image(fp))
            if emb is not None:
                early_embs.append(emb)
        except Exception:
            pass

    for fp in temporal_frames["late"]:
        try:
            emb = _get_face_embedding(_load_image(fp))
            if emb is not None:
                late_embs.append(emb)
        except Exception:
            pass

    if early_embs and late_embs:
        # Cross-temporal face consistency
        cross_sims = []
        for e in early_embs:
            for l in late_embs:
                cross_sims.append(_cosine_similarity(e, l))
        results["face_early_vs_late"] = round(float(np.mean(cross_sims)), 4)

        # Within-early consistency
        if len(early_embs) >= 2:
            early_sims = [_cosine_similarity(early_embs[i], early_embs[j])
                         for i in range(len(early_embs)) for j in range(i+1, len(early_embs))]
            results["face_early_consistency"] = round(float(np.mean(early_sims)), 4)

        # Within-late consistency
        if len(late_embs) >= 2:
            late_sims = [_cosine_similarity(late_embs[i], late_embs[j])
                        for i in range(len(late_embs)) for j in range(i+1, len(late_embs))]
            results["face_late_consistency"] = round(float(np.mean(late_sims)), 4)

        # Identity match degradation
        if ref_emb is not None:
            early_id = float(np.mean([_cosine_similarity(ref_emb, e) for e in early_embs]))
            late_id = float(np.mean([_cosine_similarity(ref_emb, l) for l in late_embs]))
            results["identity_early"] = round(early_id, 4)
            results["identity_late"] = round(late_id, 4)
            results["identity_degradation"] = round(early_id - late_id, 4)

    # Sharpness comparison: early vs late
    early_sharpness = []
    late_sharpness = []
    for fp in temporal_frames["early"]:
        try:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lap = cv2.Laplacian(img, cv2.CV_64F)
                early_sharpness.append(float(lap.var()))
        except Exception:
            pass
    for fp in temporal_frames["late"]:
        try:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lap = cv2.Laplacian(img, cv2.CV_64F)
                late_sharpness.append(float(lap.var()))
        except Exception:
            pass

    if early_sharpness and late_sharpness:
        results["sharpness_early"] = round(float(np.mean(early_sharpness)), 2)
        results["sharpness_late"] = round(float(np.mean(late_sharpness)), 2)
        results["sharpness_degradation"] = round(
            float(np.mean(early_sharpness)) - float(np.mean(late_sharpness)), 2)

    # Cleanup
    for frames in temporal_frames.values():
        for fp in frames:
            fp.unlink(missing_ok=True)
    temp_dir.rmdir()

    # Overall visual degradation score (0 = no degradation, 1 = severe)
    degradation_signals = []
    if "identity_degradation" in results:
        degradation_signals.append(max(0, results["identity_degradation"]) * 5)
    if "sharpness_degradation" in results and results.get("sharpness_early", 0) > 0:
        relative_sharp_drop = results["sharpness_degradation"] / max(results["sharpness_early"], 1)
        degradation_signals.append(max(0, relative_sharp_drop))
    if "face_early_vs_late" in results:
        degradation_signals.append(max(0, 1.0 - results["face_early_vs_late"]))

    results["visual_degradation_score"] = round(
        float(np.mean(degradation_signals)) if degradation_signals else 0, 4)

    return results


def score_audio_degradation(video_path: Path) -> dict:
    """Measure audio quality degradation from start to end of clip."""
    import soundfile as sf

    # Get duration
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True).stdout.strip())

    results = {}

    # Extract early audio (first 30%) and late audio (last 30%)
    temp_dir = video_path.parent / "temporal_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)

    early_audio = temp_dir / "early.wav"
    late_audio = temp_dir / "late.wav"

    early_dur = dur * 0.3
    late_start = dur * 0.7

    subprocess.run([
        "ffmpeg", "-y", "-t", str(early_dur),
        "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", str(early_audio),
    ], check=True, capture_output=True)

    subprocess.run([
        "ffmpeg", "-y", "-ss", str(late_start),
        "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", str(late_audio),
    ], check=True, capture_output=True)

    try:
        early_data, sr = sf.read(str(early_audio))
        late_data, _ = sf.read(str(late_audio))

        # RMS energy comparison
        early_rms = float(np.sqrt(np.mean(early_data**2)))
        late_rms = float(np.sqrt(np.mean(late_data**2)))
        results["audio_rms_early"] = round(early_rms, 4)
        results["audio_rms_late"] = round(late_rms, 4)

        # Speech ratio comparison
        from scipy.fft import fft
        def speech_ratio(data, sr):
            spectrum = np.abs(fft(data[:sr*2]))[:sr]
            freqs_per_bin = sr / len(spectrum)
            s_start = int(100 / freqs_per_bin)
            s_end = int(4000 / freqs_per_bin)
            speech_e = np.sum(spectrum[s_start:s_end]**2)
            total_e = np.sum(spectrum**2)
            return float(speech_e / max(total_e, 1e-10))

        early_speech = speech_ratio(early_data, sr)
        late_speech = speech_ratio(late_data, sr)
        results["speech_ratio_early"] = round(early_speech, 4)
        results["speech_ratio_late"] = round(late_speech, 4)

        # UTMOS on early vs late
        try:
            import torch
            predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            with torch.no_grad():
                early_mos = predictor(torch.from_numpy(early_data).float().unsqueeze(0), sr).item()
                late_mos = predictor(torch.from_numpy(late_data).float().unsqueeze(0), sr).item()
            results["utmos_early"] = round(early_mos, 3)
            results["utmos_late"] = round(late_mos, 3)
            results["utmos_degradation"] = round(early_mos - late_mos, 3)
        except Exception as e:
            results["utmos_error"] = str(e)

        # Overall audio degradation
        degradation = 0
        if "utmos_degradation" in results:
            degradation = max(0, results["utmos_degradation"]) / 2  # normalize
        results["audio_degradation_score"] = round(min(1.0, degradation), 4)

    except Exception as e:
        results["error"] = str(e)

    # Cleanup
    early_audio.unlink(missing_ok=True)
    late_audio.unlink(missing_ok=True)
    temp_dir.rmdir()

    return results


def score_temporal(video_path: Path, reference_path: Path = None) -> dict:
    """Full temporal degradation scoring — visual + audio.

    Returns combined degradation score. High degradation = quality drops
    significantly from start to end of clip.
    """
    results = {}

    logger.info("  Scoring visual degradation (early vs late frames)...")
    vis_deg = score_visual_degradation(video_path, reference_path)
    results["visual_temporal"] = vis_deg

    logger.info("  Scoring audio degradation (early vs late audio)...")
    aud_deg = score_audio_degradation(video_path)
    results["audio_temporal"] = aud_deg

    # Combined temporal degradation
    v_deg = vis_deg.get("visual_degradation_score", 0)
    a_deg = aud_deg.get("audio_degradation_score", 0)
    results["temporal_degradation"] = round((v_deg * 0.6 + a_deg * 0.4), 4)

    return results
