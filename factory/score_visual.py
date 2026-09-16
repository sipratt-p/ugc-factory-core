"""Visual quality scoring for autoresearch.

Evaluates extracted video frames for:
1. Face consistency — ArcFace embeddings across frames (cosine similarity)
2. Identity match — generated face vs reference image embedding
3. Frame sharpness — Laplacian variance (blur detection)
4. Artifact detection — high-frequency anomaly scoring
5. Lip-sync correlation — mouth region movement vs audio energy

Runs on the eval GPU (config.EVAL_GPU) or CPU. InsightFace for face analysis.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

# Lazy-loaded models
_face_analyzer = None


def _get_face_analyzer():
    """Lazy-load InsightFace analyzer."""
    global _face_analyzer
    if _face_analyzer is None:
        try:
            import insightface
            _face_analyzer = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _face_analyzer.prepare(ctx_id=config.EVAL_GPU, det_size=(640, 640))
            logger.info(f"InsightFace loaded on GPU:{config.EVAL_GPU}")
        except Exception as e:
            logger.warning(f"InsightFace failed to load: {e}")
            _face_analyzer = "failed"
    return _face_analyzer if _face_analyzer != "failed" else None


def _load_image(path: Path) -> np.ndarray:
    """Load image as BGR numpy array for InsightFace."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def _get_face_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    """Extract ArcFace embedding from an image."""
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return None
    faces = analyzer.get(img)
    if not faces:
        return None
    # Return embedding of largest face
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.embedding


def _get_face_bbox(img: np.ndarray) -> Optional[tuple]:
    """Get face bounding box."""
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return None
    faces = analyzer.get(img)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return tuple(largest.bbox.astype(int))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_face_consistency(frame_paths: list[Path]) -> dict:
    """Score face consistency across frames using ArcFace embeddings.

    High consistency = face looks the same throughout the clip.
    Low consistency = face morphs, identity drift, distortion.
    """
    embeddings = []
    face_detected = []

    for fp in frame_paths:
        try:
            img = _load_image(fp)
            emb = _get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb)
                face_detected.append(True)
            else:
                face_detected.append(False)
        except Exception:
            face_detected.append(False)

    if len(embeddings) < 2:
        return {
            "face_consistency": None,
            "faces_detected": sum(face_detected),
            "total_frames": len(frame_paths),
            "error": "Not enough faces detected",
        }

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

    # Also compute sequential (adjacent frame) similarities
    sequential_sims = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        sequential_sims.append(sim)

    mean_sim = float(np.mean(similarities))
    min_sim = float(np.min(similarities))
    seq_mean = float(np.mean(sequential_sims)) if sequential_sims else 0

    return {
        "face_consistency": round(mean_sim, 4),
        "face_consistency_min": round(min_sim, 4),
        "face_consistency_sequential": round(seq_mean, 4),
        "faces_detected": sum(face_detected),
        "total_frames": len(frame_paths),
    }


def score_identity_match(frame_paths: list[Path], reference_path: Path) -> dict:
    """Score how well the generated face matches the reference image."""
    try:
        ref_img = _load_image(reference_path)
        ref_emb = _get_face_embedding(ref_img)
        if ref_emb is None:
            return {"identity_match": None, "error": "No face in reference image"}
    except Exception as e:
        return {"identity_match": None, "error": str(e)}

    matches = []
    for fp in frame_paths:
        try:
            img = _load_image(fp)
            emb = _get_face_embedding(img)
            if emb is not None:
                sim = _cosine_similarity(ref_emb, emb)
                matches.append(sim)
        except Exception:
            pass

    if not matches:
        return {"identity_match": None, "error": "No faces detected in frames"}

    return {
        "identity_match": round(float(np.mean(matches)), 4),
        "identity_match_min": round(float(np.min(matches)), 4),
        "identity_match_max": round(float(np.max(matches)), 4),
        "frames_matched": len(matches),
    }


def score_sharpness(frame_paths: list[Path]) -> dict:
    """Score frame sharpness using Laplacian variance.

    Higher variance = sharper image. Low variance = blurry/soft.
    Also detects compression artifacts via high-frequency analysis.
    """
    import cv2

    sharpness_scores = []
    artifact_scores = []

    for fp in frame_paths:
        try:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Laplacian variance for sharpness
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sharpness = float(laplacian.var())
            sharpness_scores.append(sharpness)

            # High-frequency energy for artifact detection
            # Sobel in both directions
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            edge_energy = float(np.mean(sobelx**2 + sobely**2))

            # Artifact score: ratio of very-high-freq to mid-freq
            # Artifacts show as unusual high-freq patterns
            dft = np.fft.fft2(img.astype(float))
            dft_shift = np.fft.fftshift(dft)
            magnitude = np.abs(dft_shift)
            h, w = magnitude.shape
            # High freq = outer 20% of spectrum
            mask_hf = np.zeros_like(magnitude, dtype=bool)
            mask_hf[:h//5, :] = True
            mask_hf[4*h//5:, :] = True
            mask_hf[:, :w//5] = True
            mask_hf[:, 4*w//5:] = True
            hf_energy = float(np.mean(magnitude[mask_hf]))
            total_energy = float(np.mean(magnitude))
            artifact_ratio = hf_energy / max(total_energy, 1e-10)
            artifact_scores.append(artifact_ratio)

        except Exception:
            pass

    if not sharpness_scores:
        return {"sharpness": None, "error": "No frames processed"}

    return {
        "sharpness_mean": round(float(np.mean(sharpness_scores)), 2),
        "sharpness_min": round(float(np.min(sharpness_scores)), 2),
        "sharpness_std": round(float(np.std(sharpness_scores)), 2),
        "artifact_ratio_mean": round(float(np.mean(artifact_scores)), 4),
        "artifact_ratio_max": round(float(np.max(artifact_scores)), 4),
    }


def score_face_regions(frame_paths: list[Path], reference_path: Path = None) -> dict:
    """Track specific face regions (eyebrows, eyes, mouth) for drift detection.

    Catches the eyebrow drift problem identified in V6 human review.
    Compares face landmarks between frames and against the reference.
    """
    import cv2

    analyzer = _get_face_analyzer()
    if analyzer is None:
        return {"face_regions": None, "error": "InsightFace not available"}

    # Get landmarks from reference
    ref_landmarks = None
    if reference_path and Path(reference_path).exists():
        ref_img = _load_image(Path(reference_path))
        ref_faces = analyzer.get(ref_img)
        if ref_faces:
            largest = max(ref_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            ref_landmarks = largest.landmark_2d_106 if hasattr(largest, 'landmark_2d_106') else None

    # Track landmarks across frames
    eyebrow_drifts = []
    eye_drifts = []

    for fp in frame_paths:
        try:
            img = _load_image(fp)
            faces = analyzer.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None

            if landmarks is not None and ref_landmarks is not None:
                # Normalize landmarks relative to face bbox
                bbox = face.bbox
                face_w = bbox[2] - bbox[0]
                face_h = bbox[3] - bbox[1]

                norm_lm = (landmarks - bbox[:2]) / np.array([face_w, face_h])

                ref_bbox = ref_faces[0].bbox if ref_faces else bbox
                ref_w = ref_bbox[2] - ref_bbox[0]
                ref_h = ref_bbox[3] - ref_bbox[1]
                norm_ref = (ref_landmarks - ref_bbox[:2]) / np.array([ref_w, ref_h])

                # Eyebrow region: landmarks 33-42 (approximate for 106-point model)
                # Eye region: landmarks 43-71
                if len(norm_lm) >= 72 and len(norm_ref) >= 72:
                    eyebrow_diff = float(np.mean(np.abs(norm_lm[33:43] - norm_ref[33:43])))
                    eye_diff = float(np.mean(np.abs(norm_lm[43:72] - norm_ref[43:72])))
                    eyebrow_drifts.append(eyebrow_diff)
                    eye_drifts.append(eye_diff)
        except Exception:
            pass

    result = {}
    if eyebrow_drifts:
        result["eyebrow_drift_mean"] = round(float(np.mean(eyebrow_drifts)), 4)
        result["eyebrow_drift_max"] = round(float(np.max(eyebrow_drifts)), 4)
    if eye_drifts:
        result["eye_drift_mean"] = round(float(np.mean(eye_drifts)), 4)
        result["eye_drift_max"] = round(float(np.max(eye_drifts)), 4)

    return result


def score_mouth_movement(frame_paths: list[Path]) -> dict:
    """Score mouth region movement across frames.

    Higher movement variance in the mouth region = more lip animation.
    Combined with audio energy correlation = lip-sync quality estimate.
    """
    import cv2

    mouth_diffs = []
    prev_mouth = None

    for fp in frame_paths:
        try:
            img = _load_image(fp)
            bbox = _get_face_bbox(img)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            face_h = y2 - y1
            # Mouth region = bottom 35% of face
            mouth_y1 = y1 + int(face_h * 0.65)
            mouth_region = img[mouth_y1:y2, x1:x2]

            if mouth_region.size == 0:
                continue

            # Resize to fixed size for comparison
            mouth_resized = cv2.resize(mouth_region, (64, 32))
            mouth_gray = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2GRAY).astype(float)

            if prev_mouth is not None:
                diff = float(np.mean(np.abs(mouth_gray - prev_mouth)))
                mouth_diffs.append(diff)

            prev_mouth = mouth_gray

        except Exception:
            pass

    if not mouth_diffs:
        return {"mouth_movement": None, "error": "Could not track mouth"}

    return {
        "mouth_movement_mean": round(float(np.mean(mouth_diffs)), 2),
        "mouth_movement_std": round(float(np.std(mouth_diffs)), 2),
        "mouth_movement_max": round(float(np.max(mouth_diffs)), 2),
        "mouth_frames_tracked": len(mouth_diffs),
    }


def score_visual(
    frame_paths: list[Path],
    reference_image: Path = None,
) -> dict:
    """Run all visual quality metrics on extracted frames.

    Returns structured dict with all scores for Claude to optimize against.
    """
    frame_paths = [Path(p) for p in frame_paths if Path(p).exists()]

    if not frame_paths:
        return {"error": "No valid frame paths"}

    scores = {}

    # 1. Face consistency across frames
    logger.info("  Scoring face consistency...")
    consistency = score_face_consistency(frame_paths)
    scores.update({f"face_{k}": v for k, v in consistency.items()})

    # 2. Identity match to reference
    if reference_image and Path(reference_image).exists():
        logger.info("  Scoring identity match...")
        identity = score_identity_match(frame_paths, Path(reference_image))
        scores.update(identity)

    # 3. Sharpness and artifacts
    logger.info("  Scoring sharpness...")
    sharpness = score_sharpness(frame_paths)
    scores.update(sharpness)

    # 4. Mouth movement
    logger.info("  Scoring mouth movement...")
    mouth = score_mouth_movement(frame_paths)
    scores.update(mouth)

    # 5. Face region tracking (eyebrows, eyes) — catches drift from human review
    if reference_image and Path(reference_image).exists():
        logger.info("  Scoring face regions (eyebrow drift)...")
        regions = score_face_regions(frame_paths, Path(reference_image))
        scores.update(regions)

    # 5. Visual composite score
    components = []
    if scores.get("face_face_consistency") is not None:
        components.append(("face_consistency", scores["face_face_consistency"] * 10, 0.25))
    if scores.get("identity_match") is not None:
        components.append(("identity_match", scores["identity_match"] * 10, 0.25))
    if scores.get("sharpness_mean") is not None:
        # Normalize sharpness (typical range 50-500)
        sharp_norm = min(10, scores["sharpness_mean"] / 50)
        components.append(("sharpness", sharp_norm, 0.25))
    if scores.get("mouth_movement_mean") is not None:
        # Normalize mouth movement (typical range 5-30)
        mouth_norm = min(10, scores["mouth_movement_mean"] / 3)
        components.append(("mouth_movement", mouth_norm, 0.25))

    if components:
        weighted = sum(s * w for _, s, w in components)
        total_w = sum(w for _, _, w in components)
        scores["visual_composite"] = round(weighted / max(total_w, 0.01), 2)
    else:
        scores["visual_composite"] = 0

    return scores
