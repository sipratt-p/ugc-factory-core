"""SyncNet-based lip sync scoring for UGC video factory.

Three levels of lip sync evaluation:

  Level 1 (fast, <1s):  score_mouth_movement() — InsightFace 106-point lip aperture
  Level 2 (quality, ~3-5s): score_syncnet_fast() — SyncNet LSE-D/LSE-C, InsightFace crops
  Level 3 (comprehensive): score_sync_full() — Level 1 + Level 2 + per-frame breakdown

Key innovation: bypasses the slow S3FD face detector used by syncnet-python's default
pipeline. Instead, uses the InsightFace buffalo_l model that's already loaded in our
scoring pipeline to detect the face once, then tracks the bbox across frames via simple
smoothing. This cuts ~8s of S3FD detection down to near-zero overhead.

SyncNet v2 architecture expects:
  - Video: 5 consecutive 224x224 face crops, shape (B, 3, 5, 224, 224)
  - Audio: MFCC features, shape (B, 1, 13, 20) per 5-frame window
  - Output: 1024-dim embedding per modality; pairwise distance = sync quality

LSE-D (lip sync error - distance): lower = better sync (good < 7.0, bad > 9.0)
LSE-C (lip sync error - confidence): higher = better sync (good > 5.0, bad < 2.0)
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

from . import config

logger = logging.getLogger(__name__)

# CUDA device for SyncNet/InsightFace (env UGC_SCORE_DEVICE, default the eval GPU)
SCORE_DEVICE = os.environ.get("UGC_SCORE_DEVICE", f"cuda:{config.EVAL_GPU}")

# ---------------------------------------------------------------------------
# Shared state: lazy-loaded models
# ---------------------------------------------------------------------------
_syncnet_model = None
_face_analyzer = None

# Default paths for SyncNet v2 weights (env UGC_SYNCNET_WEIGHTS wins)
_SYNCNET_WEIGHTS_PATHS = [
    Path(p) for p in [os.environ.get("UGC_SYNCNET_WEIGHTS", "")] if p
] + [
    config.MODELS_DIR / "syncnet_v2.model",
]


def _get_face_analyzer():
    """Reuse the InsightFace analyzer from score_visual (singleton).

    Falls back to creating a new one if score_visual hasn't been imported yet.
    """
    global _face_analyzer
    if _face_analyzer is not None:
        return _face_analyzer if _face_analyzer != "failed" else None

    # Try to reuse the analyzer from score_visual first
    try:
        from .score_visual import _get_face_analyzer as _get_shared_analyzer
        analyzer = _get_shared_analyzer()
        if analyzer is not None:
            _face_analyzer = analyzer
            return _face_analyzer
    except Exception:
        pass

    # Create our own instance
    try:
        import insightface
        _face_analyzer = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_analyzer.prepare(ctx_id=config.EVAL_GPU, det_size=(640, 640))
        logger.info(f"score_sync: InsightFace loaded on GPU:{config.EVAL_GPU}")
    except Exception as e:
        logger.warning(f"score_sync: InsightFace failed to load: {e}")
        _face_analyzer = "failed"

    return _face_analyzer if _face_analyzer != "failed" else None


def _get_syncnet_model(device: str = SCORE_DEVICE) -> Optional[torch.nn.Module]:
    """Lazy-load SyncNet v2 model weights.

    Tries local weight paths first, then falls back to HuggingFace download.
    """
    global _syncnet_model
    if _syncnet_model is not None:
        return _syncnet_model if _syncnet_model != "failed" else None

    # Find weights
    weights_path = None
    for p in _SYNCNET_WEIGHTS_PATHS:
        if p.exists():
            weights_path = p
            break

    if weights_path is None:
        # Try HuggingFace download
        try:
            from huggingface_hub import hf_hub_download
            weights_path = Path(hf_hub_download(
                repo_id="lithiumice/syncnet",
                filename="syncnet_v2.model",
                cache_dir=str(config.MODELS_DIR / ".cache"),
            ))
            logger.info(f"score_sync: Downloaded SyncNet weights to {weights_path}")
        except Exception as e:
            logger.error(f"score_sync: Cannot find or download SyncNet weights: {e}")
            _syncnet_model = "failed"
            return None

    try:
        from syncnet_python.SyncNetModel import S
        from syncnet_python.SyncNetInstance import SyncNetInstance

        net = S()
        state_dict = torch.load(str(weights_path), map_location=device, weights_only=False)

        # Handle both full model and state_dict formats
        if isinstance(state_dict, dict) and any(k.startswith("netcnn") or k.startswith("netfc") for k in state_dict):
            net.load_state_dict(state_dict)
        elif isinstance(state_dict, torch.nn.Module):
            net = state_dict
        else:
            net.load_state_dict(state_dict)

        net.eval()
        instance = SyncNetInstance(net=net, device=device)
        _syncnet_model = instance
        logger.info(f"score_sync: SyncNet v2 loaded from {weights_path} on {device}")
        return _syncnet_model
    except Exception as e:
        logger.error(f"score_sync: Failed to load SyncNet model: {e}")
        _syncnet_model = "failed"
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MouthMovementResult:
    """Level 1: mouth aperture analysis from InsightFace 106-point landmarks."""
    mouth_open_std: float = 0.0
    mouth_open_max: float = 0.0
    mouth_open_mean: float = 0.0
    is_static: bool = True
    face_height: float = 0.0
    frames_analyzed: int = 0
    apertures: list = field(default_factory=list)  # per-frame lip aperture
    error: Optional[str] = None


@dataclass
class SyncNetResult:
    """Level 2: SyncNet LSE-D/LSE-C scores."""
    lse_d: float = float("inf")      # lower = better (distance)
    lse_c: float = 0.0               # higher = better (confidence)
    av_offset: int = 0               # frames of audio-video offset
    confidence_per_frame: list = field(default_factory=list)
    quality: str = "UNKNOWN"          # GOOD / FAIR / POOR / NO_FACE / ERROR
    error: Optional[str] = None


@dataclass
class SyncScore:
    """Level 3: comprehensive lip sync score combining all metrics."""
    # Level 1
    mouth_open_std: float = 0.0
    mouth_open_max: float = 0.0
    mouth_open_mean: float = 0.0
    is_static: bool = True

    # Level 2
    lse_d: float = float("inf")
    lse_c: float = 0.0
    av_offset: int = 0
    quality: str = "UNKNOWN"

    # Per-frame analysis
    confidence_per_frame: list = field(default_factory=list)
    apertures: list = field(default_factory=list)

    # Breakdown: where does sync fail?
    sync_break_frames: list = field(default_factory=list)  # frame indices where sync drops
    worst_sync_region: tuple = (0, 0)  # (start_frame, end_frame) of worst region

    # Metadata
    frames_analyzed: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio_wav(video_path: Path, output_path: Path, sr: int = 16000) -> Path:
    """Extract mono WAV audio from video at the given sample rate."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path


def _get_video_fps(video_path: Path) -> float:
    """Get video frame rate via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def _read_video_frames(video_path: Path, max_frames: int = 0) -> tuple[list[np.ndarray], float]:
    """Read all frames from video. Returns (frames_bgr, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def _detect_face_insightface(frame: np.ndarray) -> Optional[dict]:
    """Detect the largest face in a frame using InsightFace.

    Returns dict with bbox, landmark_2d_106, and face_height, or None.
    """
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return None

    faces = analyzer.get(frame)
    if not faces:
        return None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = face.bbox.astype(int)
    lm106 = face.landmark_2d_106 if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None else None

    return {
        "bbox": bbox,  # [x1, y1, x2, y2]
        "landmark_2d_106": lm106,
        "face_height": float(bbox[3] - bbox[1]),
    }


def _lip_aperture_from_106(landmarks: np.ndarray) -> float:
    """Compute inner lip aperture (vertical opening) from 106-point landmarks.

    InsightFace 106-point landmark layout for the mouth region:
      - Outer lip:  84-95 (12 points around outer lip contour)
      - Inner lip:  96-103 (8 points around inner lip contour)
        96 = inner lip top-left
        97 = inner lip top-center-left
        98 = inner lip top-center
        99 = inner lip top-center-right
        100 = inner lip top-right
        101 = inner lip bottom-right
        102 = inner lip bottom-center
        103 = inner lip bottom-left

    We measure the vertical distance between the top inner lip midpoint
    and bottom inner lip midpoint. This directly captures mouth opening.
    """
    if landmarks is None or len(landmarks) < 104:
        return 0.0

    # Inner lip top: average of points 97, 98, 99 (top center region)
    top_inner = np.mean(landmarks[97:100], axis=0)

    # Inner lip bottom: point 102 (bottom center)
    bottom_inner = landmarks[102]

    # Vertical aperture
    aperture = float(np.linalg.norm(top_inner - bottom_inner))
    return aperture


def _smooth_bboxes(bboxes: list[np.ndarray], kernel_size: int = 13) -> list[np.ndarray]:
    """Smooth bounding boxes over time using median filter (same as syncnet_pipeline)."""
    if len(bboxes) < kernel_size:
        return bboxes

    arr = np.array(bboxes, dtype=float)
    smoothed = np.zeros_like(arr)
    for col in range(4):
        smoothed[:, col] = signal.medfilt(arr[:, col], kernel_size=min(kernel_size, len(arr) | 1))
    return [smoothed[i] for i in range(len(smoothed))]


def _crop_face_for_syncnet(frame: np.ndarray, bbox: np.ndarray, crop_scale: float = 0.40) -> np.ndarray:
    """Crop and resize face region to 224x224 for SyncNet, matching syncnet_pipeline._crop.

    The crop_scale adds padding around the face to include jaw/neck context
    that SyncNet expects (it was trained with this framing).
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h = y2 - y1
    w = x2 - x1
    s = max(h, w) / 2.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Pad to include context (matching syncnet_pipeline's crop logic)
    pad = int(s * (1 + 2 * crop_scale))

    # Apply border padding to avoid going out of bounds
    frame_h, frame_w = frame.shape[:2]
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110))

    # Shift center coordinates by padding amount
    cx_p = cx + pad
    cy_p = cy + pad

    # Crop window: from syncnet_pipeline logic
    crop_y1 = int(cy_p - s)
    crop_y2 = int(cy_p + s * (1 + 2 * crop_scale))
    crop_x1 = int(cx_p - s * (1 + crop_scale))
    crop_x2 = int(cx_p + s * (1 + crop_scale))

    crop = padded[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        # Fallback: simple center crop
        crop = padded[
            max(0, int(cy_p - s * 1.5)):int(cy_p + s * 1.5),
            max(0, int(cx_p - s * 1.5)):int(cx_p + s * 1.5),
        ]

    if crop.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    return cv2.resize(crop, (224, 224))


def _compute_mfcc(audio_path: Path) -> np.ndarray:
    """Compute MFCC features from WAV file, matching SyncNet's expected format.

    SyncNet expects: 13 MFCC coefficients, with 4 audio frames per video frame
    at 25fps (640 samples per video frame at 16kHz).
    """
    import python_speech_features

    sample_rate, audio = wavfile.read(str(audio_path))

    # python_speech_features.mfcc returns (num_frames, num_cep)
    # zip(*mfcc) transposes it to (num_cep, num_frames) = (13, T)
    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc])  # shape: (13, T)

    return mfcc


# ---------------------------------------------------------------------------
# Level 1: Fast mouth movement scoring
# ---------------------------------------------------------------------------

def score_mouth_movement(clip_path: Path, frame_skip: int = 1) -> MouthMovementResult:
    """Level 1: Fast lip aperture analysis using InsightFace 106-point landmarks.

    Reads video frames, detects the face once on frame 0 to establish the bbox,
    then uses landmark_2d_106 inner lip points to compute actual lip aperture
    per frame.

    Args:
        clip_path: Path to video file.
        frame_skip: Analyze every N-th frame (1 = all frames, 2 = every other).

    Returns:
        MouthMovementResult with aperture statistics and static detection.
    """
    clip_path = Path(clip_path)
    if not clip_path.exists():
        return MouthMovementResult(error=f"File not found: {clip_path}")

    try:
        frames, fps = _read_video_frames(clip_path)
        if len(frames) < 5:
            return MouthMovementResult(error=f"Too few frames: {len(frames)}")

        analyzer = _get_face_analyzer()
        if analyzer is None:
            return MouthMovementResult(error="InsightFace not available")

        apertures = []
        face_height = 0.0

        for i in range(0, len(frames), frame_skip):
            frame = frames[i]
            faces = analyzer.get(frame)
            if not faces:
                # Carry forward last aperture if we had one
                if apertures:
                    apertures.append(apertures[-1])
                continue

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            fh = float(face.bbox[3] - face.bbox[1])
            if fh > face_height:
                face_height = fh

            lm106 = face.landmark_2d_106 if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None else None
            ap = _lip_aperture_from_106(lm106)

            # Normalize by face height for scale invariance
            if face_height > 0:
                ap_norm = ap / face_height
            else:
                ap_norm = ap

            apertures.append(ap_norm)

        if not apertures:
            return MouthMovementResult(error="No faces detected in any frame")

        arr = np.array(apertures)
        std = float(np.std(arr))
        static_threshold = 0.02  # relative to face height (already normalized)

        return MouthMovementResult(
            mouth_open_std=round(std, 5),
            mouth_open_max=round(float(np.max(arr)), 5),
            mouth_open_mean=round(float(np.mean(arr)), 5),
            is_static=std < static_threshold,
            face_height=round(face_height, 1),
            frames_analyzed=len(apertures),
            apertures=[round(float(a), 5) for a in apertures],
        )

    except Exception as e:
        logger.error(f"score_mouth_movement failed: {e}", exc_info=True)
        return MouthMovementResult(error=str(e))


# ---------------------------------------------------------------------------
# Level 2: SyncNet LSE-D / LSE-C scoring (InsightFace crops, no S3FD)
# ---------------------------------------------------------------------------

def score_syncnet_fast(
    clip_path: Path,
    batch_size: int = 20,
    vshift: int = 15,
    device: str = SCORE_DEVICE,
) -> SyncNetResult:
    """Level 2: SyncNet LSE-D / LSE-C, bypassing S3FD with InsightFace crops.

    Pipeline:
      1. Read all video frames.
      2. Detect face on first frame with InsightFace (one detection).
      3. Track bbox across all frames via InsightFace detections on keyframes +
         interpolation (much faster than per-frame S3FD).
      4. Crop each frame to 224x224 face region (matching SyncNet's training).
      5. Extract 16kHz mono WAV and compute MFCC.
      6. Feed 5-frame video windows + corresponding MFCC windows into SyncNet.
      7. Compute LSE-D, LSE-C, and per-frame confidence.

    Args:
        clip_path: Path to video file.
        batch_size: Number of 5-frame windows to process per GPU batch.
        vshift: Temporal search range for offset detection (frames).
        device: CUDA device for SyncNet inference.

    Returns:
        SyncNetResult with lse_d, lse_c, av_offset, and per-frame confidence.
    """
    clip_path = Path(clip_path)
    if not clip_path.exists():
        return SyncNetResult(error=f"File not found: {clip_path}", quality="ERROR")

    work_dir = None
    try:
        # Load SyncNet model
        syncnet = _get_syncnet_model(device=device)
        if syncnet is None:
            return SyncNetResult(error="SyncNet model not available", quality="ERROR")

        # Read video frames
        frames, fps = _read_video_frames(clip_path)
        n_frames = len(frames)
        if n_frames < 10:
            return SyncNetResult(error=f"Too few frames: {n_frames}", quality="ERROR")

        # ---- Face detection + tracking via InsightFace ----
        analyzer = _get_face_analyzer()
        if analyzer is None:
            return SyncNetResult(error="InsightFace not available", quality="NO_FACE")

        # Detect on keyframes (every 10 frames + first + last)
        keyframe_indices = sorted(set(
            [0, n_frames - 1] + list(range(0, n_frames, 10))
        ))
        keyframe_bboxes = {}

        for ki in keyframe_indices:
            faces = analyzer.get(frames[ki])
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                keyframe_bboxes[ki] = face.bbox.copy()

        if not keyframe_bboxes:
            return SyncNetResult(error="No face detected on any keyframe", quality="NO_FACE")

        # Interpolate bbox for all frames
        kf_idx = sorted(keyframe_bboxes.keys())
        kf_bboxes = np.array([keyframe_bboxes[k] for k in kf_idx])

        all_bboxes = []
        for i in range(n_frames):
            if i in keyframe_bboxes:
                all_bboxes.append(keyframe_bboxes[i].copy())
            else:
                # Find surrounding keyframes and interpolate
                left_idx = max(k for k in kf_idx if k <= i) if any(k <= i for k in kf_idx) else kf_idx[0]
                right_idx = min(k for k in kf_idx if k >= i) if any(k >= i for k in kf_idx) else kf_idx[-1]

                if left_idx == right_idx:
                    all_bboxes.append(keyframe_bboxes[left_idx].copy())
                else:
                    t = (i - left_idx) / (right_idx - left_idx)
                    interp = keyframe_bboxes[left_idx] * (1 - t) + keyframe_bboxes[right_idx] * t
                    all_bboxes.append(interp)

        # Smooth bboxes (matching syncnet_pipeline's median filter approach)
        all_bboxes = _smooth_bboxes(all_bboxes, kernel_size=13)

        # ---- Crop face regions to 224x224 ----
        face_crops = []
        for i, frame in enumerate(frames):
            crop = _crop_face_for_syncnet(frame, np.array(all_bboxes[i]))
            face_crops.append(crop)

        # ---- Build video tensor for SyncNet ----
        # SyncNet expects: (B, 3, 5, 224, 224) — 5-frame windows
        im = np.stack(face_crops, axis=3)       # (H, W, C, T) = (224, 224, 3, N)
        im = np.expand_dims(im, axis=0)          # (1, H, W, C, T)
        im = np.transpose(im, (0, 3, 4, 1, 2))  # (1, T, C, H, W) — match SyncNetInstance
        imtv = torch.from_numpy(im.astype(np.float32))

        # ---- Extract audio and compute MFCC ----
        work_dir = Path(tempfile.mkdtemp(prefix="syncnet_"))
        audio_path = work_dir / "audio.wav"
        _extract_audio_wav(clip_path, audio_path, sr=16000)

        sample_rate, audio_data = wavfile.read(str(audio_path))
        import python_speech_features
        mfcc = zip(*python_speech_features.mfcc(audio_data, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])   # (13, T_audio)

        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)  # (1, 1, 13, T_audio)
        cct = torch.from_numpy(cc.astype(np.float32))

        # ---- Length alignment ----
        min_length = min(n_frames, math.floor(len(audio_data) / 640))
        lastframe = min_length - 5
        if lastframe < 1:
            return SyncNetResult(error="Clip too short for SyncNet (need >5 frames of aligned AV)", quality="ERROR")

        # ---- SyncNet forward pass ----
        # Access the inner SyncNet model (S). It's stored as an nn.Module
        # submodule with key '__S__' — direct attribute access is blocked by
        # Python name-mangling, so we go through _modules.
        net = syncnet._modules.get("__S__")
        if net is None:
            return SyncNetResult(error="Cannot access SyncNet inner model", quality="ERROR")
        net.to(device)
        net.eval()

        im_feat = []
        cc_feat = []

        with torch.no_grad():
            for i in range(0, lastframe, batch_size):
                end = min(lastframe, i + batch_size)

                # Video: 5-frame windows
                im_batch = [imtv[:, :, vf:vf+5, :, :] for vf in range(i, end)]
                im_in = torch.cat(im_batch, 0).to(device)
                im_out = net.forward_lip(im_in)
                im_feat.append(im_out.data.cpu())

                # Audio: 20-MFCC-frame windows (4 per video frame, 5 frames = 20)
                cc_batch = [cct[:, :, :, vf*4:vf*4+20] for vf in range(i, end)]

                # Validate audio window sizes (skip truncated windows)
                valid_cc = [c for c in cc_batch if c.shape[3] == 20]
                if len(valid_cc) < len(cc_batch):
                    # Trim video features to match
                    im_feat[-1] = im_feat[-1][:len(valid_cc)]
                    if not valid_cc:
                        continue

                cc_in = torch.cat(valid_cc, 0).to(device)
                cc_out = net.forward_aud(cc_in)
                cc_feat.append(cc_out.data.cpu())

        if not im_feat or not cc_feat:
            return SyncNetResult(error="No features extracted (clip may be too short)", quality="ERROR")

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # Trim to same length if needed
        min_feat = min(len(im_feat), len(cc_feat))
        im_feat = im_feat[:min_feat]
        cc_feat = cc_feat[:min_feat]

        # ---- Compute offset, distance, confidence (matching SyncNetInstance.evaluate) ----
        win_size = vshift * 2 + 1
        cc_feat_padded = torch.nn.functional.pad(cc_feat, (0, 0, vshift, vshift))

        dists = []
        for i in range(len(im_feat)):
            dists.append(
                torch.nn.functional.pairwise_distance(
                    im_feat[[i], :].repeat(win_size, 1),
                    cc_feat_padded[i:i + win_size, :],
                )
            )

        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)
        offset = int(vshift - minidx)
        conf = float(torch.median(mdist) - minval)
        lse_d = float(minval)
        lse_c = conf

        # Per-frame confidence
        fdist = np.stack([dist[minidx].numpy() for dist in dists])
        fconf = float(torch.median(mdist).numpy()) - fdist
        fconfm = signal.medfilt(fconf, kernel_size=min(9, len(fconf) | 1))

        # Quality assessment
        if lse_c > 5.0 and lse_d < 7.0:
            quality = "GOOD"
        elif lse_c > 3.0 and lse_d < 8.0:
            quality = "FAIR"
        elif lse_c > 1.5 and lse_d < 10.0:
            quality = "MEDIOCRE"
        else:
            quality = "POOR"

        return SyncNetResult(
            lse_d=round(lse_d, 3),
            lse_c=round(lse_c, 3),
            av_offset=offset,
            confidence_per_frame=[round(float(c), 3) for c in fconfm],
            quality=quality,
        )

    except Exception as e:
        logger.error(f"score_syncnet_fast failed: {e}", exc_info=True)
        return SyncNetResult(error=str(e), quality="ERROR")
    finally:
        if work_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Level 3: Comprehensive sync scoring
# ---------------------------------------------------------------------------

def score_sync_full(
    clip_path: Path,
    frame_skip: int = 1,
    batch_size: int = 20,
    vshift: int = 15,
    device: str = SCORE_DEVICE,
) -> SyncScore:
    """Level 3: Full lip sync analysis combining mouth movement + SyncNet.

    Runs Level 1 (mouth aperture) and Level 2 (SyncNet LSE-D/LSE-C), then
    analyzes per-frame confidence to identify WHERE sync breaks down.

    Returns a SyncScore dataclass with all metrics and diagnostic info.
    """
    clip_path = Path(clip_path)

    # --- Level 1: mouth movement ---
    logger.info(f"score_sync_full: Level 1 — mouth movement analysis ({clip_path.name})")
    mouth = score_mouth_movement(clip_path, frame_skip=frame_skip)

    # --- Level 2: SyncNet ---
    logger.info(f"score_sync_full: Level 2 — SyncNet LSE-D/LSE-C ({clip_path.name})")
    syncnet_result = score_syncnet_fast(
        clip_path,
        batch_size=batch_size,
        vshift=vshift,
        device=device,
    )

    # --- Level 3: per-frame breakdown + diagnostics ---
    sync_break_frames = []
    worst_start, worst_end = 0, 0

    if syncnet_result.confidence_per_frame:
        conf_arr = np.array(syncnet_result.confidence_per_frame)
        median_conf = float(np.median(conf_arr))

        # Frames where confidence drops below 0 (worse than median distance)
        for i, c in enumerate(conf_arr):
            if c < 0:
                sync_break_frames.append(i)

        # Find the worst contiguous region of low confidence
        if sync_break_frames:
            # Find longest run of consecutive bad frames
            runs = []
            run_start = sync_break_frames[0]
            prev = sync_break_frames[0]
            for idx in sync_break_frames[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    runs.append((run_start, prev))
                    run_start = idx
                    prev = idx
            runs.append((run_start, prev))

            # Worst = longest run
            worst_run = max(runs, key=lambda r: r[1] - r[0])
            worst_start, worst_end = worst_run

    # Combine errors
    errors = []
    if mouth.error:
        errors.append(f"L1: {mouth.error}")
    if syncnet_result.error:
        errors.append(f"L2: {syncnet_result.error}")

    return SyncScore(
        # Level 1
        mouth_open_std=mouth.mouth_open_std,
        mouth_open_max=mouth.mouth_open_max,
        mouth_open_mean=mouth.mouth_open_mean,
        is_static=mouth.is_static,

        # Level 2
        lse_d=syncnet_result.lse_d,
        lse_c=syncnet_result.lse_c,
        av_offset=syncnet_result.av_offset,
        quality=syncnet_result.quality,

        # Per-frame
        confidence_per_frame=syncnet_result.confidence_per_frame,
        apertures=mouth.apertures,

        # Diagnostics
        sync_break_frames=sync_break_frames,
        worst_sync_region=(worst_start, worst_end),

        # Metadata
        frames_analyzed=mouth.frames_analyzed,
        error="; ".join(errors) if errors else None,
    )


# ---------------------------------------------------------------------------
# Convenience: dict output for integration with quality gate
# ---------------------------------------------------------------------------

def score_sync_dict(clip_path: Path, level: int = 2) -> dict:
    """Return sync scores as a flat dict (for integration with existing scoring pipeline).

    Args:
        clip_path: Path to video file.
        level: 1 = mouth movement only, 2 = SyncNet, 3 = full.

    Returns:
        Dict compatible with quality_gate and other factory scoring modules.
    """
    if level == 1:
        result = score_mouth_movement(clip_path)
        return {
            "mouth_open_std": result.mouth_open_std,
            "mouth_open_max": result.mouth_open_max,
            "mouth_open_mean": result.mouth_open_mean,
            "is_static": result.is_static,
            "sync_error": result.error,
        }
    elif level == 2:
        result = score_syncnet_fast(clip_path)
        return {
            "lse_d": result.lse_d,
            "lse_c": result.lse_c,
            "av_offset": result.av_offset,
            "sync_quality": result.quality,
            "sync_error": result.error,
        }
    else:
        result = score_sync_full(clip_path)
        return {
            "mouth_open_std": result.mouth_open_std,
            "mouth_open_max": result.mouth_open_max,
            "mouth_open_mean": result.mouth_open_mean,
            "is_static": result.is_static,
            "lse_d": result.lse_d,
            "lse_c": result.lse_c,
            "av_offset": result.av_offset,
            "sync_quality": result.quality,
            "sync_break_count": len(result.sync_break_frames),
            "worst_sync_region": result.worst_sync_region,
            "frames_analyzed": result.frames_analyzed,
            "sync_error": result.error,
        }


# ---------------------------------------------------------------------------
# CLI entrypoint for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SyncNet lip sync scorer")
    parser.add_argument("clip", type=str, help="Path to video clip")
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3],
                        help="Scoring level: 1=mouth, 2=syncnet, 3=full")
    parser.add_argument("--device", type=str, default=SCORE_DEVICE,
                        help="CUDA device for SyncNet inference")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    clip = Path(args.clip)
    if not clip.exists():
        print(f"Error: {clip} not found", file=sys.stderr)
        sys.exit(1)

    if args.level == 1:
        result = score_mouth_movement(clip)
        if args.json:
            print(json.dumps(score_sync_dict(clip, level=1), indent=2))
        else:
            print(f"Mouth Movement (Level 1):")
            print(f"  std:     {result.mouth_open_std:.5f}")
            print(f"  max:     {result.mouth_open_max:.5f}")
            print(f"  mean:    {result.mouth_open_mean:.5f}")
            print(f"  static:  {result.is_static}")
            print(f"  frames:  {result.frames_analyzed}")
            if result.error:
                print(f"  error:   {result.error}")

    elif args.level == 2:
        result = score_syncnet_fast(clip, device=args.device)
        if args.json:
            print(json.dumps(score_sync_dict(clip, level=2), indent=2))
        else:
            print(f"SyncNet (Level 2):")
            print(f"  LSE-D:      {result.lse_d:.3f}  (lower = better, good < 7.0)")
            print(f"  LSE-C:      {result.lse_c:.3f}  (higher = better, good > 5.0)")
            print(f"  AV offset:  {result.av_offset} frames")
            print(f"  Quality:    {result.quality}")
            if result.error:
                print(f"  error:      {result.error}")

    else:
        result = score_sync_full(clip, device=args.device)
        if args.json:
            print(json.dumps(score_sync_dict(clip, level=3), indent=2))
        else:
            print(f"Full Sync (Level 3):")
            print(f"  --- Mouth Movement ---")
            print(f"  std:            {result.mouth_open_std:.5f}")
            print(f"  max:            {result.mouth_open_max:.5f}")
            print(f"  mean:           {result.mouth_open_mean:.5f}")
            print(f"  static:         {result.is_static}")
            print(f"  --- SyncNet ---")
            print(f"  LSE-D:          {result.lse_d:.3f}")
            print(f"  LSE-C:          {result.lse_c:.3f}")
            print(f"  AV offset:      {result.av_offset} frames")
            print(f"  Quality:        {result.quality}")
            print(f"  --- Diagnostics ---")
            print(f"  sync breaks:    {len(result.sync_break_frames)} frames")
            print(f"  worst region:   frames {result.worst_sync_region[0]}-{result.worst_sync_region[1]}")
            print(f"  frames scored:  {result.frames_analyzed}")
            if result.error:
                print(f"  error:          {result.error}")
