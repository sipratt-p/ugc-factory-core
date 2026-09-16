"""Advanced scoring metrics — DINOv2 drift, RAFT optical flow, CLAPScore.

These complement the existing scoring stack with:
- DINOv2: semantic drift beyond face (clothing, background, lighting changes)
- RAFT: motion discontinuity at seams (optical flow jump at cuts)
- CLAPScore: audio-text alignment (does the audio match the prompt intent?)
- LPIPS temporal: frame-to-frame perceptual flicker within a clip

All use models already in torchvision or pip-installed packages.
Run on the eval GPU (config.EVAL_GPU).
"""

import logging
import subprocess
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-loaded models
_dino_model = None
_dino_transform = None
_raft_model = None
_raft_transforms = None
_clap_model = None


# === DINOv2 Semantic Drift ===

def _get_dino():
    """Lazy-load DINOv2 ViT-S/14 (smallest, ~22M params, ~200MB VRAM).

    Uses torchvision's bundled DINOv2 to avoid xformers dependency issue
    on Blackwell GPUs (sm_120).
    """
    global _dino_model, _dino_transform
    if _dino_model is None:
        import torch
        import torchvision.transforms as T
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        # Use torchvision's ViT with DINO-pretrained weights as fallback
        # since facebookresearch/dinov2 requires xformers on CUDA
        try:
            # Try DINOv2 with SDPA forced
            import os
            os.environ["XFORMERS_DISABLED"] = "1"
            _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
            _dino_model = _dino_model.cuda().eval()
        except Exception:
            # Fallback: use torchvision ViT-B/16 (CLIP-pretrained, still good for drift)
            logger.info("DINOv2 failed (xformers), using torchvision ViT-B/16 for drift scoring")
            weights = ViT_B_16_Weights.DEFAULT
            _dino_model = vit_b_16(weights=weights)
            _dino_model = _dino_model.cuda().eval()

        _dino_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _dino_model, _dino_transform


def score_dino_drift(frame_paths: list[Path]) -> dict:
    """Score semantic drift across frames using DINOv2 embeddings.

    Catches changes beyond face identity: background shifts, clothing changes,
    lighting drift, object disappearance. Complements ArcFace which only
    tracks face identity.

    Returns:
      - dino_consistency: mean pairwise cosine similarity (higher = more consistent)
      - dino_drift: 1 - similarity between first and last frame
      - dino_flicker: std of consecutive-frame similarities (higher = more flickery)
    """
    try:
        import torch
        from PIL import Image

        model, transform = _get_dino()
        embeddings = []

        for fp in frame_paths:
            if not Path(fp).exists():
                continue
            img = Image.open(fp).convert('RGB')
            tensor = transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                emb = model(tensor)
                # Handle different output shapes
                if emb.dim() == 3:
                    emb = emb[:, 0]  # CLS token for ViT
                emb = emb.squeeze()
            embeddings.append(emb.cpu())

        if len(embeddings) < 2:
            return {"dino_consistency": None}

        stack = torch.stack(embeddings)
        stack = stack / stack.norm(dim=1, keepdim=True)

        # Consecutive similarities
        consec_sims = []
        for i in range(len(stack) - 1):
            sim = float(torch.dot(stack[i], stack[i + 1]))
            consec_sims.append(sim)

        # First vs last
        first_last_sim = float(torch.dot(stack[0], stack[-1]))

        return {
            "dino_consistency": round(float(np.mean(consec_sims)), 4),
            "dino_drift": round(1.0 - first_last_sim, 4),
            "dino_flicker": round(float(np.std(consec_sims)), 4),
        }

    except Exception as e:
        logger.warning(f"DINOv2 drift scoring failed: {e}")
        return {"dino_consistency": None, "dino_error": str(e)}


# === RAFT Optical Flow at Seams ===

def _get_raft():
    """Lazy-load RAFT optical flow model from torchvision."""
    global _raft_model, _raft_transforms
    if _raft_model is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        weights = Raft_Small_Weights.DEFAULT
        _raft_transforms = weights.transforms()
        _raft_model = raft_small(weights=weights).cuda().eval()
    return _raft_model, _raft_transforms


def score_flow_discontinuity(
    clip_before: Path,
    clip_after: Path,
) -> dict:
    """Score motion discontinuity at a seam using optical flow.

    Compares the flow pattern at the cut point:
    - flow_before: between last 2 frames of clip N (normal intra-clip motion)
    - flow_across: between last frame of clip N and first frame of clip N+1
    - discontinuity = difference between these two flows

    High discontinuity = jarring motion jump at the cut.
    """
    try:
        import torch
        import cv2

        # Get duration of clip_before
        dur = float(subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip_before)],
            capture_output=True, text=True, check=True).stdout.strip())

        work_dir = clip_before.parent / "_flow_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Extract: second-to-last frame of clip A, last frame of clip A, first frame of clip B
        frame_a1 = work_dir / "a_penultimate.png"
        frame_a2 = work_dir / "a_last.png"
        frame_b1 = work_dir / "b_first.png"

        subprocess.run(["ffmpeg", "-y", "-ss", str(max(0, dur - 0.083)),
                        "-i", str(clip_before), "-frames:v", "1", "-q:v", "1",
                        str(frame_a1)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-sseof", "-0.042",
                        "-i", str(clip_before), "-frames:v", "1", "-q:v", "1",
                        str(frame_a2)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-ss", "0",
                        "-i", str(clip_after), "-frames:v", "1", "-q:v", "1",
                        str(frame_b1)], check=True, capture_output=True)

        def _load_for_raft(path):
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (520, 960))  # RAFT needs specific sizes
            return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        model, transforms = _get_raft()

        img_a1 = _load_for_raft(frame_a1).cuda()
        img_a2 = _load_for_raft(frame_a2).cuda()
        img_b1 = _load_for_raft(frame_b1).cuda()

        with torch.no_grad():
            # Flow within clip A (normal motion)
            flow_intra = model(img_a1, img_a2)[-1]
            # Flow across the cut
            flow_cross = model(img_a2, img_b1)[-1]

        # Discontinuity = difference in flow patterns
        flow_diff = torch.abs(flow_cross - flow_intra)
        discontinuity = float(flow_diff.mean().item())
        max_discontinuity = float(flow_diff.max().item())

        # Also measure raw cross-cut flow magnitude (large = big jump)
        cross_magnitude = float(torch.sqrt((flow_cross ** 2).sum(dim=1)).mean().item())

        # Cleanup
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)

        return {
            "flow_discontinuity": round(discontinuity, 4),
            "flow_max_discontinuity": round(max_discontinuity, 4),
            "flow_cross_magnitude": round(cross_magnitude, 4),
        }

    except Exception as e:
        logger.warning(f"Optical flow scoring failed: {e}")
        return {"flow_discontinuity": None, "flow_error": str(e)}


# === CLAPScore — Audio-Text Alignment ===

def _get_clap():
    """Lazy-load CLAP model for audio-text alignment."""
    global _clap_model
    if _clap_model is None:
        try:
            import laion_clap
            _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            _clap_model.load_ckpt()  # downloads default checkpoint
        except Exception as e:
            logger.warning(f"CLAP model loading failed: {e}")
            _clap_model = "failed"
    return _clap_model if _clap_model != "failed" else None


def score_clap(audio_path: Path, text_prompt: str) -> dict:
    """Score audio-text alignment using CLAP embeddings.

    Measures how well the audio matches the intended prompt/description.
    Catches: wrong voice character, unexpected sounds, music when speech expected.

    Returns:
      - clap_score: cosine similarity between audio and text embeddings (0-1)
    """
    try:
        model = _get_clap()
        if model is None:
            return {"clap_score": None, "error": "CLAP not loaded"}

        # Ensure audio is WAV
        wav_path = audio_path
        if not str(audio_path).endswith('.wav'):
            wav_path = audio_path.parent / "_clap_audio.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(audio_path),
                "-vn", "-ar", "48000", "-ac", "1", str(wav_path),
            ], check=True, capture_output=True)

        # Get embeddings
        audio_emb = model.get_audio_embedding_from_filelist([str(wav_path)])
        text_emb = model.get_text_embedding([text_prompt])

        # Cosine similarity
        sim = float(np.dot(audio_emb[0], text_emb[0]) /
                     (np.linalg.norm(audio_emb[0]) * np.linalg.norm(text_emb[0]) + 1e-8))

        if wav_path != audio_path:
            wav_path.unlink(missing_ok=True)

        return {"clap_score": round(sim, 4)}

    except Exception as e:
        logger.warning(f"CLAP scoring failed: {e}")
        return {"clap_score": None, "clap_error": str(e)}


# === LPIPS Temporal Flicker ===

def score_lpips_temporal(frame_paths: list[Path]) -> dict:
    """Score temporal flicker using LPIPS between consecutive frames.

    High mean LPIPS = frames are very different (lots of change).
    High std LPIPS = inconsistent rate of change (flickery).

    Returns:
      - lpips_temporal_mean: average perceptual distance between consecutive frames
      - lpips_temporal_std: flicker indicator (high = jittery)
      - lpips_temporal_max: worst single frame-to-frame jump
    """
    try:
        import lpips
        import torch
        import cv2

        fn = lpips.LPIPS(net='alex', verbose=False)

        def _to_tensor(fp):
            img = cv2.imread(str(fp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            return t.unsqueeze(0) * 2 - 1

        dists = []
        valid_paths = [p for p in frame_paths if Path(p).exists()]

        for i in range(len(valid_paths) - 1):
            t1 = _to_tensor(valid_paths[i])
            t2 = _to_tensor(valid_paths[i + 1])
            d = float(fn(t1, t2).item())
            dists.append(d)

        if not dists:
            return {"lpips_temporal_mean": None}

        return {
            "lpips_temporal_mean": round(float(np.mean(dists)), 4),
            "lpips_temporal_std": round(float(np.std(dists)), 4),
            "lpips_temporal_max": round(float(np.max(dists)), 4),
        }

    except Exception as e:
        logger.warning(f"LPIPS temporal scoring failed: {e}")
        return {"lpips_temporal_mean": None, "lpips_error": str(e)}
