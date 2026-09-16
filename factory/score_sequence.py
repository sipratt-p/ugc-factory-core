"""Sequence-level scoring for multi-clip videos.

From ChatGPT5 Pro research: score at three levels:
1. Shot score — per-clip quality (audio + visual)
2. Cut score — boundary quality (pause landing, loudness delta, pose jump)
3. Sequence score — global consistency (face drift, voice drift, pace match)

Final score = 55% shot + 25% cut + 20% sequence, with hard penalties.
"""

import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

from .score_av import score_clip, extract_audio_wav, score_nisqa
from .score_visual import (
    score_face_consistency, score_identity_match, score_sharpness,
    score_mouth_movement, _load_image, _get_face_embedding, _cosine_similarity,
)
from .score_seam import score_seam, score_within_clip, score_lip_sync
from .score_advanced import score_dino_drift, score_flow_discontinuity, score_lpips_temporal

logger = logging.getLogger(__name__)


def score_cut(
    clip_before: Path,
    clip_after: Path,
    cut_time_s: float = None,
) -> dict:
    """Score the quality of a hard cut between two clips.

    Measures:
    - Does the cut land in a pause/silence?
    - Loudness jump across the cut
    - Pose/face jump across the cut
    """
    scores = {}

    try:
        # Extract last frame of clip_before and first frame of clip_after
        last_frame = clip_before.parent / "cut_last.png"
        first_frame = clip_after.parent / "cut_first.png"

        subprocess.run([
            "ffmpeg", "-y", "-sseof", "-0.042",
            "-i", str(clip_before), "-frames:v", "1", "-q:v", "1", str(last_frame),
        ], check=True, capture_output=True)

        subprocess.run([
            "ffmpeg", "-y", "-ss", "0",
            "-i", str(clip_after), "-frames:v", "1", "-q:v", "1", str(first_frame),
        ], check=True, capture_output=True)

        # Face similarity across cut
        img_before = _load_image(last_frame)
        img_after = _load_image(first_frame)
        emb_before = _get_face_embedding(img_before)
        emb_after = _get_face_embedding(img_after)

        if emb_before is not None and emb_after is not None:
            face_jump = 1.0 - _cosine_similarity(emb_before, emb_after)
            scores["face_jump"] = round(float(face_jump), 4)
        else:
            scores["face_jump"] = None

        # Cleanup
        last_frame.unlink(missing_ok=True)
        first_frame.unlink(missing_ok=True)

        # Loudness analysis at the cut point
        import soundfile as sf

        # Extract last 0.5s audio from clip_before
        audio_end = clip_before.parent / "cut_audio_end.wav"
        subprocess.run([
            "ffmpeg", "-y", "-sseof", "-0.5",
            "-i", str(clip_before), "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", str(audio_end),
        ], check=True, capture_output=True)

        # Extract first 0.5s audio from clip_after
        audio_start = clip_after.parent / "cut_audio_start.wav"
        subprocess.run([
            "ffmpeg", "-y", "-t", "0.5",
            "-i", str(clip_after), "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", str(audio_start),
        ], check=True, capture_output=True)

        data_end, _ = sf.read(str(audio_end))
        data_start, _ = sf.read(str(audio_start))

        rms_end = float(np.sqrt(np.mean(data_end[-1600:]**2)))  # last 100ms
        rms_start = float(np.sqrt(np.mean(data_start[:1600]**2)))  # first 100ms

        # Loudness jump (lower is better — means cut lands in quiet moment)
        loudness_delta = abs(rms_end - rms_start)
        scores["loudness_delta"] = round(loudness_delta, 4)

        # Does the cut land in silence? (both sides low energy)
        lands_in_pause = rms_end < 0.02 or rms_start < 0.02
        scores["lands_in_pause"] = bool(lands_in_pause)

        # Cleanup
        audio_end.unlink(missing_ok=True)
        audio_start.unlink(missing_ok=True)

        # Cut score (0-1, higher is better)
        cut_quality = 1.0
        if scores.get("face_jump") is not None:
            cut_quality -= scores["face_jump"] * 2  # penalize face jump
        if not lands_in_pause:
            cut_quality -= 0.2  # penalize cuts not in pauses
        if loudness_delta > 0.05:
            cut_quality -= loudness_delta * 2  # penalize loudness jumps

        scores["cut_score"] = round(max(0, min(1, cut_quality)), 3)

    except Exception as e:
        logger.warning(f"Cut scoring failed: {e}")
        scores["cut_error"] = str(e)
        scores["cut_score"] = 0.5

    return scores


def score_sequence(
    clips: list[Path],
    reference_image: Path = None,
    target_script: str = "",
    reference_audio: Path = None,
) -> dict:
    """Score a complete multi-clip sequence at all three levels.

    Returns shot_scores, cut_scores, and sequence_scores.
    Final score = 50% shot + 25% seam + 25% sequence (V9 rebalanced).
    """
    result = {
        "shot_scores": [],
        "cut_scores": [],
        "sequence_scores": {},
        "final_score": 0,
    }

    if not clips:
        return result

    # === SHOT SCORES ===
    all_face_embeddings = []
    all_nisqa = []  # per-clip NISQA for seam scoring

    for i, clip in enumerate(clips):
        logger.info(f"  Scoring shot {i+1}/{len(clips)}...")

        # Audio scoring per clip (includes NISQA)
        clip_speech = target_script.split(".")[i] if i < len(target_script.split(".")) else ""
        audio = score_clip(clip, clip_speech, reference_audio)

        # Store NISQA scores for seam delta computation
        clip_nisqa = {
            "nisqa_mos": audio.get("nisqa_mos"),
            "nisqa_coloration": audio.get("nisqa_coloration"),
            "nisqa_discontinuity": audio.get("nisqa_discontinuity"),
            "nisqa_noisiness": audio.get("nisqa_noisiness"),
            "nisqa_loudness": audio.get("nisqa_loudness"),
        }
        all_nisqa.append(clip_nisqa)

        # Within-clip consistency (identity drift + sharpness slope)
        within = score_within_clip(clip)

        # Extract a mid-frame for face embedding
        mid_frame = clip.parent / f"shot_{i}_mid.png"
        dur = 3.0
        try:
            dur = float(subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
                capture_output=True, text=True, check=True).stdout.strip())
        except Exception:
            pass

        subprocess.run([
            "ffmpeg", "-y", "-ss", str(dur/2),
            "-i", str(clip), "-frames:v", "1", "-q:v", "1", str(mid_frame),
        ], check=True, capture_output=True)

        # Face embedding for this shot
        face_emb = None
        try:
            img = _load_image(mid_frame)
            face_emb = _get_face_embedding(img)
            if face_emb is not None:
                all_face_embeddings.append(face_emb)
        except Exception:
            pass
        mid_frame.unlink(missing_ok=True)

        # Identity match to reference
        ref_match = None
        if face_emb is not None and reference_image and Path(reference_image).exists():
            try:
                ref_img = _load_image(Path(reference_image))
                ref_emb = _get_face_embedding(ref_img)
                if ref_emb is not None:
                    ref_match = _cosine_similarity(ref_emb, face_emb)
            except Exception:
                pass

        # DINOv2 temporal drift (catches semantic changes beyond face)
        dino_scores = {}
        try:
            eval_dir = clip.parent / f"_seq_eval_{i}"
            eval_dir.mkdir(exist_ok=True)
            eval_frames = []
            for j in range(6):
                fp = eval_dir / f"ef{j}.png"
                subprocess.run(["ffmpeg", "-y", "-ss", str(dur * j / 5), "-i", str(clip),
                                "-frames:v", "1", "-q:v", "1", str(fp)],
                               check=True, capture_output=True)
                eval_frames.append(fp)
            dino_scores = score_dino_drift(eval_frames)

            # LPIPS temporal flicker
            lpips_scores = score_lpips_temporal(eval_frames)
            dino_scores.update(lpips_scores)

            import shutil
            shutil.rmtree(eval_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"DINOv2/LPIPS scoring failed for shot {i}: {e}")

        # Lip sync correlation
        lip_sync = {}
        try:
            lip_sync = score_lip_sync(clip)
        except Exception as e:
            logger.warning(f"Lip sync scoring failed for shot {i}: {e}")

        lip_sync_corr = lip_sync.get("lip_sync_corr")
        lip_sync_score = None
        if lip_sync_corr is not None:
            lip_sync_score = max(0.0, min(10.0, (lip_sync_corr + 1.0) * 5.0))

        shot_composite = audio.get("audio_composite")
        if shot_composite is not None and lip_sync_score is not None:
            shot_composite = round(shot_composite * 0.75 + lip_sync_score * 0.25, 3)

        shot_score = {
            "shot_id": i,
            "wer": audio.get("wer"),
            "utmos": audio.get("utmos"),
            "mos_estimate": audio.get("mos_estimate"),
            "nisqa_mos": audio.get("nisqa_mos"),
            "nisqa_coloration": audio.get("nisqa_coloration"),
            "identity_match": round(ref_match, 4) if ref_match else None,
            "identity_drop": within.get("identity_drop", 0),
            "sharpness_drop_pct": within.get("sharpness_drop_pct", 0),
            "audio_composite": audio.get("audio_composite"),
            "dino_consistency": dino_scores.get("dino_consistency"),
            "dino_drift": dino_scores.get("dino_drift"),
            "lpips_temporal_mean": dino_scores.get("lpips_temporal_mean"),
            "lpips_temporal_std": dino_scores.get("lpips_temporal_std"),
            "lip_sync_corr": lip_sync_corr,
            "lip_sync_offset_ms": lip_sync.get("lip_sync_offset_ms"),
            "transcription_method": audio.get("transcription_method"),
            "shot_composite": shot_composite if shot_composite is not None else audio.get("audio_composite"),
        }
        result["shot_scores"].append(shot_score)

    # === CUT/SEAM SCORES ===
    for i in range(len(clips) - 1):
        logger.info(f"  Scoring seam {i+1}→{i+2}...")
        seam = score_seam(
            clips[i], clips[i + 1],
            nisqa_before=all_nisqa[i] if i < len(all_nisqa) else None,
            nisqa_after=all_nisqa[i + 1] if i + 1 < len(all_nisqa) else None,
        )
        seam["cut_id"] = f"s{i:02d}_s{i+1:02d}"

        # RAFT optical flow discontinuity at seam
        try:
            flow = score_flow_discontinuity(clips[i], clips[i + 1])
            seam["flow_discontinuity"] = flow.get("flow_discontinuity")
            seam["flow_cross_magnitude"] = flow.get("flow_cross_magnitude")
        except Exception as e:
            logger.warning(f"RAFT flow scoring failed at seam {i}→{i+1}: {e}")

        result["cut_scores"].append(seam)

    # === SEQUENCE SCORES ===
    seq = {}

    # Global face drift (first vs last clip)
    if len(all_face_embeddings) >= 2:
        drift = 1.0 - _cosine_similarity(all_face_embeddings[0], all_face_embeddings[-1])
        seq["global_face_drift"] = round(float(drift), 4)

        # Face consistency across all clips
        sims = []
        for i in range(len(all_face_embeddings)):
            for j in range(i+1, len(all_face_embeddings)):
                sims.append(_cosine_similarity(all_face_embeddings[i], all_face_embeddings[j]))
        seq["cross_clip_face_consistency"] = round(float(np.mean(sims)), 4)
        seq["cross_clip_face_min"] = round(float(np.min(sims)), 4)

    # Global identity match (all clips vs reference)
    ref_matches = [s["identity_match"] for s in result["shot_scores"] if s.get("identity_match")]
    if ref_matches:
        seq["global_identity_match"] = round(float(np.mean(ref_matches)), 4)
        seq["identity_degradation"] = round(float(ref_matches[0] - ref_matches[-1]), 4) if len(ref_matches) > 1 else 0

    # Global face preservation: first frame of seg 0 vs last frame of final seg
    if len(clips) >= 2:
        try:
            first_fp = clips[0].parent / "_global_first.png"
            last_fp = clips[-1].parent / "_global_last.png"
            subprocess.run(["ffmpeg", "-y", "-ss", "0", "-i", str(clips[0]),
                            "-frames:v", "1", "-q:v", "1", str(first_fp)],
                           check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-sseof", "-0.042", "-i", str(clips[-1]),
                            "-frames:v", "1", "-q:v", "1", str(last_fp)],
                           check=True, capture_output=True)

            first_emb = _get_face_embedding(_load_image(first_fp))
            last_emb = _get_face_embedding(_load_image(last_fp))
            if first_emb is not None and last_emb is not None:
                arcface_sim = float(_cosine_similarity(first_emb, last_emb))
                seq["global_face_preservation"] = round(arcface_sim, 4)
                logger.info(f"  Global face preservation (ArcFace): {arcface_sim:.4f}")

            # DINOv2 first-vs-last
            try:
                from .score_advanced import _load_dino
                dino_model = _load_dino()
                import torchvision.transforms as T
                import torch
                transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                first_img = _load_image(first_fp)
                last_img = _load_image(last_fp)
                with torch.no_grad():
                    f1 = dino_model(transform(first_img).unsqueeze(0).cuda()).squeeze()
                    f2 = dino_model(transform(last_img).unsqueeze(0).cuda()).squeeze()
                    dino_sim = float(torch.nn.functional.cosine_similarity(f1, f2, dim=0))
                seq["global_dino_preservation"] = round(dino_sim, 4)
                logger.info(f"  Global DINO preservation: {dino_sim:.4f}")
            except Exception as e:
                logger.warning(f"  DINO preservation scoring failed: {e}")

            first_fp.unlink(missing_ok=True)
            last_fp.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"  Global face preservation scoring failed: {e}")

    # Audio consistency
    wers = [s["wer"] for s in result["shot_scores"] if s.get("wer") is not None]
    if wers:
        seq["mean_wer"] = round(float(np.mean(wers)), 4)
        seq["worst_wer"] = round(float(np.max(wers)), 4)

    result["sequence_scores"] = seq

    # === FINAL SCORE ===
    # 50% mean shot + 25% mean seam + 25% sequence (rebalanced for V9)
    shot_avg = np.mean([s.get("shot_composite", 5) for s in result["shot_scores"]]) if result["shot_scores"] else 5
    seam_avg = np.mean([c.get("seam_score", 0.5) for c in result["cut_scores"]]) if result["cut_scores"] else 0.5
    seq_score = seq.get("cross_clip_face_consistency", 0.8) if seq else 0.8

    # Normalize to 0-10 scale
    shot_component = shot_avg  # already 0-10ish
    seam_component = seam_avg * 10  # 0-1 → 0-10
    seq_component = seq_score * 10  # 0-1 → 0-10

    final = (shot_component * 0.50 + seam_component * 0.25 + seq_component * 0.25)

    # === HARD PENALTIES (from human review feedback + V9 metrics) ===

    penalties = []

    # WER hard floor: ANY clip with WER > 0.5 = catastrophic
    worst_wer = seq.get("worst_wer", 0)
    if worst_wer > 0.8:
        penalties.append(("wer_catastrophic", -3.0, f"WER {worst_wer:.2f} in worst clip"))
    elif worst_wer > 0.5:
        penalties.append(("wer_severe", -2.0, f"WER {worst_wer:.2f} in worst clip"))
    elif worst_wer > 0.3:
        penalties.append(("wer_moderate", -1.0, f"WER {worst_wer:.2f} in worst clip"))

    # Per-clip minimum gate: sequence can't score higher than 2x its worst clip
    shot_composites = [s.get("shot_composite", 5) for s in result["shot_scores"]]
    if shot_composites:
        worst_shot = min(shot_composites)
        if worst_shot < 4.0:
            penalties.append(("worst_shot_floor", -(5.0 - worst_shot), f"Worst clip score: {worst_shot:.1f}"))

    # Face drift
    if seq.get("global_face_drift", 0) > 0.3:
        penalties.append(("face_drift", -1.0, f"Drift: {seq['global_face_drift']:.3f}"))

    # Face jump at seam (using new seam scorer)
    for c in result["cut_scores"]:
        face_d = c.get("face_delta")
        if face_d is not None and face_d > 0.3:
            penalties.append(("face_jump", -0.5, f"Face delta {face_d:.3f} at {c.get('cut_id')}"))

    # NISQA Coloration — wiriness penalty
    for s in result["shot_scores"]:
        col = s.get("nisqa_coloration")
        if col is not None and col < 3.0:
            penalties.append(("wiry_audio", -1.0, f"NISQA Coloration {col:.2f} in shot {s['shot_id']}"))

    # Lip sync penalties
    for s in result["shot_scores"]:
        corr = s.get("lip_sync_corr")
        if corr is not None and corr < 0.10:
            penalties.append(("lip_sync_low", -1.0, f"Lip sync corr {corr:.3f} in shot {s['shot_id']}"))
        offset = abs(s.get("lip_sync_offset_ms", 0) or 0)
        if offset > 120:
            penalties.append(("lip_sync_offset", -0.5, f"Lip sync offset {offset:.0f}ms in shot {s['shot_id']}"))

    # Within-clip identity degradation
    for s in result["shot_scores"]:
        id_drop = s.get("identity_drop", 0)
        if id_drop > 0.25:
            penalties.append(("identity_degrad", -0.5, f"Identity drop {id_drop:.3f} in shot {s['shot_id']}"))

    # Within-clip sharpness degradation
    for s in result["shot_scores"]:
        sharp_drop = s.get("sharpness_drop_pct", 0)
        if sharp_drop > 30:
            penalties.append(("sharpness_degrad", -0.5, f"Sharpness drop {sharp_drop:.0f}% in shot {s['shot_id']}"))

    # WER count: how many clips have WER > 0.5
    bad_wer_count = sum(1 for s in result["shot_scores"] if (s.get("wer") or 0) > 0.5)
    if bad_wer_count > 1:
        penalties.append(("multi_bad_wer", -bad_wer_count * 0.5, f"{bad_wer_count} clips with WER > 0.5"))

    # Seam loudness jump
    for c in result["cut_scores"]:
        ld = c.get("loudness_delta", 0)
        if ld > 0.1:
            penalties.append(("loudness_jump", -0.3, f"Loudness delta {ld:.3f} at {c.get('cut_id')}"))

    total_penalty = sum(p[1] for p in penalties)
    result["penalties"] = [{"type": p[0], "amount": p[1], "reason": p[2]} for p in penalties]

    result["final_score"] = round(max(0, final + total_penalty), 2)

    return result
