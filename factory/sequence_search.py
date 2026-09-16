"""Sequence search — find the best assembly path through candidate clips.

Given K candidates per segment, build a lattice and search for the
optimal path that maximizes shot quality + transition quality.

GPT5 Pro: "Node cost = shot quality. Edge cost = face continuity +
voice continuity + cut quality."
"""

import logging
import subprocess
import numpy as np
from pathlib import Path

from .score_visual import _cosine_similarity
from .handle_gen import assemble_with_handles

logger = logging.getLogger(__name__)


def score_transition(clip_a: dict, clip_b: dict) -> float:
    """Score the transition quality between two candidate clips.

    Uses boundary-aware metrics (V10):
    - Face similarity at the actual cut point (last frame A vs first frame B)
    - Loudness jump at the boundary (tail RMS A vs head RMS B)
    - NISQA quality consistency between clips
    - Falls back to mid-frame similarity if boundary data unavailable

    Returns 0-1 (higher = better transition).
    """
    score = 0.5  # neutral default

    # Primary: last frame of A vs first frame of B (the actual cut point)
    last_emb = clip_a.get("last_frame_embedding")
    first_emb = clip_b.get("first_frame_embedding")

    if last_emb is not None and first_emb is not None:
        face_sim = _cosine_similarity(last_emb, first_emb)
        score = float(face_sim) * 0.6  # 60% weight on face match at cut
    elif clip_a.get("face_embedding") is not None and clip_b.get("face_embedding") is not None:
        # Fallback: mid-frame similarity
        face_sim = _cosine_similarity(clip_a["face_embedding"], clip_b["face_embedding"])
        score = float(face_sim) * 0.5

    # Loudness continuity: small delta = better
    tail_rms = clip_a.get("tail_rms", 0)
    head_rms = clip_b.get("head_rms", 0)
    if tail_rms > 0 or head_rms > 0:
        loudness_delta = abs(tail_rms - head_rms)
        # Perfect match = 0.2 bonus, big jump = 0 bonus
        loudness_bonus = max(0, 0.2 - loudness_delta * 2)
        score += loudness_bonus

    # NISQA quality consistency
    nisqa_a = clip_a.get("audio_scores", {}).get("nisqa_mos")
    nisqa_b = clip_b.get("audio_scores", {}).get("nisqa_mos")
    if nisqa_a is not None and nisqa_b is not None:
        nisqa_delta = abs(nisqa_a - nisqa_b)
        # Small quality gap = 0.2 bonus
        nisqa_bonus = max(0, 0.2 - nisqa_delta * 0.2)
        score += nisqa_bonus

    return min(1.0, max(0.0, score))


def greedy_search(
    candidates_per_segment: list[list[dict]],
    shot_weight: float = 0.7,
    edge_weight: float = 0.3,
) -> list[dict]:
    """Greedy best-path search through the candidate lattice.

    For each segment, pick the candidate that maximizes:
    shot_weight * shot_score + edge_weight * transition_score_to_previous

    Returns the winning candidate for each segment.
    """
    n_segments = len(candidates_per_segment)
    if n_segments == 0:
        return []

    path = []

    # First segment: just pick best shot
    best_first = max(candidates_per_segment[0],
                     key=lambda c: c.get("shot_score", 0))
    path.append(best_first)
    logger.info(f"  Segment 0: picked candidate {best_first['index']} "
               f"(shot={best_first.get('shot_score',0):.2f})")

    # Subsequent segments: maximize shot + transition
    for seg_idx in range(1, n_segments):
        prev = path[-1]
        best_score = -1
        best_candidate = None

        for candidate in candidates_per_segment[seg_idx]:
            if candidate.get("clip_path") is None:
                continue

            shot = candidate.get("shot_score", 0)
            transition = score_transition(prev, candidate)
            combined = shot_weight * shot + edge_weight * transition

            if combined > best_score:
                best_score = combined
                best_candidate = candidate
                best_candidate["_transition_score"] = round(transition, 3)

        if best_candidate:
            path.append(best_candidate)
            logger.info(f"  Segment {seg_idx}: picked candidate {best_candidate['index']} "
                       f"(shot={best_candidate.get('shot_score',0):.2f}, "
                       f"transition={best_candidate.get('_transition_score',0):.2f})")
        else:
            # Fallback: pick any valid candidate
            for c in candidates_per_segment[seg_idx]:
                if c.get("clip_path"):
                    path.append(c)
                    break

    return path


def assemble_best_path(
    path: list[dict],
    output_path: Path,
    use_handles: bool = True,
    handle_frames: int = 15,
) -> dict:
    """Assemble the winning path into a final video.

    Uses audio-aware handle splicing for smooth transitions.
    """
    clips = [Path(c["clip_path"]) for c in path if c.get("clip_path")]

    if not clips:
        return {"error": "No valid clips in path"}

    if use_handles and len(clips) > 1:
        result = assemble_with_handles(clips, output_path, handle_frames)
    else:
        # Single clip or no handles
        if len(clips) == 1:
            import shutil
            shutil.copy2(clips[0], output_path)
            result = {"method": "single_clip"}
        else:
            # Hard concat fallback
            list_file = output_path.parent / "concat.txt"
            with open(list_file, "w") as f:
                for c in clips:
                    f.write(f"file '{c}'\n")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", str(output_path),
            ], check=True, capture_output=True)
            list_file.unlink(missing_ok=True)
            result = {"method": "hard_concat"}

    # Add path info to result
    result["path"] = [{
        "segment": i,
        "candidate_index": c.get("index"),
        "seed": c.get("seed"),
        "shot_score": c.get("shot_score"),
        "transition_score": c.get("_transition_score"),
        "wer": c.get("wer"),
        "utmos": c.get("utmos"),
        "nisqa_coloration": c.get("nisqa_coloration"),
        "identity_drop": c.get("within_clip", {}).get("identity_drop") if c.get("within_clip") else None,
        "sharpness_drop_pct": c.get("within_clip", {}).get("sharpness_drop_pct") if c.get("within_clip") else None,
    } for i, c in enumerate(path)]

    result["mean_shot_score"] = round(np.mean([c.get("shot_score", 0) for c in path]), 3)

    logger.info(f"Assembled {len(clips)} clips, mean shot score: {result['mean_shot_score']}")
    return result
