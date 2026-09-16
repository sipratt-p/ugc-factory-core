"""Best-of-K candidate generation for sequence search.

Instead of generating one clip per segment and hoping it's good,
generate K candidates with different seeds and let the sequence
search pick the best assembly path.

GPT5 Pro: "Stop treating one render per clip as the unit of optimization."

Dual-GPU parallel generation: with two GPUs (config.ALL_GPUS) we run
2 candidates simultaneously, one per GPU, so a K=2 beam takes the
same wall-clock time as K=1.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .autoresearch import TestConfig, generate_test_clip
from .audio_utils import resolve_reference_audio_path
from .score_av import score_clip
from .score_visual import score_visual, _load_image, _get_face_embedding
from .scene_bible import SceneBible
from .score_seam import score_lip_sync

logger = logging.getLogger(__name__)


def _generate_and_score_one(
    i: int,
    seed: int,
    speech: str,
    bible: SceneBible,
    output_dir: Path,
    num_frames: int,
    first_frame: str,
    gpu_id: int,
) -> dict:
    """Generate and score a single candidate on a specific GPU."""
    clip_id = uuid.uuid4().hex[:8]
    clip_dir = output_dir / f"candidate_{i}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Generating candidate {i+1} (seed={seed}, GPU={gpu_id})...")

    reference_audio = resolve_reference_audio_path(
        Path(bible.reference_image).parent,
        Path(bible.reference_audio) if bible.reference_audio else None,
    )

    cfg = TestConfig(
        test_id=clip_id,
        height=config.GENERATION_HEIGHT, width=config.GENERATION_WIDTH,
        num_frames=num_frames, seed=seed,
        identity_guidance=config.IDENTITY_GUIDANCE,
        audio_guidance=config.AUDIO_GUIDANCE,
        distilled_s1=config.DISTILLED_S1, distilled_s2=config.DISTILLED_S2,
        video_guidance=config.VIDEO_GUIDANCE, video_rescale=config.VIDEO_RESCALE,
        prompt_visual=bible.build_visual_prompt(),
        prompt_speech=speech,
        prompt_sounds=bible.build_sounds_prompt(),
        first_frame=first_frame,
        reference_audio=str(reference_audio) if reference_audio else "",
    )

    try:
        raw_clip = generate_test_clip(cfg, gpu_id=gpu_id)

        # Re-encode at proper bitrate
        import subprocess
        hq_clip = clip_dir / f"clip_hq.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(raw_clip),
            "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
            "-c:a", "aac", "-b:a", "256k", str(hq_clip),
        ], check=True, capture_output=True)

        # Score the candidate
        audio_scores = score_clip(hq_clip, speech.strip('"'), reference_audio)
        lip_sync = score_lip_sync(hq_clip)

        # Get face embedding for transition scoring
        from .autoresearch import extract_evaluation_frames
        frames = extract_evaluation_frames(hq_clip, clip_dir / "eval", count=4)
        visual_scores = score_visual(frames, Path(bible.reference_image))

        import subprocess as _sp
        face_emb = None
        first_emb = None
        last_emb = None

        mid_frame = frames[len(frames)//2] if frames else None
        if mid_frame:
            try:
                face_emb = _get_face_embedding(_load_image(mid_frame))
            except Exception:
                pass

        # Extract first and last frame for seam-aware beam search
        try:
            first_fp = clip_dir / "eval" / "_first.png"
            last_fp = clip_dir / "eval" / "_last.png"
            _sp.run(["ffmpeg", "-y", "-ss", "0", "-i", str(hq_clip),
                     "-frames:v", "1", "-q:v", "1", str(first_fp)],
                    check=True, capture_output=True)
            _sp.run(["ffmpeg", "-y", "-sseof", "-0.042", "-i", str(hq_clip),
                     "-frames:v", "1", "-q:v", "1", str(last_fp)],
                    check=True, capture_output=True)
            first_emb = _get_face_embedding(_load_image(first_fp))
            last_emb = _get_face_embedding(_load_image(last_fp))
            first_fp.unlink(missing_ok=True)
            last_fp.unlink(missing_ok=True)
        except Exception:
            pass

        # Within-clip consistency: first frame vs last frame
        from .score_seam import score_within_clip
        within_clip = score_within_clip(hq_clip)

        # Shot quality score
        ac = audio_scores.get("audio_composite", 0)
        vc = visual_scores.get("visual_composite", 0)
        lip_sync_corr = lip_sync.get("lip_sync_corr")
        lip_sync_score = None
        if lip_sync_corr is not None:
            lip_sync_score = max(0.0, min(10.0, (lip_sync_corr + 1.0) * 5.0))

        weighted_terms = []
        if ac:
            weighted_terms.append((ac, 0.40))
        if vc:
            weighted_terms.append((vc, 0.35))
        if lip_sync_score is not None:
            weighted_terms.append((lip_sync_score, 0.25))

        if weighted_terms:
            shot_score = sum(score * weight for score, weight in weighted_terms) / sum(
                weight for _, weight in weighted_terms
            )
        else:
            shot_score = 0

        # Hard reject: WER > 0.5
        if (audio_scores.get("wer") or 0) > 0.5:
            shot_score *= 0.3

        # Hard reject: low lip sync correlation or large temporal offset
        lip_sync_offset = abs(lip_sync.get("lip_sync_offset_ms", 0) or 0)
        if lip_sync_corr is not None and lip_sync_corr < 0.10:
            shot_score *= 0.5
            logger.warning(f"    Lip sync corr {lip_sync_corr:.3f} < 0.10 — sync penalty")
        if lip_sync_offset > 120:
            shot_score *= 0.7
            logger.warning(f"    Lip sync offset {lip_sync_offset:.0f}ms > 120ms — offset penalty")

        # Hard reject: NISQA Coloration < 3.0 (wiry/metallic audio)
        nisqa_col = audio_scores.get("nisqa_coloration")
        if nisqa_col is not None and nisqa_col < 3.0:
            shot_score *= 0.5
            logger.warning(f"    NISQA Coloration {nisqa_col:.2f} < 3.0 — wiry audio penalty")

        # Hard reject: within-clip identity degradation > 0.25
        identity_drop = within_clip.get("identity_drop", 0)
        if identity_drop > 0.25:
            shot_score *= 0.6
            logger.warning(f"    Within-clip identity drop {identity_drop:.3f} > 0.25 — degradation penalty")

        # Hard reject: within-clip sharpness drop > 30%
        sharpness_drop = within_clip.get("sharpness_drop_pct", 0)
        if sharpness_drop > 30:
            shot_score *= 0.7
            logger.warning(f"    Sharpness drop {sharpness_drop:.0f}% > 30% — degradation penalty")

        # Get tail/head loudness for transition scoring
        tail_rms = 0.0
        head_rms = 0.0
        try:
            import soundfile as sf
            tail_wav = clip_dir / "_tail_rms.wav"
            head_wav = clip_dir / "_head_rms.wav"
            _sp.run(["ffmpeg", "-y", "-sseof", "-0.25", "-i", str(hq_clip),
                     "-vn", "-ar", "16000", "-ac", "1", str(tail_wav)],
                    check=True, capture_output=True)
            _sp.run(["ffmpeg", "-y", "-t", "0.25", "-i", str(hq_clip),
                     "-vn", "-ar", "16000", "-ac", "1", str(head_wav)],
                    check=True, capture_output=True)
            import numpy as _np
            d, _ = sf.read(str(tail_wav))
            tail_rms = float(_np.sqrt(_np.mean(d**2))) if len(d) > 0 else 0
            d, _ = sf.read(str(head_wav))
            head_rms = float(_np.sqrt(_np.mean(d**2))) if len(d) > 0 else 0
            tail_wav.unlink(missing_ok=True)
            head_wav.unlink(missing_ok=True)
        except Exception:
            pass

        result = {
            "index": i,
            "seed": seed,
            "gpu_id": gpu_id,
            "clip_path": str(hq_clip),
            "audio_scores": audio_scores,
            "visual_scores": visual_scores,
            "within_clip": within_clip,
            "shot_score": round(shot_score, 3),
            "face_embedding": face_emb,
            "first_frame_embedding": first_emb,
            "last_frame_embedding": last_emb,
            "tail_rms": round(tail_rms, 4),
            "head_rms": round(head_rms, 4),
            "wer": audio_scores.get("wer"),
            "utmos": audio_scores.get("utmos"),
            "nisqa_coloration": nisqa_col,
            "lip_sync_corr": lip_sync_corr,
            "lip_sync_offset_ms": lip_sync.get("lip_sync_offset_ms"),
        }

        logger.info(f"    Candidate {i+1}: shot={shot_score:.2f}, "
                   f"WER={audio_scores.get('wer')}, UTMOS={audio_scores.get('utmos')}, "
                   f"lip_sync={lip_sync_corr}, NISQA_col={nisqa_col}, id_drop={identity_drop:.3f}")
        return result

    except Exception as e:
        logger.error(f"    Candidate {i+1} failed: {e}")
        return {
            "index": i, "seed": seed, "clip_path": None,
            "shot_score": 0, "error": str(e),
        }


def generate_candidates(
    speech: str,
    bible: SceneBible,
    output_dir: Path,
    k: int = 3,
    base_seed: int = 42,
    num_frames: int = 121,
    first_frame_override: str = None,
    parallel: bool = True,
) -> list[dict]:
    """Generate K candidate clips for a single script segment.

    Each candidate uses a different seed but identical prompt/conditioning.
    With parallel=True and K>=2, runs candidates on separate GPUs simultaneously.
    Returns list of scored candidates sorted by quality.

    Args:
        first_frame_override: If provided, use this image as the first frame
            instead of the bible's reference image. Used for last-frame
            conditioning: pass the last frame of the previous clip to make
            this clip's opening match the previous clip's ending.
        parallel: If True, run up to 2 candidates simultaneously on separate GPUs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    first_frame = first_frame_override or bible.reference_image
    gpus = config.ALL_GPUS
    max_parallel = len(gpus) if parallel else 1

    candidates = []

    # Process candidates in batches of max_parallel
    for batch_start in range(0, k, max_parallel):
        batch_end = min(batch_start + max_parallel, k)
        batch_size = batch_end - batch_start

        if batch_size == 1 or max_parallel == 1:
            # Sequential: single candidate
            for i in range(batch_start, batch_end):
                result = _generate_and_score_one(
                    i=i, seed=base_seed + i, speech=speech, bible=bible,
                    output_dir=output_dir, num_frames=num_frames,
                    first_frame=first_frame, gpu_id=gpus[0],
                )
                candidates.append(result)
        else:
            # Parallel: one candidate per GPU
            logger.info(f"  Parallel batch: candidates {batch_start+1}-{batch_end} "
                       f"on GPUs {gpus[:batch_size]}")
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for j, i in enumerate(range(batch_start, batch_end)):
                    gpu = gpus[j % len(gpus)]
                    future = executor.submit(
                        _generate_and_score_one,
                        i=i, seed=base_seed + i, speech=speech, bible=bible,
                        output_dir=output_dir, num_frames=num_frames,
                        first_frame=first_frame, gpu_id=gpu,
                    )
                    futures[future] = i

                for future in as_completed(futures):
                    result = future.result()
                    candidates.append(result)

    # Sort by shot score (best first)
    candidates.sort(key=lambda c: c.get("shot_score", 0), reverse=True)
    return candidates
