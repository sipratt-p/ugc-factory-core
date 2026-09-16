"""Automated UGC video producer with quality gates.

Takes a character + script → produces a finished video with:
- Per-segment quality gating (auto-reseed on failures)
- Automatic leading silence trim
- Automatic post-speech audio fade
- Optional background music mixing
- Hard concat assembly with A/V sync preservation
"""

import gc
import logging
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

from . import config
from .autoresearch import TestConfig, generate_test_clip, RESULTS_DIR
from .scene_bible import SceneBible
from .score_seam import score_lip_sync
from .quality_gate import generate_with_quality_gate, SegmentQuality

logger = logging.getLogger("video_producer")


@dataclass
class VideoResult:
    """Result of a full video production."""
    final_path: Path
    duration: float
    segments: list[SegmentQuality]
    gen_time: float
    passed_count: int
    total_count: int


def _get_duration(path: Path) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())


def _process_segment(clip: Path, output: Path, is_last: bool = False) -> Path:
    """Trim leading silence + chop at speech end using trim/atrim filters.

    Uses hybrid silence detection: 0.05s for leading (fine-grained),
    0.2s for trailing (skip inter-word pauses), fallback to 0.05s if needed.
    """
    dur = _get_duration(clip)

    # Leading silence: fine-grained (0.05s)
    sil_fine = subprocess.run(
        ["ffmpeg", "-i", str(clip), "-af", "silencedetect=n=-25dB:d=0.05",
         "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    lead_ends = re.findall(r"silence_end: ([\d.]+)", sil_fine)
    lead_end = float(lead_ends[0]) if lead_ends else 0

    # Trailing silence: use the LAST silence_start after speech begins
    # This always catches the real trailing silence, not inter-word pauses
    fine_starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", sil_fine)]
    post_lead = [s for s in fine_starts if s > lead_end + 0.3]
    speech_end = post_lead[-1] if post_lead else dur - 0.15

    trim_start = max(0, lead_end - 0.15)
    trim_end = min(dur, speech_end + 0.15)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(clip),
        "-vf", f"trim=start={trim_start}:end={trim_end},setpts=PTS-STARTPTS",
        "-af", f"atrim=start={trim_start}:end={trim_end},asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        str(output),
    ], capture_output=True)
    return output


def produce_video(
    bible: SceneBible,
    scripts: list[str],
    output_dir: Path = None,
    num_frames: int = 121,
    music_path: Path = None,
    music_volume_db: float = -20,
    gpu_id: int = None,
    skip_lint: bool = False,
    best_of: int = 1,
) -> VideoResult:
    """Produce a complete UGC video from a character bible and script lines.

    Each segment is generated with quality gating — failed segments are
    automatically regenerated with different seeds.

    num_frames defaults to 121 (V8/V13 sweet spot: sharper per-frame, less
    within-clip drift, better lip sync than 129/161).

    Scripts are linted before any GPU time (multi-sentence, link-in-bio,
    length vs frame window — the known lip-sync killers). ERROR-level issues
    raise ValueError; pass skip_lint=True to bypass.
    """
    if not skip_lint:
        from .script_lint import lint_scripts
        errors, _ = lint_scripts(scripts, num_frames=num_frames)
        if errors:
            raise ValueError(
                f"script lint found {errors} ERROR(s) — fix the segments above "
                f"or pass skip_lint=True")

    if output_dir is None:
        test_id = f"produce_{uuid.uuid4().hex[:6]}"
        output_dir = RESULTS_DIR / test_id
    # Absolute paths required: the ffmpeg concat demuxer resolves relative
    # entries against the LIST FILE's directory, not the cwd.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = gpu_id if gpu_id is not None else config.VIDEO_GPU
    start_time = time.time()

    logger.info(f"Producing {len(scripts)} segments → {output_dir}")
    logger.info(f"Character: {bible.character_name}, Frames: {num_frames}, GPU: {gpu}")

    # Generate each segment with quality gate
    clips = []
    qualities = []

    for i, script in enumerate(scripts):
        gc.collect()
        torch.cuda.empty_cache()
        wc = len(script.split())
        logger.info(f"Segment {i + 1}/{len(scripts)} ({wc} words)")

        def gen_fn(seed, tid, seg_script, seg_frames):
            cfg = TestConfig(
                test_id=tid,
                height=config.GENERATION_HEIGHT, width=config.GENERATION_WIDTH,
                num_frames=seg_frames, seed=seed,
                identity_guidance=config.IDENTITY_GUIDANCE,
                audio_guidance=config.AUDIO_GUIDANCE,
                video_guidance=config.VIDEO_GUIDANCE,
                video_rescale=config.VIDEO_RESCALE,
                distilled_s1=config.DISTILLED_S1, distilled_s2=config.DISTILLED_S2,
                prompt_visual=bible.build_visual_prompt(),
                prompt_speech=seg_script,
                prompt_sounds=bible.build_sounds_prompt(),
                first_frame=bible.reference_image,
                reference_audio=bible.reference_audio,
            )
            generate_test_clip(cfg, gpu_id=gpu)
            seg_dir = RESULTS_DIR / tid
            hq = sorted(seg_dir.glob("*_hq.mp4"))
            return hq[-1] if hq else sorted(seg_dir.glob("output_*.mp4"))[-1]

        def score_fn(clip_path):
            return score_lip_sync(clip_path)

        # best_of > 1 = quality mode (the V8 beam-search paradigm): generate
        # K independent gated candidates from disjoint seed banks and keep
        # the one with the highest SyncNet LSE-C among passers.
        SEED_BANKS = [[42, 137, 256], [1337, 2025, 777], [9001, 31337, 4242]]
        candidates = []
        for k in range(max(1, best_of)):
            clip_k, quality_k, _ = generate_with_quality_gate(
                generate_fn=gen_fn,
                script=script,
                test_id_base=f"{output_dir.name}_seg{i:02d}_k{k}",
                score_fn=score_fn,
                num_frames=num_frames,
                seeds=SEED_BANKS[k % len(SEED_BANKS)],
            )
            candidates.append((clip_k, quality_k))
            logger.info(
                f"  candidate {k}: LSE-C={quality_k.syncnet_lse_c:.2f} "
                f"({quality_k.syncnet_quality}) "
                f"{'PASS' if quality_k.passes else 'BEST-EFFORT'}"
            )

        # Rank: pass > quality label (GOOD beats FAIR — the label also weighs
        # LSE-D distance) > raw LSE-C as tiebreak.
        _QUALITY_RANK = {"GOOD": 3, "FAIR": 2, "MEDIOCRE": 1, "POOR": 0, "UNKNOWN": 0}
        clip, quality = max(
            candidates,
            key=lambda c: (c[1].passes,
                           _QUALITY_RANK.get(c[1].syncnet_quality, 0),
                           c[1].syncnet_lse_c))
        if best_of > 1:
            logger.info(f"  → selected LSE-C={quality.syncnet_lse_c:.2f}")

        clips.append(clip)
        qualities.append(quality)
        logger.info(
            f"  → corr={quality.lip_sync_corr:.3f} mouth={quality.mouth_movement:.4f} "
            f"{'PASS' if quality.passes else 'BEST-EFFORT'}"
        )

    # Post-process: trim/atrim each segment
    logger.info("Post-processing segments...")
    processed = []
    for i, clip in enumerate(clips):
        out = output_dir / f"processed_{i:02d}.mp4"
        _process_segment(clip, out)
        td = _get_duration(out)
        logger.info(f"  seg{i}: → {td:.1f}s")
        processed.append(out)

    # Assemble: hard concat
    logger.info("Assembling...")
    concat_file = output_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for p in processed:
            f.write(f"file '{p}'\n")

    # 15M + slow preset: this is the LAST video encode (the music mux copies
    # the stream), so dropping bitrate here throws away segment quality.
    assembled = output_dir / "assembled.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        str(assembled),
    ], check=True, capture_output=True)

    # Add background music if provided
    if music_path and music_path.exists():
        logger.info(f"Mixing background music: {music_path.name} at {music_volume_db}dB")
        video_dur = _get_duration(assembled)
        final = output_dir / "final.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(assembled),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{video_dur},"
            f"volume={music_volume_db}dB,"
            f"afade=t=in:d=1,afade=t=out:st={video_dur - 1.5}:d=1.5[bg];"
            f"[0:a]volume=1.0[speech];"
            f"[speech][bg]amerge=inputs=2,"
            f"pan=stereo|c0=c0+c2|c1=c1+c3[outa]",
            "-map", "0:v", "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            str(final),
        ], check=True, capture_output=True)
    else:
        # assembled is already the final-grade encode — don't re-encode down.
        final = output_dir / "final.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(assembled),
            "-c:v", "copy", "-c:a", "copy", str(final),
        ], check=True, capture_output=True)

    duration = _get_duration(final)
    gen_time = time.time() - start_time
    passed = sum(1 for q in qualities if q.passes)

    # Print summary
    logger.info("=" * 80)
    logger.info(f"PRODUCTION COMPLETE: {final}")
    logger.info(f"Duration: {duration:.1f}s | Gen time: {gen_time:.0f}s")
    logger.info(f"Quality: {passed}/{len(qualities)} passed gate")
    for i, q in enumerate(qualities):
        status = "PASS" if q.passes else f"BEST-EFFORT ({q.failure_reason})"
        logger.info(f"  seg{i}: corr={q.lip_sync_corr:.3f} mouth={q.mouth_movement:.4f} {status}")
    logger.info("=" * 80)

    return VideoResult(
        final_path=final,
        duration=duration,
        segments=qualities,
        gen_time=gen_time,
        passed_count=passed,
        total_count=len(qualities),
    )
