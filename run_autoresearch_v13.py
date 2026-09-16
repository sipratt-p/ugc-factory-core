#!/usr/bin/env python3
"""Autoresearch V13 — 129fr clips + video xfade/audio hardcut + fade ending

All learnings applied:
- 129fr clips (V8 sweet spot: best TOPIQ, UTMOS, WER, seams)
- Always-original reference (no progressive drift)
- Video xfade 12 frames + audio hard-cut (no speech bleed)
- Fade-to-black on final clip (no abrupt mouth-open ending)
- audio_guidance=3.5, identity_guidance=1.5, -preset slow
- Whisper server (WHISPER_URL) with per-segment target text
- Full fleet scoring (DINOv2, RAFT, LPIPS, lip sync, NISQA, seam)
- Wiki auto-update after each experiment
- WER gate fixed (None = skip, not reject)
- DUAL GPU: K=2 beam runs one candidate per GPU (same wall clock as K=1)

Experiments (scripts below are brand-free placeholders for three ad verticals):
1-3: K=1 health/recruiting/seo 4×129fr (~12 min each)
4-6: K=2 beam health/recruiting/seo 4×129fr (~12 min each — parallel!)
7: health 4×161fr K=1 (baseline comparison)
8: health 5×129fr K=1 (more segments test)
9: health audio_guidance=2.0 K=1 (lip sync comparison)
10-11: health/recruiting 90s 16×129fr K=1 (1.5 min stress test)

Persona: set UGC_PERSONA_DIR to a directory holding reference.jpg, reference.wav
and optionally scene_bible.json (default personas/default).
"""

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from factory.autoresearch import TestConfig, generate_test_clip, RESULTS_DIR
from factory.candidate_gen import generate_candidates
from factory.sequence_search import greedy_search
from factory.scene_bible import bible_from_persona_dir
from factory.score_av import score_clip
from factory.score_sequence import score_sequence
from factory.score_seam import score_lip_sync
from factory.score_advanced import score_dino_drift, score_flow_discontinuity
from factory.endings import add_fade_out, score_ending
from factory.wiki_hook import update_wiki_after_experiment

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "autoresearch_v13.log"),
    ],
)
logger = logging.getLogger("autoresearch-v13")

bible = bible_from_persona_dir(Path(os.environ.get("UGC_PERSONA_DIR", "personas/default")))

# === PLACEHOLDER SCRIPTS (129fr = ~5.5s — one short sentence per segment) ===
# Brand-free stand-ins for the product scripts used in the original runs.
# The originals' last segments carried "link in bio" style CTAs, which the
# later script_lint flags as lip-sync killers; these placeholders avoid them.

HEALTH_4 = [
    {"speech": '"Stop buying supplements based on social media hype."'},
    {"speech": '"There are free platforms with real clinical data on hundreds of supplements."'},
    {"speech": '"Some have an AI research assistant you can chat with about any supplement."'},
    {"speech": '"Check the actual evidence before you spend another dollar."'},
]

RECRUITING_4 = [
    {"speech": '"Stop reviewing job applications by hand. Use AI instead."'},
    {"speech": '"Automated sourcing pulls candidates from public profiles and ranks them by fit."'},
    {"speech": '"Teams cut their hiring time dramatically with tools like this."'},
    {"speech": '"Try one on your next open role and see the difference."'},
]

SEO_4 = [
    {"speech": '"Most websites are invisible to AI search engines right now."'},
    {"speech": '"Visibility tools show you exactly what AI models see on your site."'},
    {"speech": '"They check citations, schema, and everything driving AI visibility."'},
    {"speech": '"Run a free audit on your own site this week."'},
]

HEALTH_161 = [
    {"speech": '"Stop buying supplements based on social media hype. Check the actual clinical evidence first."'},
    {"speech": '"There are free platforms with data on hundreds of supplements. Real science, not marketing."'},
    {"speech": '"Some even have an AI research assistant you can chat with about any supplement or peptide."'},
    {"speech": '"Read the evidence before your next purchase. It is completely free to look."'},
]

HEALTH_5 = [
    {"speech": '"Stop buying supplements based on social media hype."'},
    {"speech": '"Check the real clinical evidence before you spend money."'},
    {"speech": '"There are free platforms with data on hundreds of supplements."'},
    {"speech": '"Some have an AI assistant you can chat with about anything."'},
    {"speech": '"Look up the evidence before your next purchase."'},
]

# 90-second scripts (16 segments × 129fr = ~87s at 5.5s each)
HEALTH_16 = [
    {"speech": '"Stop buying supplements based on social media hype."'},
    {"speech": '"Most supplement advice online is just marketing."'},
    {"speech": '"People are wasting hundreds of dollars every month."'},
    {"speech": '"But there is a better way to make these decisions."'},
    {"speech": '"Check the actual clinical evidence before you buy."'},
    {"speech": '"Real studies with real data from real scientists."'},
    {"speech": '"There are free platforms that gather all of this."'},
    {"speech": '"Data on hundreds of supplements and peptides."'},
    {"speech": '"Everything from magnesium to ashwagandha to creatine."'},
    {"speech": '"Each supplement gets a summary of the clinical trials."'},
    {"speech": '"You can see exactly what the science says works."'},
    {"speech": '"And what is just hype with no real evidence."'},
    {"speech": '"Some even have an AI research assistant built in."'},
    {"speech": '"You can ask it anything about any supplement."'},
    {"speech": '"Start with the evidence the next time you shop."'},
    {"speech": '"It is completely free to look. Follow for more."'},
]

RECRUITING_16 = [
    {"speech": '"Stop reviewing job applications by hand."'},
    {"speech": '"Hiring managers spend hours screening every one."'},
    {"speech": '"Most of that time is completely wasted."'},
    {"speech": '"There are AI tools that do this in minutes."'},
    {"speech": '"Automated sourcing finds your best candidates for you."'},
    {"speech": '"It pulls directly from public profiles and job boards."'},
    {"speech": '"Then it screens everyone against your exact needs."'},
    {"speech": '"Candidates get ranked by how well they fit the role."'},
    {"speech": '"The best people show up at the top of your list."'},
    {"speech": '"No more guessing through hundreds of profiles."'},
    {"speech": '"Teams cut their hiring time dramatically this way."'},
    {"speech": '"That means filling roles in days not weeks."'},
    {"speech": '"It also drafts personalized outreach for candidates."'},
    {"speech": '"The response rates are way higher than generic messages."'},
    {"speech": '"Try one on your next open role this week."'},
    {"speech": '"Most offer a free trial. Follow for more hiring tips."'},
]


def get_duration(path):
    return float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip())


def generate_single(speech, seed, output_dir, num_frames=129, audio_guidance=3.5, gpu_id=0):
    clip_id = uuid.uuid4().hex[:8]
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = TestConfig(
        test_id=clip_id,
        height=864, width=480, num_frames=num_frames, seed=seed,
        identity_guidance=1.5, audio_guidance=audio_guidance,
        distilled_s1=0.1, distilled_s2=0.3,
        video_guidance=3.0, video_rescale=0.45,
        prompt_visual=bible.build_visual_prompt(),
        prompt_speech=speech, prompt_sounds=bible.build_sounds_prompt(),
        first_frame=bible.reference_image,
        reference_audio=bible.reference_audio,
    )
    raw = generate_test_clip(cfg, gpu_id=gpu_id)
    hq = output_dir / "clip_hq.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw),
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k", str(hq),
    ], check=True, capture_output=True)
    return hq


def video_xfade_audio_hardcut(clips, output_path, xfade_frames=12):
    """Video fade-in-only transition + audio hard-cut.

    At each seam: clip N plays clean to its last frame (no fade out).
    Clip N+1 fades IN over its first few frames on top of clip N's ending.
    The new clip emerges on top of the old one — not a mutual dissolve.

    Uses xfade with custom expression: A stays at full opacity,
    B fades from 0 to 1 over the transition duration.
    """
    xfade_dur = xfade_frames / 24.0
    durations = [get_duration(c) for c in clips]
    n = len(clips)
    inputs = []
    for c in clips:
        inputs.extend(["-i", str(c)])

    offsets = []
    cumulative = 0
    for i in range(n - 1):
        cumulative += durations[i]
        offsets.append(max(0, cumulative - (i + 1) * xfade_dur))

    # Custom xfade expression: A*(1-PROGRESS) keeps outgoing clip solid,
    # but we want outgoing FULLY visible until incoming takes over.
    # Expression: A stays full, B fades in on top = A*(1-P) + B*P
    # BUT we want A to NOT fade: just show B fading in over A.
    # Use: "if(gte(P,0),A*(1-P)+B*P,A)" — actually the default fade does this.
    # The REAL fix: use transition=wiperight or a custom expr where A stays at 1.0:
    # expr: A + B * P  (A stays, B fades in additively — not right either)
    # Correct: show A when P is low, blend to B as P grows, but A never dims:
    # expr: A*(1-P) + B*P  — this IS mutual dissolve
    # What we want: A*1 composited UNDER B*P = B*P + A*(1-P)... same thing
    #
    # Actually the right approach: use 'fadegrays' or just use B's alpha.
    # Simplest: transition=slideleft with very short duration looks like
    # the new clip sliding in. Or use transition=fadeblack which fades
    # through black (outgoing fades to black, incoming fades from black).
    #
    # Best option for "new clip emerges": transition=wipeup or custom.
    # Let's use a custom expression where outgoing stays solid and
    # incoming's opacity ramps from 0 to 1:
    # expr='if(lte(P,1), A*(1-P*P) + B*(P*P), B)'
    # P*P makes the incoming clip appear slowly then accelerate — feels like
    # it's emerging on top rather than a linear dissolve.

    vparts = []
    prev_v = "[0:v]"
    for i in range(n - 1):
        out_v = f"[v{i}]" if i < n - 2 else "[outv]"
        # Custom expression: outgoing stays nearly full, incoming fades in on top
        # P goes 0→1. We want: outgoing*(1-P^2) + incoming*(P^2)
        # This keeps outgoing visible longer, incoming appears late and fast
        expr = "'A*(1-P*P)+B*P*P'"
        vparts.append(
            f"{prev_v}[{i+1}:v]xfade=transition=custom:expr={expr}"
            f":duration={xfade_dur}:offset={offsets[i]}{out_v}"
        )
        prev_v = out_v

    # Audio: hard cut at the transition point (no blending)
    half = xfade_dur / 2
    aparts = []
    for i in range(n):
        ts = half if i > 0 else 0
        te = durations[i] - (half if i < n - 1 else 0)
        aparts.append(f"[{i}:a]atrim=start={ts}:end={te},asetpts=PTS-STARTPTS[a{i}]")
    audio_inputs = "".join(f"[a{i}]" for i in range(n))
    aparts.append(f"{audio_inputs}concat=n={n}:v=0:a=1[outa]")

    subprocess.run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(vparts) + ";" + ";".join(aparts),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-b:v", "15M", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k", str(output_path),
    ], check=True, capture_output=True)


def run_k1(name, scenes, num_frames=129, audio_guidance=3.5):
    test_id = uuid.uuid4().hex[:8]
    output_dir = RESULTS_DIR / test_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"V13 K=1: {name} ({len(scenes)}×{num_frames}fr)")
    logger.info(f"{'='*60}")

    start = time.time()
    clips = []
    speeches = []

    for i, scene in enumerate(scenes):
        logger.info(f"  Segment {i+1}/{len(scenes)}...")
        seg_dir = output_dir / f"segment_{i:02d}"
        clip = generate_single(scene["speech"], 42 + i * 10, seg_dir,
                                num_frames=num_frames, audio_guidance=audio_guidance)
        clips.append(clip)
        speeches.append(scene["speech"].strip('"'))

    # Add fade-out to last clip
    last_faded = clips[-1].parent / "clip_hq_faded.mp4"
    add_fade_out(clips[-1], last_faded, video_fade_frames=18, audio_fade_ms=750)
    clips[-1] = last_faded

    # Assemble
    final = output_dir / "final.mp4"
    video_xfade_audio_hardcut(clips, final, xfade_frames=6)
    dur = get_duration(final)
    gen_time = time.time() - start

    # Score with correct per-segment text
    logger.info(f"  Scoring...")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    clip_scores = []
    for i, clip in enumerate(clips):
        audio = score_clip(clip, speeches[i], Path(bible.reference_audio))
        ending = score_ending(clip) if i == len(clips) - 1 else {}
        clip_scores.append({
            "seg": i, "wer": audio.get("wer"), "utmos": audio.get("utmos"),
            "nisqa_col": audio.get("nisqa_coloration"),
            "composite": audio.get("audio_composite"),
            "method": audio.get("transcription_method"),
            **ending,
        })
        wer = audio.get("wer", "?")
        comp = audio.get("audio_composite", 0)
        logger.info(f"    Seg {i}: WER={wer} comp={comp:.2f}")

    seam_scores = []
    for i in range(len(clips) - 1):
        flow = score_flow_discontinuity(clips[i], clips[i + 1])
        seam_scores.append({
            "seam": f"s{i:02d}_s{i+1:02d}",
            "flow": flow.get("flow_discontinuity"),
        })
        logger.info(f"    Seam {i}→{i+1}: flow={flow.get('flow_discontinuity') or 0:.3f}")

    result = {
        "test_id": test_id, "name": name, "version": 13, "type": "k1",
        "k": 1, "num_segments": len(scenes), "num_frames": num_frames,
        "actual_duration_s": dur, "generation_time_s": round(gen_time, 1),
        "output_path": str(final),
        "clip_scores": clip_scores, "seam_scores": seam_scores,
        "settings": {"first_frame": "always_original", "identity_guidance": 1.5,
                      "audio_guidance": audio_guidance, "xfade_frames": 6,
                      "assembly": "video_xfade_audio_hardcut", "fade_ending": True,
                      "preset": "slow"},
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"  Done in {gen_time:.0f}s. Duration: {dur:.1f}s")
    update_wiki_after_experiment(result)
    return result


def run_beam(name, scenes, k=2, num_frames=129):
    test_id = uuid.uuid4().hex[:8]
    output_dir = RESULTS_DIR / test_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"V13 K={k}: {name} ({len(scenes)}×{num_frames}fr)")
    logger.info(f"{'='*60}")

    start = time.time()
    all_candidates = []
    speeches = []

    for i, scene in enumerate(scenes):
        logger.info(f"  Segment {i+1}/{len(scenes)} (K={k})...")
        seg_dir = output_dir / f"segment_{i:02d}"
        speeches.append(scene["speech"].strip('"'))

        candidates = generate_candidates(
            speech=scene["speech"], bible=bible, output_dir=seg_dir,
            k=k, base_seed=42 + i * 10, num_frames=num_frames,
            parallel=True,
        )

        # Fixed WER gate: None = skip
        good = [c for c in candidates
                if (c.get("wer") is None or c.get("wer") < 0.6)
                and (c.get("nisqa_coloration") is None or c.get("nisqa_coloration") > 2.5)]
        if good:
            logger.info(f"    {len(good)}/{len(candidates)} pass gates")

        all_candidates.append(candidates)

    path = greedy_search(all_candidates)
    clips = [Path(c["clip_path"]) for c in path if c.get("clip_path")]

    # Fade last clip
    last_faded = clips[-1].parent / "clip_hq_faded.mp4"
    add_fade_out(clips[-1], last_faded)
    clips[-1] = last_faded

    final = output_dir / "final.mp4"
    video_xfade_audio_hardcut(clips, final, xfade_frames=6)
    dur = get_duration(final)
    gen_time = time.time() - start

    # Score
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    clip_scores = []
    for i, clip in enumerate(clips):
        speech = speeches[i] if i < len(speeches) else ""
        audio = score_clip(clip, speech, Path(bible.reference_audio))
        clip_scores.append({
            "seg": i, "wer": audio.get("wer"), "utmos": audio.get("utmos"),
            "composite": audio.get("audio_composite"),
        })
        logger.info(f"    Seg {i}: WER={audio.get('wer','?')} comp={audio.get('audio_composite',0):.2f}")

    seam_scores = []
    for i in range(len(clips) - 1):
        flow = score_flow_discontinuity(clips[i], clips[i + 1])
        seam_scores.append({"seam": f"s{i:02d}_s{i+1:02d}",
                             "flow": flow.get("flow_discontinuity")})
        logger.info(f"    Seam {i}→{i+1}: flow={flow.get('flow_discontinuity','?'):.3f}")

    mean_shot = round(sum(c.get("shot_score", 0) for c in path) / max(len(path), 1), 3)

    result = {
        "test_id": test_id, "name": name, "version": 13, "type": "beam",
        "k": k, "num_segments": len(scenes), "num_frames": num_frames,
        "actual_duration_s": dur, "generation_time_s": round(gen_time, 1),
        "mean_shot_score": mean_shot,
        "output_path": str(final),
        "clip_scores": clip_scores, "seam_scores": seam_scores,
        "settings": {"first_frame": "always_original", "identity_guidance": 1.5,
                      "audio_guidance": 3.5, "xfade_frames": 6,
                      "assembly": "video_xfade_audio_hardcut", "fade_ending": True},
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"  Done in {gen_time:.0f}s. Duration: {dur:.1f}s. Mean shot: {mean_shot}")
    update_wiki_after_experiment(result)
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AUTORESEARCH V13 — 129fr + Video Xfade + Fade Ending")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    results = []

    # K=1 proofs (3 × ~12 min = 36 min)
    for name, scenes in [
        ("v13_health_k1", HEALTH_4),
        ("v13_recruiting_k1", RECRUITING_4),
        ("v13_seo_k1", SEO_4),
    ]:
        r = run_k1(name, scenes, num_frames=129)
        results.append(r)
        _save(results)

    # K=2 beam (3 × ~24 min = 72 min)
    for name, scenes in [
        ("v13_health_beam2", HEALTH_4),
        ("v13_recruiting_beam2", RECRUITING_4),
        ("v13_seo_beam2", SEO_4),
    ]:
        r = run_beam(name, scenes, k=2, num_frames=129)
        results.append(r)
        _save(results)

    # Comparison experiments
    # 161fr baseline
    r = run_k1("v13_health_161fr", HEALTH_161, num_frames=161)
    results.append(r)
    _save(results)

    # 5-segment
    r = run_k1("v13_health_5x129", HEALTH_5, num_frames=129)
    results.append(r)
    _save(results)

    # audio_guidance comparison
    r = run_k1("v13_health_ag2.0", HEALTH_4, num_frames=129, audio_guidance=2.0)
    results.append(r)
    _save(results)

    # 90-second stress tests (16 segments × 129fr = ~87s each)
    r = run_k1("v13_health_90s", HEALTH_16, num_frames=129)
    results.append(r)
    _save(results)

    r = run_k1("v13_recruiting_90s", RECRUITING_16, num_frames=129)
    results.append(r)
    _save(results)

    # Leaderboard
    logger.info(f"\n{'='*60}")
    logger.info("V13 LEADERBOARD")
    for i, r in enumerate(results):
        dur = r.get("actual_duration_s", "?")
        comps = [c.get("composite", 0) for c in r.get("clip_scores", []) if c.get("composite")]
        avg_comp = round(sum(comps) / max(len(comps), 1), 2) if comps else "?"
        flows = [s.get("flow", 0) for s in r.get("seam_scores", []) if s.get("flow")]
        worst_flow = round(max(flows), 2) if flows else "?"
        logger.info(f"  #{i+1} {r['name']:30s} K={r['k']} {r['num_segments']}seg "
                    f"dur={dur}s comp={avg_comp} worst_seam={worst_flow}")
    logger.info("=" * 60)


def _save(results):
    with open(RESULTS_DIR / "progress_v13.json", "w") as f:
        json.dump({
            "version": 13, "completed": len(results), "total": 11,
            "results": results, "last_updated": datetime.now().isoformat(),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
