#!/usr/bin/env python3
"""Autoresearch overnight runner — parameter optimization for UGC video quality.

Generates clips across a parameter grid, extracts frames + audio metrics,
and saves everything for scoring. Run with Claude to score and rank.

Usage: python run_autoresearch.py
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dataclasses import asdict
from factory.autoresearch import (
    TestConfig, TestResult, generate_test_clip, generate_multi_clip,
    extract_evaluation_frames, analyze_audio, save_leaderboard, RESULTS_DIR,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "autoresearch.log"),
    ],
)
logger = logging.getLogger("autoresearch")

# === REFERENCES ===
# Point these at your own persona (both required). The original runs used the
# author's photo and a 48 kHz voice reference; neither ships with this repo.
REF_IMAGE = os.environ.get("UGC_REF_IMAGE", "")
REF_AUDIO = os.environ.get("UGC_REF_AUDIO", "")
STYLE_PROMPT = os.environ.get(
    "UGC_STYLE_PROMPT",
    "Head-and-shoulders close-up of a person speaking directly to camera, "
    "warm natural window lighting, fixed camera, clean composition, "
    "subtle eye blinks, slight natural head motion.",
)
if not REF_IMAGE or not REF_AUDIO:
    sys.exit("Set UGC_REF_IMAGE (portrait photo) and UGC_REF_AUDIO (48 kHz voice reference) first.")

# === PLACEHOLDER SCRIPTS (recruiting vertical) ===
# Brand-free stand-ins for the product scripts used in the original runs.
RECRUITING_SCENES = {
    "30s": [
        {
            "visual": STYLE_PROMPT,
            "speech": "If you are a hiring manager still screening every application by hand, you are losing hours every single week to work a machine does better.",
        },
        {
            "visual": STYLE_PROMPT,
            "speech": "Modern sourcing tools pull candidates from public profiles, screen them against your requirements, and rank them by fit before you read a single one.",
        },
        {
            "visual": STYLE_PROMPT,
            "speech": "Teams that adopt this cut their time to hire dramatically, so try a sourcing tool on your next open role and see the difference for yourself.",
        },
    ],
    "60s": [
        {
            "visual": STYLE_PROMPT,
            "speech": "If you are a hiring manager still screening applications by hand you are wasting hours every week. Let me show you what changed for our team.",
        },
        {
            "visual": STYLE_PROMPT,
            "speech": "Automated sourcing pulls candidates from public profiles, screens them against your exact requirements, and ranks them by fit score. No more guessing.",
        },
        {
            "visual": STYLE_PROMPT,
            "speech": "But here is what really sets it apart. It drafts personalized outreach for each candidate. The response rates are far better than generic templates.",
        },
        {
            "visual": STYLE_PROMPT,
            "speech": "One team went from taking three weeks to fill a role down to five days. Their cost per hire dropped sharply. Those are the numbers that matter.",
        },
        {
            "visual": STYLE_PROMPT,
            "speech": "Most tools offer a free trial with a couple of open roles included, so you can test it on a real search before you commit to anything.",
        },
        {
            "visual": STYLE_PROMPT,
            "speech": "If you are tired of the old way of recruiting, give automated sourcing a fair shot. I promise you will not go back. See you in the next one!",
        },
    ],
}

# === EXPERIMENT GRID ===
def build_experiments() -> list[dict]:
    """Build the full experiment grid."""
    experiments = []

    # --- SINGLE CLIP PARAMETER SWEEP (5s each, fast iteration) ---

    # Baseline: current best (Test C settings)
    experiments.append({
        "config": TestConfig(
            name="baseline_best",
            description="Current best: full precision, 480x864, id_guidance 2.0, 161 frames",
            height=864, width=480, num_frames=161,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            prompt_visual=STYLE_PROMPT,
            prompt_speech="Hey everyone welcome back. Today I want to talk about something that has completely changed how I think about recruiting and hiring.",
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
        ),
        "type": "single",
    })

    # Identity guidance sweep
    for ig in [1.0, 1.5, 3.0]:
        experiments.append({
            "config": TestConfig(
                name=f"id_guidance_{ig}",
                description=f"Identity guidance {ig} (baseline is 2.0)",
                height=864, width=480, num_frames=161,
                identity_guidance=ig, distilled_s1=0.25, distilled_s2=0.5,
                prompt_visual=STYLE_PROMPT,
                prompt_speech="Hey everyone welcome back. Today I want to talk about something that has completely changed how I think about recruiting and hiring.",
                first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            ),
            "type": "single",
        })

    # LoRA strength sweep
    for s1, s2 in [(0.1, 0.3), (0.5, 0.6)]:
        experiments.append({
            "config": TestConfig(
                name=f"lora_{s1}_{s2}",
                description=f"Distilled LoRA s1={s1}, s2={s2}",
                height=864, width=480, num_frames=161,
                identity_guidance=2.0, distilled_s1=s1, distilled_s2=s2,
                prompt_visual=STYLE_PROMPT,
                prompt_speech="Hey everyone welcome back. Today I want to talk about something that has completely changed how I think about recruiting and hiring.",
                first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            ),
            "type": "single",
        })

    # Resolution sweep
    for h, w in [(576, 320), (704, 384)]:
        experiments.append({
            "config": TestConfig(
                name=f"res_{h}x{w}",
                description=f"Base resolution {h}x{w} → {h*2}x{w*2}",
                height=h, width=w, num_frames=161,
                identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
                prompt_visual=STYLE_PROMPT,
                prompt_speech="Hey everyone welcome back. Today I want to talk about something that has completely changed how I think about recruiting and hiring.",
                first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            ),
            "type": "single",
        })

    # Frame count / clip length
    for nf in [121, 201]:
        experiments.append({
            "config": TestConfig(
                name=f"frames_{nf}",
                description=f"{nf} frames = {nf/24:.1f}s at 24fps",
                height=864, width=480, num_frames=nf,
                identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
                prompt_visual=STYLE_PROMPT,
                prompt_speech="Hey everyone welcome back. Today I want to talk about something that has completely changed how I think about recruiting and hiring.",
                first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            ),
            "type": "single",
        })

    # Audio guidance sweep
    for ag in [3.0, 5.0]:
        experiments.append({
            "config": TestConfig(
                name=f"audio_guidance_{ag}",
                description=f"Audio guidance {ag} (baseline is 7.0)",
                height=864, width=480, num_frames=161,
                identity_guidance=2.0, audio_guidance=ag,
                distilled_s1=0.25, distilled_s2=0.5,
                prompt_visual=STYLE_PROMPT,
                prompt_speech="Hey everyone welcome back. Today I want to talk about something that has completely changed how I think about recruiting and hiring.",
                first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            ),
            "type": "single",
        })

    # --- MULTI-CLIP TESTS (recruiting content) ---

    # 30s recruiting ad (3 clips x ~6.7s with hard cuts)
    experiments.append({
        "config": TestConfig(
            name="recruiting_30s",
            description="30s recruiting ad, 3 clips, hard cuts, full precision",
            height=864, width=480, num_frames=161,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            prompt_sounds="close-mic natural speech, quiet room, no music",
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            num_clips=3, clip_assembly="hardcut",
        ),
        "type": "multi",
        "scenes": RECRUITING_SCENES["30s"],
    })

    # 60s recruiting ad (6 clips x ~6.7s with hard cuts)
    experiments.append({
        "config": TestConfig(
            name="recruiting_60s",
            description="60s recruiting ad, 6 clips, hard cuts, full precision",
            height=864, width=480, num_frames=161,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            prompt_sounds="close-mic natural speech, quiet room, no music",
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            num_clips=6, clip_assembly="hardcut",
        ),
        "type": "multi",
        "scenes": RECRUITING_SCENES["60s"],
    })

    return experiments


def run_experiment(exp: dict) -> dict:
    """Run a single experiment and collect metrics."""
    cfg = exp["config"]
    exp_type = exp["type"]

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {cfg.name}")
    logger.info(f"Description: {cfg.description}")
    logger.info(f"{'='*60}")

    result = {
        "test_id": cfg.test_id,
        "name": cfg.name,
        "description": cfg.description,
        "config": asdict(cfg),
        "type": exp_type,
        "status": "pending",
    }

    try:
        start = time.time()

        if exp_type == "single":
            video_path = generate_test_clip(cfg)
        else:
            video_path = generate_multi_clip(cfg, exp["scenes"])

        gen_time = time.time() - start

        # Extract frames for evaluation
        eval_dir = RESULTS_DIR / cfg.test_id / "eval"
        frames = extract_evaluation_frames(video_path, eval_dir, count=8)

        # Analyze audio
        audio_metrics = analyze_audio(video_path, eval_dir)

        # Get video metadata
        duration = float(subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip())

        res_info = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split(",")

        result.update({
            "status": "completed",
            "output_path": str(video_path),
            "frames": [str(f) for f in frames],
            "audio_metrics": audio_metrics,
            "duration_s": duration,
            "resolution": f"{res_info[0]}x{res_info[1]}",
            "generation_time_s": round(gen_time, 1),
        })

        logger.info(f"[{cfg.name}] Completed in {gen_time:.0f}s, {duration:.1f}s video, {res_info[0]}x{res_info[1]}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"[{cfg.name}] Failed: {e}")

    # Save individual result
    result_path = RESULTS_DIR / cfg.test_id / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("UGC AUTORESEARCH — Overnight Parameter Optimization")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*60)

    experiments = build_experiments()
    logger.info(f"Total experiments: {len(experiments)}")

    # Estimate time
    single_count = sum(1 for e in experiments if e["type"] == "single")
    multi_clips = sum(e["config"].num_clips for e in experiments if e["type"] == "multi")
    total_clips = single_count + multi_clips
    est_minutes = total_clips * 15  # ~15 min per clip at full precision
    logger.info(f"Single clips: {single_count}, Multi-clip total: {multi_clips}")
    logger.info(f"Estimated time: ~{est_minutes} minutes ({est_minutes/60:.1f} hours)")

    results = []
    for i, exp in enumerate(experiments):
        logger.info(f"\n--- Experiment {i+1}/{len(experiments)} ---")
        result = run_experiment(exp)
        results.append(result)

        # Save progress after each experiment
        progress_path = RESULTS_DIR / "progress.json"
        with open(progress_path, "w") as f:
            json.dump({
                "completed": i + 1,
                "total": len(experiments),
                "results": results,
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)

    logger.info("\n" + "="*60)
    logger.info("AUTORESEARCH COMPLETE")
    logger.info(f"Finished: {datetime.now().isoformat()}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Successful: {sum(1 for r in results if r['status'] == 'completed')}")
    logger.info(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info("="*60)
    logger.info("\nRun Claude to score the results and generate the leaderboard.")


if __name__ == "__main__":
    main()
