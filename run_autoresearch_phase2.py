#!/usr/bin/env python3
"""Autoresearch Phase 2 — additional experiments after Phase 1 completes.

More multi-clip tests, content variations, and parameter combos.
Run after run_autoresearch.py finishes.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from factory.autoresearch import (
    TestConfig, generate_test_clip, generate_multi_clip,
    extract_evaluation_frames, analyze_audio, RESULTS_DIR,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(RESULTS_DIR / "autoresearch_phase2.log"),
    ],
)
logger = logging.getLogger("autoresearch-p2")

# Point these at your own persona (both required); see run_autoresearch.py.
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

# === PLACEHOLDER SCRIPTS (health vertical) ===
# Brand-free stand-ins for the product scripts used in the original runs.
HEALTH_30S = [
    {
        "visual": STYLE_PROMPT,
        "speech": "Stop what you are doing. If you take supplements and you are not checking the evidence first, you could be wasting your money or worse. There is a better way.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "Free evidence based research platforms now cover hundreds of supplements with real clinical data. Not marketing hype, actual science you can read yourself.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "Look up the clinical evidence before your next purchase. It is completely free. Follow for more evidence based health tips.",
    },
]

HEALTH_60S = [
    {
        "visual": STYLE_PROMPT,
        "speech": "Raise your hand if you have ever bought a supplement because you saw it on social media. Yeah me too. But here is the problem. Most supplement advice online is just marketing.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "That is why I started reading the actual clinical evidence behind every supplement. Hundreds of supplements, dozens of peptides, all sourced from real studies.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "Want to know if vitamin D actually helps joint pain? Or if ashwagandha really reduces cortisol? The real data comes with confidence scores based on study quality.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "Some platforms even have an AI research assistant you can chat with. Ask it anything about supplements and it pulls from the evidence database to give a straight answer.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "The best part is the good ones are completely free. No subscription, no paywalls. Just real evidence based research you can trust. Go check the evidence right now.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "Follow this account for more evidence based health content. And if you found this useful, share it with someone who needs to hear it. See you in the next one.",
    },
]

# === PLACEHOLDER SCRIPTS (SEO vertical) ===
SEO_30S = [
    {
        "visual": STYLE_PROMPT,
        "speech": "If you run a small business and you are not showing up in AI search results, you are invisible to half your potential customers. That is the reality this year.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "AI visibility tools monitor how the big assistants describe your business and tell you exactly what to fix to get found by them.",
    },
    {
        "visual": STYLE_PROMPT,
        "speech": "Run a free audit on your own site this week. It takes about five minutes to see your first results. Follow for more search tips.",
    },
]

def build_phase2_experiments() -> list[dict]:
    experiments = []

    # --- BEST SETTINGS COMBO (combine winners from phase 1) ---
    experiments.append({
        "config": TestConfig(
            name="combo_best_predicted",
            description="Predicted best: id_guidance 2.0, full prec, 480x864, 161 frames, LoRA 0.25/0.5",
            height=864, width=480, num_frames=161,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            video_guidance=3.0, audio_guidance=7.0, video_rescale=0.45,
            prompt_visual=STYLE_PROMPT,
            prompt_speech="The single most important thing I have learned about health this year is that most of us are not getting enough magnesium. Let me explain why that matters.",
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
        ),
        "type": "single",
    })

    # --- LOWER IDENTITY GUIDANCE TEST ---
    experiments.append({
        "config": TestConfig(
            name="id_guidance_0.5",
            description="Very low identity guidance 0.5 — maximum face smoothness",
            height=864, width=480, num_frames=161,
            identity_guidance=0.5, distilled_s1=0.25, distilled_s2=0.5,
            prompt_visual=STYLE_PROMPT,
            prompt_speech="Hey everyone welcome back. Today I want to share three simple changes that improved my sleep quality dramatically.",
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
        ),
        "type": "single",
    })

    # --- VIDEO RESCALE SWEEP ---
    for vr in [0.3, 0.6]:
        experiments.append({
            "config": TestConfig(
                name=f"video_rescale_{vr}",
                description=f"Video rescale {vr} (baseline 0.45)",
                height=864, width=480, num_frames=161,
                identity_guidance=2.0, video_rescale=vr,
                distilled_s1=0.25, distilled_s2=0.5,
                prompt_visual=STYLE_PROMPT,
                prompt_speech="Hey everyone welcome back. Today I want to share three simple changes that improved my sleep quality dramatically.",
                first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            ),
            "type": "single",
        })

    # --- HEALTH MULTI-CLIP ---
    experiments.append({
        "config": TestConfig(
            name="health_30s",
            description="30s health ad, 3 clips hard cut",
            height=864, width=480, num_frames=161,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            num_clips=3, clip_assembly="hardcut",
        ),
        "type": "multi",
        "scenes": HEALTH_30S,
    })

    experiments.append({
        "config": TestConfig(
            name="health_60s",
            description="60s health ad, 6 clips hard cut",
            height=864, width=480, num_frames=161,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            num_clips=6, clip_assembly="hardcut",
        ),
        "type": "multi",
        "scenes": HEALTH_60S,
    })

    # --- SEO MULTI-CLIP ---
    experiments.append({
        "config": TestConfig(
            name="seo_30s",
            description="30s SEO ad, 3 clips hard cut",
            height=864, width=480, num_frames=161,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            num_clips=3, clip_assembly="hardcut",
        ),
        "type": "multi",
        "scenes": SEO_30S,
    })

    # --- LONGER CLIP TEST (201 frames = 8.4s) for fewer seams ---
    experiments.append({
        "config": TestConfig(
            name="recruiting_30s_longer_clips",
            description="30s recruiting ad with 201-frame clips (8.4s each, only 2 clips needed + 1 shorter)",
            height=864, width=480, num_frames=201,
            identity_guidance=2.0, distilled_s1=0.25, distilled_s2=0.5,
            first_frame=REF_IMAGE, reference_audio=REF_AUDIO,
            num_clips=2, clip_assembly="hardcut",
        ),
        "type": "multi",
        "scenes": [
            {
                "visual": STYLE_PROMPT,
                "speech": "If you are still manually screening applications you are wasting hours every week. Automated sourcing finds the best candidates in minutes. It pulls from public profiles, screens against your requirements, and ranks by fit.",
            },
            {
                "visual": STYLE_PROMPT,
                "speech": "Teams using automated sourcing cut their time to hire dramatically. Most tools offer a free trial with a few open roles included. Follow for more hiring tips.",
            },
        ],
    })

    return experiments


def run_experiment(exp):
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

        eval_dir = RESULTS_DIR / cfg.test_id / "eval"
        frames = extract_evaluation_frames(video_path, eval_dir, count=8)
        audio_metrics = analyze_audio(video_path, eval_dir)

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

        logger.info(f"[{cfg.name}] Completed in {gen_time:.0f}s")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"[{cfg.name}] Failed: {e}")

    result_path = RESULTS_DIR / cfg.test_id / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    # Wait for phase 1 to finish
    phase1_progress = RESULTS_DIR / "progress.json"
    if phase1_progress.exists():
        p = json.load(open(phase1_progress))
        if p["completed"] < p["total"]:
            logger.info(f"Phase 1 still running ({p['completed']}/{p['total']}). Waiting...")
            while True:
                time.sleep(60)
                p = json.load(open(phase1_progress))
                if p["completed"] >= p["total"]:
                    break
                logger.info(f"Phase 1: {p['completed']}/{p['total']} done, waiting...")

    logger.info("="*60)
    logger.info("AUTORESEARCH PHASE 2 — Additional Experiments")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*60)

    experiments = build_phase2_experiments()
    logger.info(f"Phase 2 experiments: {len(experiments)}")

    results = []
    for i, exp in enumerate(experiments):
        logger.info(f"\n--- Phase 2 Experiment {i+1}/{len(experiments)} ---")
        result = run_experiment(exp)
        results.append(result)

        progress_path = RESULTS_DIR / "progress_phase2.json"
        with open(progress_path, "w") as f:
            json.dump({
                "phase": 2,
                "completed": i + 1,
                "total": len(experiments),
                "results": results,
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)

    logger.info(f"\nPhase 2 complete: {len(results)} experiments")
    logger.info(f"Successful: {sum(1 for r in results if r['status'] == 'completed')}")


if __name__ == "__main__":
    main()
