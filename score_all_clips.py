#!/usr/bin/env python3
"""Score all existing clips with NISQA + ECAPA.

Dumps per-clip NISQA decomposition and ECAPA speaker embeddings,
plus cross-clip consistency for multi-clip experiments.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from factory.score_av import score_nisqa, get_ecapa_embedding, score_speaker_consistency

import torch
import soundfile as sf

OUTPUTS = Path(__file__).parent / "outputs"
REF_AUDIO = Path(os.environ.get("UGC_REF_AUDIO", Path(__file__).parent / "personas" / "default" / "reference.wav"))


def extract_audio(video_path: Path, sr: int = 48000) -> Path:
    """Extract audio as mono WAV at given sample rate."""
    out = video_path.parent / "_score_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", str(out),
    ], check=True, capture_output=True)
    return out


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def score_experiment(exp_dir: Path) -> dict:
    """Score all clips in an experiment directory."""
    exp_id = exp_dir.name

    # Find clips
    clips = sorted(exp_dir.glob("video/scene_*/clip_*.mp4"))
    final = list(exp_dir.glob("*_final.mp4"))

    if not clips:
        return None

    result = {
        "experiment": exp_id,
        "num_clips": len(clips),
        "has_final": bool(final),
        "clips": [],
    }

    ecapa_audio_paths = []  # 16kHz for ECAPA

    # Get reference embedding (needs 16kHz)
    ref_emb = None
    if REF_AUDIO.exists():
        ref_16k = REF_AUDIO.parent / "_ref_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(REF_AUDIO),
            "-ar", "16000", "-ac", "1", str(ref_16k),
        ], check=True, capture_output=True)
        ref_emb = get_ecapa_embedding(ref_16k)
        ref_16k.unlink(missing_ok=True)

    for clip_path in clips:
        scene = clip_path.parent.name
        print(f"  Scoring {scene}/{clip_path.name}...", end=" ", flush=True)

        try:
            dur = get_duration(clip_path)

            # NISQA needs 48kHz
            audio_48k = extract_audio(clip_path, sr=48000)
            nisqa = score_nisqa(audio_48k)
            audio_48k.unlink(missing_ok=True)

            # ECAPA needs 16kHz
            audio_16k = clip_path.parent / "_score_audio_16k.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(clip_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_16k),
            ], check=True, capture_output=True)
            ecapa_audio_paths.append(audio_16k)

            clip_emb = get_ecapa_embedding(audio_16k)
            speaker_cos = None
            if clip_emb is not None and ref_emb is not None:
                speaker_cos = round(torch.nn.functional.cosine_similarity(
                    clip_emb.unsqueeze(0), ref_emb.unsqueeze(0)
                ).item(), 4)

            nisqa_mos = nisqa.get('nisqa_mos')
            mos_str = f"{nisqa_mos:.3f}" if nisqa_mos is not None else "FAIL"

            clip_result = {
                "scene": scene,
                "file": clip_path.name,
                "duration_s": round(dur, 2),
                "speaker_cos_ref": speaker_cos,
                **nisqa,
            }
            result["clips"].append(clip_result)
            print(f"NISQA={mos_str} ECAPA={speaker_cos}")
        except Exception as e:
            print(f"ERROR: {e}")
            result["clips"].append({"scene": scene, "error": str(e)})

    # Cross-clip consistency
    if len(ecapa_audio_paths) >= 2:
        print(f"  Cross-clip consistency ({len(ecapa_audio_paths)} clips)...", flush=True)
        consistency = score_speaker_consistency(ecapa_audio_paths)
        result["cross_clip_consistency"] = consistency

    # Cleanup temp audio files
    for ap in ecapa_audio_paths:
        ap.unlink(missing_ok=True)

    return result


def main():
    print("=" * 70)
    print("SCORING ALL CLIPS: NISQA + ECAPA")
    print("=" * 70)

    all_results = []

    exp_dirs = sorted(OUTPUTS.iterdir())
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue

        clips = list(exp_dir.glob("video/scene_*/clip_*.mp4"))
        if not clips:
            print(f"\n{exp_dir.name}: no clips, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Experiment: {exp_dir.name} ({len(clips)} clips)")
        print(f"{'='*50}")

        result = score_experiment(exp_dir)
        if result:
            all_results.append(result)

    # Save full results
    out_path = OUTPUTS.parent / "nisqa_ecapa_scores.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Experiment':<14} {'Scene':<12} {'Dur':>5} {'NISQA':>6} {'Noise':>6} {'Disc':>6} {'Color':>6} {'Loud':>6} {'ECAPA':>7}")
    print("-" * 120)

    for r in all_results:
        for c in r["clips"]:
            if "error" in c:
                print(f"{r['experiment']:<14} {c.get('scene','?'):<12} {'ERR':>5}")
                continue

            def fmt(v, w=6, d=3):
                return f"{v:{w}.{d}f}" if v is not None else f"{'N/A':>{w}}"

            print(f"{r['experiment']:<14} {c['scene']:<12} "
                  f"{c.get('duration_s', 0):>5.1f} "
                  f"{fmt(c.get('nisqa_mos'))} "
                  f"{fmt(c.get('nisqa_noisiness'))} "
                  f"{fmt(c.get('nisqa_discontinuity'))} "
                  f"{fmt(c.get('nisqa_coloration'))} "
                  f"{fmt(c.get('nisqa_loudness'))} "
                  f"{fmt(c.get('speaker_cos_ref'), 7, 4)}")

        if r.get("cross_clip_consistency"):
            cc = r["cross_clip_consistency"]
            print(f"  └── Consistency: centroid_mean={cc.get('centroid_mean', '?')} "
                  f"spread={cc.get('centroid_spread', '?')} "
                  f"pairwise_min={cc.get('pairwise_min', '?')}")

    print("=" * 120)


if __name__ == "__main__":
    main()
