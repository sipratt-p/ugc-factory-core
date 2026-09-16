#!/usr/bin/env python3
"""Re-score V9-V12 clips with the whisper server + full fleet.

Now that Whisper actually works, we'll get real WER scores and accurate
composite ratings. Also runs DINOv2, RAFT flow at seams, LPIPS, lip sync.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from factory import config

BASE = config.AUTORESEARCH_DIR  # env UGC_RESULTS_DIR

# All multi-clip experiments V9-V12
EXPERIMENTS = {}
for f in sorted(BASE.glob("*/result.json")):
    try:
        d = json.load(open(f))
        name = d.get("name", "")
        version = d.get("version", 0)
        if version >= 9 or "v9" in name or "v10" in name or "v11" in name or "v12" in name:
            test_id = d["test_id"]
            n_segs = d.get("num_segments", 0)
            if n_segs >= 3:
                EXPERIMENTS[test_id] = name
    except Exception:
        pass

print(f"Found {len(EXPERIMENTS)} experiments to re-score")
for tid, name in sorted(EXPERIMENTS.items(), key=lambda x: x[1]):
    print(f"  {tid}: {name}")


def find_winning_clips(exp_dir):
    result_file = exp_dir / "result.json"
    clips = []
    speeches = []

    if result_file.exists():
        rdata = json.load(open(result_file))
        path_info = rdata.get("assembly", {}).get("path", [])
        # Try to get speech from scenes
        for p in path_info:
            seg = p.get("segment", 0)
            cand = p.get("candidate_index", 0)
            clip_path = exp_dir / f"segment_{seg:02d}" / f"candidate_{cand}" / "clip_hq.mp4"
            if clip_path.exists():
                clips.append(clip_path)

    if not clips:
        segs = sorted(exp_dir.glob("segment_*"))
        for seg in segs:
            hq = seg / "candidate_0" / "clip_hq.mp4"
            if not hq.exists():
                hqs = list(seg.glob("candidate_*/clip_hq.mp4"))
                hq = hqs[0] if hqs else None
            if not hq:
                hq = seg / "clip_hq.mp4"
            if hq and hq.exists():
                clips.append(hq)

    return clips


def extract_frames(clip_path, out_dir, count=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
        capture_output=True, text=True, check=True).stdout.strip())
    frames = []
    for i in range(count):
        t = dur * i / max(count - 1, 1)
        fp = out_dir / f"f{i:02d}.png"
        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(clip_path),
                        "-frames:v", "1", "-q:v", "1", str(fp)],
                       check=True, capture_output=True)
        frames.append(fp)
    return frames, dur


def score_clip_full(clip_path, target_speech, reference_audio, label):
    """Score one clip with the whisper server + full fleet."""
    print(f"    {label}...", end=" ", flush=True)
    work = clip_path.parent / "_rescore"
    frames, dur = extract_frames(clip_path, work, count=6)

    r = {"clip": label, "dur": round(dur, 2)}

    # Audio scoring via score_clip (uses WHISPER_URL when reachable)
    try:
        from factory.score_av import score_clip
        audio = score_clip(clip_path, target_speech, Path(reference_audio) if reference_audio else None)
        r["wer"] = audio.get("wer")
        r["transcript"] = audio.get("transcript", "")[:100]
        r["method"] = audio.get("transcription_method", "unknown")
        r["utmos"] = audio.get("utmos")
        r["nisqa_mos"] = audio.get("nisqa_mos")
        r["nisqa_col"] = audio.get("nisqa_coloration")
        r["speaker_cos"] = audio.get("speaker_cos")
        r["composite"] = audio.get("audio_composite")
    except Exception as e:
        r["audio_error"] = str(e)[:80]

    # DINOv2
    try:
        from factory.score_advanced import score_dino_drift
        r.update(score_dino_drift(frames))
    except Exception as e:
        r["dino_error"] = str(e)[:60]

    # LPIPS temporal
    try:
        from factory.score_advanced import score_lpips_temporal
        r.update(score_lpips_temporal(frames))
    except Exception as e:
        r["lpips_error"] = str(e)[:60]

    # Lip sync
    try:
        from factory.score_seam import score_lip_sync
        r.update(score_lip_sync(clip_path))
    except Exception as e:
        r["sync_error"] = str(e)[:60]

    # TOPIQ-face
    try:
        import pyiqa, torch
        from PIL import Image
        import torchvision.transforms as T
        mid = frames[len(frames) // 2]
        img = Image.open(mid).convert("RGB")
        t = T.ToTensor()(img).unsqueeze(0).cuda()
        r["topiq_face"] = round(float(pyiqa.create_metric("topiq_nr-face", device="cuda")(t).item()), 4)
    except Exception:
        pass

    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print(f"WER={r.get('wer','?')} UTMOS={r.get('utmos','?')} DINO={r.get('dino_consistency','?')}")
    return r


def score_seam_full(clip_a, clip_b, label):
    r = {"seam": label}
    try:
        from factory.score_seam import score_seam
        seam = score_seam(clip_a, clip_b)
        r.update(seam)
    except Exception as e:
        r["seam_error"] = str(e)[:60]

    try:
        from factory.score_advanced import score_flow_discontinuity
        flow = score_flow_discontinuity(clip_a, clip_b)
        r.update(flow)
    except Exception as e:
        r["flow_error"] = str(e)[:60]

    return r


# Target speech for WER. The original runs looked this up per product; here it
# comes from the environment (one line applied to every segment).
DEFAULT_SPEECH = os.environ.get("UGC_RESCORE_SPEECH", "Hello, this is a test.")

# Speaker reference for ECAPA similarity (48 kHz WAV); empty disables it.
REF_AUDIO = os.environ.get("UGC_REF_AUDIO", "")


def guess_speech(name):
    return DEFAULT_SPEECH


def main():
    all_results = []

    for test_id, name in sorted(EXPERIMENTS.items(), key=lambda x: x[1]):
        exp_dir = BASE / test_id
        clips = find_winning_clips(exp_dir)
        if len(clips) < 2:
            print(f"\n{name}: only {len(clips)} clips, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"{name} ({test_id}) — {len(clips)} clips")
        print(f"{'='*50}")

        speech = guess_speech(name)
        exp = {"experiment": name, "test_id": test_id, "clips": [], "seams": []}

        for i, clip in enumerate(clips):
            r = score_clip_full(clip, speech, REF_AUDIO, f"seg_{i}")
            exp["clips"].append(r)

        for i in range(len(clips) - 1):
            r = score_seam_full(clips[i], clips[i + 1], f"s{i:02d}_s{i+1:02d}")
            exp["seams"].append(r)
            fd = r.get("flow_discontinuity")
            ss = r.get("seam_score")
            print(f"    Seam {i}→{i+1}: flow={fd} seam_score={ss} "
                  f"lpips={r.get('lpips_delta')} pose={r.get('head_pose_delta')}")

        all_results.append(exp)

    # Save
    out = BASE / "rescore_v9_v12.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Summary table
    print(f"\n{'='*140}")
    print(f"{'Experiment':<30} {'Seg':>4} {'WER':>5} {'UTMOS':>6} {'NISQA':>6} {'DINO':>6} "
          f"{'LipSync':>7} {'TOPIQ':>6} {'Composite':>9}")
    print("-" * 140)

    for exp in all_results:
        for c in exp["clips"]:
            def f(v, w=6, d=3):
                return f"{v:{w}.{d}f}" if v is not None else f"{'?':>{w}}"
            print(f"{exp['experiment']:<30} {c['clip']:>4} "
                  f"{f(c.get('wer'),5,3)} {f(c.get('utmos'))} {f(c.get('nisqa_mos'))} "
                  f"{f(c.get('dino_consistency'))} {f(c.get('lip_sync_corr'),7,3)} "
                  f"{f(c.get('topiq_face'))} {f(c.get('composite'),9,2)}")
        for s in exp.get("seams", []):
            fd = s.get("flow_discontinuity")
            ss = s.get("seam_score")
            lp = s.get("lpips_delta")
            hp = s.get("head_pose_delta")
            if fd is not None:
                print(f"{'':>30} {s['seam']:>8}  flow={fd:.3f} seam={ss} lpips={lp} pose={hp}")
        print()

    print("=" * 140)


if __name__ == "__main__":
    main()
