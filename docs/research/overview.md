---
title: UGC Factory Research Overview
type: overview
created: 2026-04-05
updated: 2026-04-06
---

# UGC Factory Research Overview

## Current Best Pipeline (v4, updated 2026-04-06)

**ID-LoRA `inference_two_stage_hq.py`** with these settings:

```json
{
  "model": "ltx-2.3-22b-dev.safetensors",
  "precision": "bf16 (NO quantization)",
  "id_lora": "id-lora-talkvid",
  "identity_guidance": 1.5,
  "audio_guidance": 2.0,
  "video_guidance": 3.0,
  "video_rescale": 0.45,
  "distilled_s1": 0.1,
  "distilled_s2": 0.3,
  "height": 864, "width": 480,
  "output_resolution": "960x1728",
  "num_frames": 161,
  "fps": 25,
  "reference": "real photograph (NOT AI-generated)",
  "sounds_tag": "confident authoritative male speaking voice, measured pace, clear enunciation, professional studio quality"
}
```

## Experiments Completed: 51 total
- V1: 22 experiments (visual-only scoring)
- V2: 15 experiments (audio scoring added)
- V3: 14 experiments (advanced [SOUNDS], anti-wiriness)
- V4: 15 experiments (audio+visual combined scoring)

## Key Findings (ranked by evidence strength)

### Confirmed (multiple experiments)
1. **Full precision bf16 >> int8 quantization** — most impactful single change
2. **Real photos >> AI references** — eliminates AI sheen
3. **ID-LoRA pipeline >> ComfyUI native** — identity + audio guidance critical
4. **Audio guidance 2.0 is optimal** — balances audio quality (7.52) + perfect WER + good visuals
5. **Descriptive [SOUNDS] tags: MOS 3.3 vs 2.0** — massive impact on audio quality
6. **Identity guidance 1.5 optimal** — best teeth, natural expressions
7. **LoRA 0.1/0.3 >> 0.25/0.5** — more natural skin texture
8. **Video rescale has ZERO effect on audio** — purely visual parameter
9. **Speech accuracy (WER) depends on prompt length** — <20 words = 0.000, >30 words = 0.29

### Visual quality metrics (from InsightFace scoring)
10. **Face consistency: 0.85-0.91 for single clips, drops to 0.77-0.80 for multi-clip**
11. **Identity match: 0.83-0.91** (vs reference photo)
12. **Longer speech = better visual quality but worse audio accuracy** (tradeoff)
13. **Sharpness varies 24-64** — longer prompts with more content = sharper frames

### Audio-visual tradeoff
14. **Low audio guidance (0.5-1.5) → best visuals, slightly worse lip-sync**
15. **High audio guidance (7.0) → best lip-sync, slightly worse visual quality**
16. **Sweet spot: audio guidance 2.0** (combined score 6.97)

## Next Frontier: OmniVoice + LTX 2.3

**2-step workflow being tested:**
1. Generate high-quality cloned-voice audio with OmniVoice (zero-shot TTS)
2. Feed that audio into LTX 2.3's audio-conditioned video pipeline

**Why:** OmniVoice gives exact voice control (cloning, 646 languages, non-verbal expressions) while LTX 2.3 provides native lip-sync from that audio. Best of both worlds.

**Status:** ComfyUI nodes available, model is ~4GB, fits on GPU:1. Testing in V5.
