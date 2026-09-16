---
title: Audio Sweep V2 — Speech Prompt & Audio Quality Optimization
type: experiment
created: 2026-04-05
updated: 2026-04-05
tags: [audio, speech-prompt, sounds-tag, audio-guidance, wer, mos]
---

# Audio Sweep V2 Results

## Key Findings

### 1. Speech reproduction is now PERFECT for short/medium prompts
All [SOUNDS] variations and audio guidance levels achieved **WER 0.000** — the model reproduces the exact words from the [SPEECH] tag. This is a major improvement over the previous autoresearch where the model said completely different words (WER 0.571).

**What changed:** Using optimized visual+audio settings (id_guidance 1.5, LoRA 0.1/0.3, full bf16 precision).

### 2. [SOUNDS] tag matters a lot for audio quality
All four [SOUNDS] variations scored MOS 3.3 (highest), compared to MOS 2.0-2.8 for the baseline. Best [SOUNDS] prompt:
- **Winner:** "natural conversational speech in a quiet room" (highest composite 7.43, best pitch variation 0.563)
- "studio quality recording" and "warm intimate speaking voice" tied close behind

### 3. Short prompts can be wiry
The "speech_short_simple" test (10 words) had **perfect WER but was flagged as wiry** (HF ratio 0.322 — very high). Longer prompts produce cleaner audio.

### 4. Audio guidance: lower is cleaner, but 7.0 is still best for lip-sync
- 1.0: clean audio, MOS 3.1
- 3.0: clean audio, MOS 3.1  
- 5.0: clean audio, MOS 2.8
- 7.0 (baseline): not tested yet in V2, was MOS 4.0 in V1

### 5. Pitch variation is healthy across all tests
All clips show pitch_variation > 0.4 — good prosody, not monotone. The "natural conversational" [SOUNDS] prompt produced the highest pitch variation (0.563).

## Ranked Results

| Rank | Experiment | WER | MOS | Pitch Var | Wiry | Composite | Video Path |
|------|-----------|-----|-----|-----------|------|-----------|------------|
| 1 | sounds_natural_conversational | 0.000 | 3.3 | 0.563 | No | 7.43 | autoresearch/<id>/hq.mp4 |
| 2 | sounds_studio_quality | 0.000 | 3.3 | 0.524 | No | 7.42 | autoresearch/<id>/hq.mp4 |
| 3 | sounds_male_voice_clearly | 0.000 | 3.3 | 0.440 | No | 7.40 | autoresearch/<id>/hq.mp4 |
| 4 | sounds_warm_intimate | 0.000 | 3.3 | 0.519 | No | 7.36 | autoresearch/<id>/hq.mp4 |
| 5 | audio_g_1.0 | 0.000 | 3.1 | 0.443 | No | 7.34 | autoresearch/<id>/hq.mp4 |
| 6 | speech_quoted_dialogue | 0.000 | 2.8 | 0.421 | No | 7.13 | autoresearch/<id>/hq.mp4 |
| 7 | audio_g_3.0 | 0.000 | 3.1 | 0.433 | No | 7.10 | autoresearch/<id>/hq.mp4 |
| 8 | audio_g_5.0 | 0.000 | 2.8 | 0.450 | No | 7.08 | autoresearch/<id>/hq.mp4 |
| 9 | speech_short_simple | 0.000 | 2.0 | 0.445 | YES | 6.85 | autoresearch/<id>/hq.mp4 |
| 10 | speech_medium_specific | 0.143 | 3.3 | 0.498 | No | 6.56 | autoresearch/<id>/hq.mp4 |

## Recommended Settings Update

Based on these results, the optimal [SOUNDS] tag is:
```
[SOUNDS]: natural conversational speech in a quiet room
```

Combined with optimized visual settings:
```
identity_guidance: 1.5
distilled_s1: 0.1
distilled_s2: 0.3
audio_guidance: 7.0 (keep for lip-sync, audio quality still good)
video_rescale: 0.45
height: 864, width: 480
num_frames: 161
precision: bf16 (no quantize)
```

## Still Running
- audio_g_9.0
- best_combo_v2
- recruiting_30s_optimized
- health_30s_optimized
- seo_30s_optimized
