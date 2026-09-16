---
title: V4 Pace & Sound Quality Deep Dive
type: experiment
created: 2026-04-05
tags: [audio, pacing, audio-guidance, sounds-quality, visual-scoring]
---

# V4 Pace & Sound Quality Deep Dive

## Overview
15 experiments fine-tuning audio guidance (0.5-2.5 range), pacing control via [SPEECH] structure and [SOUNDS] quality+pace combinations. First round with visual scoring attempted (InsightFace).

## Results

| # | Name | WER | MOS | Pitch | FaceCon | IDMatch | Sharp | AudioComp | VisComp | Combined |
|---|------|-----|-----|-------|---------|---------|-------|-----------|---------|----------|
| 1 | ag_finetune_0.5 | 0.053 | 3.3 | 0.325 | 0.822 | 0.846 | 23.4 | 7.21 | 6.74 | **6.97** |
| 2 | quality_pace_deliberate | 0.000 | 3.3 | 0.361 | 0.849 | 0.879 | 36.7 | 7.36 | 6.55 | **6.96** |
| 3 | ag_finetune_1.5 | 0.000 | 3.3 | 0.354 | 0.851 | 0.873 | 51.6 | 7.36 | 6.46 | **6.91** |
| 4 | ag_finetune_2.0 | 0.000 | 3.3 | 0.432 | 0.849 | 0.869 | 48.0 | 7.52 | 6.20 | 6.86 |
| 5 | quality_pace_warm | 0.000 | 3.3 | 0.369 | 0.827 | 0.864 | 37.1 | 7.31 | 6.41 | 6.86 |
| 6 | ag_finetune_2.5 | 0.000 | 3.3 | 0.445 | 0.844 | 0.854 | 44.3 | 7.43 | 6.12 | 6.78 |
| 7 | pace_slow_explicit | 0.062 | 2.8 | 0.355 | 0.833 | 0.865 | 49.3 | 7.22 | 6.34 | 6.78 |
| 8 | quality_pace_storytelling | 0.053 | 3.3 | 0.332 | 0.846 | 0.829 | 40.6 | 7.16 | 6.35 | 6.75 |
| 9 | pace_ellipsis_pauses | 0.136 | 3.3 | 0.424 | 0.842 | 0.873 | 47.4 | 6.88 | 6.43 | 6.65 |
| 10 | pace_short_sentences | 0.136 | 3.3 | 0.420 | 0.852 | 0.861 | 51.1 | 6.73 | 6.38 | 6.55 |
| 11 | frames_241_audio | 0.000 | 2.8 | 0.412 | 0.884 | 0.885 | 35.5 | 6.79 | 6.01 | 6.40 |
| 12 | frames_121_audio | 0.211 | 3.3 | 0.357 | 0.875 | 0.857 | 35.9 | 6.56 | 6.20 | 6.38 |
| 13 | frames_201_audio | 0.263 | 3.1 | 0.350 | 0.829 | 0.844 | 25.1 | 6.27 | 5.72 | 5.99 |

## Key Findings

### Audio guidance 2.0 is the sweet spot
- ag 0.5: best visual (6.74) but slight WER (0.053) and lowest pitch variation
- ag 1.5: perfect WER, good balance
- ag 2.0: highest audio composite (7.52), perfect WER, good pitch
- ag 2.5: diminishing returns
- **Decision: audio_guidance 2.0 became the new default**

### "Deliberate" pace + quality combo is a top performer
- "clear confident voice with deliberate measured pacing" scored 6.96 combined
- Beat most single-dimension optimizations

### Pacing tricks have mixed results
- Short sentences: WER degrades (0.136) — model struggles with choppy text
- Ellipsis pauses: WER 0.136 — model ignores ellipsis
- Explicit "slow" in [SOUNDS]: slight MOS drop (2.8) but lower WER (0.062)
- **Best approach:** natural pacing from descriptive [SOUNDS], not structural hacks

### Frame count confirms 121-161 range
- 121: good face consistency (0.875) but WER degrades (0.211) — not enough time
- 161: balanced (default)
- 201: worst combined (5.99) — sharpness 25.1 (very low), WER 0.263
- 241: surprisingly decent WER (0.000) but MOS drops (2.8)

### speech_35_words has best visual metrics but worst audio
- Face consistency 0.911 (best of ALL experiments), identity 0.909, sharpness 63.9
- BUT WER 0.293 — model can't reproduce 35 words accurately
- **Confirms the tradeoff:** longer speech = better visuals but worse audio accuracy
