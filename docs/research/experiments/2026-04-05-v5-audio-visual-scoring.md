---
title: V5 Audio + Visual Scoring Results
type: experiment
created: 2026-04-05
updated: 2026-04-06
tags: [audio, visual, insightface, utmos, omnivoice]
---

# V5 Audio + Visual Scoring Results

## Overview
12 experiments — first round with full visual scoring (InsightFace face consistency, identity match, sharpness, mouth movement) alongside audio scoring. Also includes OmniVoice test and [SOUNDS] refinement.

## Results

| # | Name | WER | MOS | Face Con | ID Match | Sharp | Audio Comp | Visual Comp | Combined |
|---|------|-----|-----|----------|----------|-------|-----------|-------------|----------|
| 1 | v5_ag_confirm_1.0 | 0.000 | 3.3 | 0.840 | 0.853 | 38.3 | 7.35 | 6.55 | **6.95** |
| 2 | v5_speech_18_words | 0.000 | 3.6 | 0.842 | 0.865 | 53.6 | 7.52 | 6.36 | **6.94** |
| 3 | sounds_v5_podcast_casual | 0.000 | 3.3 | 0.853 | 0.859 | 42.6 | 7.35 | 6.45 | 6.90 |
| 4 | v5_baseline_best | 0.000 | 3.3 | 0.858 | 0.857 | 45.5 | 7.43 | 6.32 | 6.88 |
| 5 | v5_ag_confirm_2.0 | 0.000 | 3.3 | 0.859 | 0.858 | 45.4 | 7.43 | 6.32 | 6.88 |
| 6 | v5_ag_confirm_3.0 | 0.000 | 3.3 | 0.858 | 0.855 | 44.3 | 7.45 | 6.29 | 6.87 |
| 7 | v5_omnivoice_test | 0.000 | 3.3 | 0.857 | 0.857 | 45.5 | 7.43 | 6.29 | 6.86 |
| 8 | v5_speech_22_words | 0.083 | 3.3 | 0.793 | 0.824 | 46.8 | 6.96 | 6.59 | 6.78 |
| 9 | sounds_v5_ted_talk | 0.000 | 3.3 | 0.875 | 0.864 | 40.2 | 7.45 | 6.07 | 6.76 |
| 10 | sounds_v5_warm_authority | 0.211 | 3.3 | 0.856 | 0.859 | 39.7 | 6.59 | 6.56 | 6.57 |
| 11 | health_30s_v5 | 0.171 | 2.8 | 0.711 | 0.785 | 41.2 | 6.63 | 6.29 | 6.46 |
| 12 | recruiting_30s_v5 | 0.390 | 3.3 | 0.733 | 0.753 | 40.8 | 5.90 | 6.42 | 6.16 |

## Key Findings

### Audio guidance 1.0 produces best combined score
- v5_ag_confirm_1.0 combined 6.95 — best visual composite (6.55) because less GPU budget spent on audio
- v5_ag_confirm_2.0 and 3.0 nearly tied for audio but lower visual

### 18 words is the speech sweet spot
- v5_speech_18_words: highest MOS (3.6!), perfect WER, good sharpness (53.6)
- 22 words: WER starts dropping (0.083), face consistency drops (0.793)

### OmniVoice subjectively better despite similar scores
- UTMOS later confirmed: OmniVoice 3.703 vs native LTX 2.500-3.291
- Our heuristic MOS couldn't distinguish — led to UTMOS integration

### Multi-clip face consistency drops severely
- Single clips: face_consistency 0.84-0.86, identity_match 0.85-0.87
- 30s multi-clip: face_consistency **0.71-0.73**, identity_match **0.75-0.79**
- This gap drives the V6/V7 multi-clip strategy research

## Video Files

All clips were kept at `<UGC_RESULTS_DIR>/<test_id>/` (not included in this repository).
