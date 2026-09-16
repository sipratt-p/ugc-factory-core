---
title: V3 Advanced [SOUNDS] + Anti-Wiriness
type: experiment
created: 2026-04-05
tags: [audio, sounds-tag, wiriness, speech-length, video-rescale]
---

# V3 Advanced [SOUNDS] + Anti-Wiriness

## Overview
14 experiments pushing audio quality further. Advanced [SOUNDS] tag engineering, anti-wiriness speech length tests, video rescale impact on audio, and multi-clip product ads.

## Results

| # | Name | WER | MOS | Pitch | Wiry | Audio Comp |
|---|------|-----|-----|-------|------|-----------|
| 1 | sounds_casual_friend | 0.000 | 3.3 | 0.490 | No | **7.53** |
| 2 | sounds_authoritative | 0.000 | 3.3 | 0.436 | No | **7.52** |
| 3 | sounds_podcast_quality | 0.000 | 3.3 | 0.477 | No | **7.50** |
| 4 | sounds_deep_resonant | 0.000 | 3.3 | 0.505 | No | **7.49** |
| 5 | ultimate_combo | 0.037 | 3.3 | 0.507 | No | 7.15 |
| 6 | recruiting_30s_v3 | 0.082 | 3.3 | 0.515 | No | 6.99 |
| 7 | seo_30s_v3 | 0.137 | 3.3 | 0.479 | No | 6.80 |
| 8 | speech_25_words | 0.111 | 3.3 | 0.476 | No | 6.74 |
| 9 | sounds_asmr_intimate | 0.211 | 3.3 | 0.500 | No | 6.63 |
| 10 | health_30s_v3 | 0.180 | 3.3 | 0.502 | No | 6.55 |
| 11 | speech_35_words | 0.293 | 3.3 | 0.533 | No | 6.37 |
| 12-14 | rescale_0.3/0.5/0.7 | 0.000 | 3.3 | 0.413 | No | 7.28 each |

## Key Findings

### All advanced [SOUNDS] variants scored 7.4+ audio composite
- **casual_friend** (7.53): "casual friendly male voice like talking to a friend"
- **authoritative** (7.52): "confident authoritative male speaking voice, measured pace"
- All beat the V2 "natural conversational" winner (7.43)

### Video rescale has ZERO effect on audio
- rescale 0.3, 0.5, 0.7 all produced identical audio scores (7.28)
- This is purely a visual parameter — confirmed

### Longer speech = worse WER but better pitch variation
- 25 words: WER 0.111, pitch 0.476
- 35 words: WER 0.293, pitch 0.533 (most natural prosody)
- Sweet spot: 15-20 words per clip

### ASMR/intimate style hurts accuracy
- sounds_asmr_intimate: WER 0.211 — whisper style confuses the model

### Spoken brand names plus URLs are still problematic
- WER 0.180 on the health ad — a two-word brand name followed by "dot AI" is hard for the model
