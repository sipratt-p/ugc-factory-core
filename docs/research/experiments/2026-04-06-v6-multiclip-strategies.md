---
title: V6 Multi-Clip Strategy Comparison
type: experiment
created: 2026-04-06
updated: 2026-04-06
tags: [multi-clip, reference-strategy, seam, continuity, hard-cut]
---

# V6 Multi-Clip Strategy Comparison

## Overview
11 experiments testing different strategies for maintaining quality across multi-clip 30-60s videos. Focus on reference image strategies, seed consistency, and clip length.

## 30s Results (ranked by combined score)

| Rank | Strategy | Combined | Face Consistency | Drift | Clips | Key Insight |
|------|----------|----------|-----------------|-------|-------|-------------|
| 1 | **long_clips (201fr)** | **7.47** | 0.878 | 0.878 | 2 | Highest score BUT 39% sharpness degradation within clip. Misleading — exceeds LTX 161fr range. |
| 2 | **always_original** | **7.33** | 0.818 | 0.830 | 3 | Strong — no chain error accumulation. Best drift control. |
| 3 | **same_seed** | **7.28** | 0.809 | 0.741 | 3 | Good face consistency. Same noise pattern helps. |
| 4 | **reset_2** | **7.26** | 0.792 | 0.820 | 3 | Works but first clip drifted (eyebrows). Good recovery on clips 2-3. |
| 5 | **chain** | **7.19** | 0.781 | 0.718 | 3 | Current approach. Error accumulates. |
| 6 | **inc_seed** | **6.76** | 0.732 | 0.650 | 3 | Worst — different noise per clip = most inconsistency. |

## 60s Results (partial, 2 of 4 experiments completed)

| Strategy | Combined | Face Consistency | Drift | Clips |
|----------|----------|-----------------|-------|-------|
| reset_3 | 7.02 | 0.772 | 0.752 | 6 |
| chain | 6.80 | 0.747 | 0.637 | 6 |

## Key Findings

### 1. always_original is the best strategy
- No error accumulation — every clip starts fresh from the real photo
- Best drift control (0.830 — highest among 3-clip strategies)
- Slightly lower face_consistency than long_clips but no intra-clip degradation

### 2. long_clips score is misleading
- Scores highest (7.47) because fewer clips = fewer seams
- BUT human review found severe intra-clip degradation (39% sharpness drop)
- 201 frames exceeds LTX recommended range (121-161)
- **Conclusion: don't exceed 161 frames per clip**

### 3. same_seed helps but isn't sufficient alone
- Same noise pattern = more consistent generation
- Combine with always_original for best results

### 4. inc_seed is the worst approach
- Different noise per clip = maximum inconsistency
- Never use this

### 5. First clip is the weakest point
- Human review of reset_2: clip 1 drifted eyebrows despite good clips 2-3
- Hypothesis: need higher identity_guidance for clip 1 specifically
- V7 tests per-clip id_guidance schedules [3.0, 1.5, 1.5]

### 6. 60s chains degrade worse than 30s
- 6-clip chain: face consistency 0.747, drift 0.637
- 6-clip reset_3: better (0.772, 0.752)
- always_original for 60s should be the strongest (still running)

## Recommended Strategy

For 30s (3 clips): `always_original` + same seed + 161 frames
For 60s (6 clips): `always_original` + same seed + 121 frames (shorter = sharper)

## What V7 Tests Next

1. Scene Bible for consistent prompts across clips
2. Per-clip identity guidance (3.0 for clip 1, 1.5 for rest)
3. 121-frame clips (6 clips for 30s — max sharpness per clip)
4. Three-level scoring (shot + cut + sequence)
5. All three ad verticals (recruiting, health, SEO) with best strategy
