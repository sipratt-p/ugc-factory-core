---
title: Fleet Scoring Cross-Version Analysis (V6-V10)
type: experiment
created: 2026-04-07
---

# Fleet Scoring: Cross-Version Analysis

Scored all multi-clip experiments from V6 through V10 with the new metric fleet:
DINOv2 drift, RAFT optical flow, LPIPS temporal, lip sync correlation, TOPIQ-face, MUSIQ.

**Data:** `<UGC_RESULTS_DIR>/fleet_scores_all.json` (not included)

## Critical Discovery: Seam Quality Regressed 10-50x in V8+

| Version | Seam Flow Discontinuity | Method |
|---|---|---|
| V6 30s (6 clips) | **0.12 - 0.16** | Simple hard concat |
| V6 60s (6 clips) | **0.15 - 0.17** | Simple hard concat |
| V7 (3 clips) | **0.10 - 0.30** | Simple hard concat |
| V8 beam (3 clips) | 2.1 - 2.4 | Handle splice + trim |
| V9 6x121 (6 clips) | 1.4 - 4.7 | Handle splice + trim |
| V9b 4x161 (4 clips) | 2.3 - **10.8** | Handle splice + trim + crossfade |
| V10 (4 clips) | **0.08** - 11.2 | Last-frame cond + handle + crossfade |

**Root cause hypothesis:** The handle trim → re-encode → concat pipeline introduces frame-level discontinuities that didn't exist in V6/V7's simple concat. The re-encoding of trimmed clips at cut points may be shifting frames by fractional amounts, and the handle splice search may be cutting at suboptimal visual frames while prioritizing audio safety.

**Evidence:**
- V10's first seam (with last-frame conditioning, no handle trim needed) = 0.08 flow_disc
- V10's other seams (with handle trim) = 3.4-11.2 flow_disc
- V6/V7 (no handle trim at all) = 0.1-0.3 flow_disc

## Visual Quality Improved Massively V6→V8+

| Version | TOPIQ-face avg | MUSIQ avg | Notes |
|---|---|---|---|
| V6 30s | 0.548 | 53.0 | Low face quality |
| V6 60s | 0.392 | 62.2 | Severe degradation |
| V7 | 0.448-0.582 | 59-78 | Variable |
| V8 | 0.755 | 64.0 | **+37% vs V6** |
| V9b | 0.750 | 63.0 | Consistent |
| V10 | 0.748 | 63.0 | Consistent |

The generation quality gains from V6→V8 (full precision, optimized guidance, frozen [SOUNDS]) are massive and real. The V8+ clips look much better per-frame.

## Segment 2 is Universally Weak

Across V9b and V10, segment 2 (third clip) consistently shows:
- DINO consistency: 0.938 (vs 0.965-0.983 for other segments)
- DINO drift: 0.308 (vs 0.013-0.088)
- LPIPS temporal: 0.123 (vs 0.026-0.049) — 4x more frame-to-frame change
- Flicker: 0.068 (vs 0.003-0.017) — 7x more flickery
- TOPIQ-face: 0.734 (vs 0.755-0.802)
- MUSIQ: 58.1 (vs 62-66)

This is the same clip human review flagged for visual quality issues. It happens regardless of ad vertical (recruiting, health) suggesting a systematic issue with the third generation in sequence.

## Lip Sync is Mediocre Across All Versions

| Version | Best Correlation | Typical Range |
|---|---|---|
| V6 | 0.329 | -0.14 to 0.21 |
| V7 | 0.305 | -0.19 to 0.15 |
| V8 | 0.432 | 0.11 - 0.43 |
| V9b/V10 | 0.418 | 0.07 - 0.42 |

audio_guidance=2.0 produces mediocre lip sync across all versions. Shorter clips (121fr) have marginally better sync than 161fr.

## V10 Last-Frame Conditioning: Confirmed Win

| Seam | V9b flow_disc | V10 flow_disc | Improvement |
|---|---|---|---|
| s00→s01 (recruiting) | 2.52 | **0.24** | **10x** |
| s00→s01 (health) | 2.49 | **0.08** | **31x** |
| s00→s01 (proof) | 2.91 | **0.19** | **15x** |

Last-frame conditioning makes the first seam nearly invisible. Must extend to all transitions.

## The SEO Ad Has the Best Seam Quality

v9b_seo_beam3: all seams 1.8-2.7 flow_disc (vs 2.5-10.8 for the recruiting and health ads). The SEO scripts may produce more consistent visual output across segments, or the specific speech content creates more natural pause points.
