---
title: V9/V9b/V10 — Seam Quality + Scoring Upgrades
type: experiment
created: 2026-04-07
experiments: 11+
---

# V9/V9b/V10: Seam Quality + Scoring Upgrades

## V9 (6x121fr) — REGRESSION
- 3 experiments, 6 segments at 121 frames
- Too many cuts (every 5s), chops sentences mid-thought
- WER catastrophe: 5/6 shots WER>0.5 for the recruiting ad
- **Reverted** — V7's 3-4 segments at 161fr was better

## V9b (4x161fr) �� GOOD
- 4 experiments (recruiting proof, recruiting beam, health beam, SEO beam)
- Scripts rewritten: each segment = one complete thought
- **Human review: "pretty good, getting close"**
- NISQA Coloration: all 3.85-4.40 (no wiriness)
- Seam scores: avg 0.71 (up from 0.54 in V9)
- Face deltas: 0.10-0.19

| Name | Dur | Final | Mean Shot | Penalties |
|------|-----|-------|-----------|-----------|
| v9b_proof_k1 | 24.9s | 1.81 | 6.67 | 2 |
| v9b_recruiting_beam3 | 24.9s | 1.91 | 6.88 | 2 |
| v9b_health_beam3 | 25.0s | 2.00 | 7.02 | 2 |
| v9b_seo_beam3 | 24.7s | 2.11 | 6.94 | 2 |

## V9b + Crossfade Fix
- Handle frames: 12→24 (doubled search window)
- 4-frame cross-dissolve at every splice (ffmpeg xfade)
- **Human review: "seams improved, cuts 2-3 and 3-4 better. First transition still abrupt."**
- Mean splice score: 0.892

## V10 (RUNNING)
- Seam-aware beam: transition scoring uses actual cut-point face embeddings
- Last-frame conditioning: clip 2 uses clip 1's last frame as first_frame
- Each candidate stores first/last frame embeddings + tail/head RMS
- 24 handle frames + crossfade carried forward

## Scoring Upgrades (V9+)
- NISQA decomposition: MOS, Noisiness, Discontinuity, Coloration (wiriness), Loudness
- ECAPA-TDNN: 192-dim speaker embeddings replacing spectral fallback
- Seam scoring: face_delta, loudness_delta, NISQA quality delta at cuts
- Within-clip: identity_drop + sharpness_slope
- Rebalanced: 50% shot + 25% seam + 25% sequence
- New penalties: wiriness, identity degradation, sharpness drop, loudness jump
