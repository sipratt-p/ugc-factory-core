---
date: 2026-04-07
topic: v12-overnight-experiments
---

# V12 Overnight Experiment Plan

## Context

The V10 health-ad beam was our near-postable high water mark (human review).
V11 test fire had the best seams we've ever measured (flow_disc 0.1-0.2).
V11b/c blending approaches are dead — cause face distortion.

The remaining issue: visible "reset" at transitions when using last-frame conditioning.
Each clip starts from previous pose but identity guidance pulls face back to default.

## Experiment Grid

### Exp 1: V11 Baseline + Longer Crossfade (12 frames = 0.5s)
- **Hypothesis:** The reset happens in the first ~0.5s of each clip. A 0.5s crossfade blends through the reset period, making it invisible to the viewer.
- **Settings:** 100% last-frame conditioning, identity_guidance=1.5, audio_guidance=3.5, simple xfade 12 frames, no handle trim
- **Products:** recruiting-ad, health-ad
- **K=1** (fast, just testing crossfade duration)
- **Metrics:** flow_disc (should stay low), visual smoothness on playback

### Exp 2: Original-Ref-Only + 12-frame Crossfade
- **Hypothesis:** Skip last-frame conditioning entirely. Use original reference for ALL segments. The 12-frame crossfade handles the visual transition. This avoids both the reset AND the drift risk.
- **Settings:** Original reference for all segments, identity_guidance=1.5, audio_guidance=3.5, simple xfade 12 frames
- **Products:** recruiting-ad, health-ad
- **K=1**
- **Metrics:** flow_disc (will be higher than V11), DINO consistency, visual quality

### Exp 3: V11 + Identity Guidance 1.2 (mild reduction)
- **Hypothesis:** V11b's id_guidance=0.7 was too low (face collapse). 1.2 might be the sweet spot — slight reduction in pull-back without losing identity.
- **Settings:** 100% last-frame, identity_guidance=1.2 (segments 1-3), 1.5 (segment 0), audio_guidance=3.5, simple xfade 4 frames
- **Products:** recruiting-ad
- **K=1**
- **Metrics:** Identity similarity to reference per segment, face collapse detection

### Exp 4: Best-of-3 Beam with V11 Settings
- **Hypothesis:** Beam search K=3 with V11's proven settings will find better individual clips, and the seam-aware beam selection (using boundary face embeddings) will pick clips that transition smoothly.
- **Settings:** V11 settings (100% last-frame, id_guidance=1.5, audio_guidance=3.5, simple xfade), K=3 beam with TOPIQ-face + flow-based transition scoring
- **Products:** recruiting-ad, health-ad, SEO-ad
- **K=3**
- **Metrics:** All fleet metrics, VideoScore2 per-clip evaluation

### Exp 5: health-ad with V11 Settings (reproduce high water mark)
- **Hypothesis:** The V10 health-ad was near-postable. V11 settings (simple xfade, no handle trim, audio_guidance=3.5) should improve its seams while maintaining the quality the reviewer liked.
- **Settings:** V11 best settings, K=1 first then K=3
- **Products:** health-ad only
- **K=1 then K=3**
- **Metrics:** Direct comparison to the V10 health-ad

### Exp 6: 5-segment × 121 frames (alternative segmentation)
- **Hypothesis:** 5×121fr = 25.2s (vs 4×161fr = 26.8s). Shorter clips have better within-clip quality (less drift). More cuts but each cut is between shorter, sharper clips.
- **Settings:** V11 settings, 5 segments × 121 frames
- **Products:** recruiting-ad (5-segment script)
- **K=1**
- **Metrics:** Within-clip DINO drift, TOPIQ-face, seam quality

## Metrics for All Experiments

Per-clip:
- DINO consistency + drift (temporal stability)
- TOPIQ-face (face quality)
- MUSIQ (general quality)
- Lip sync correlation + offset
- LPIPS temporal (flicker)
- Identity similarity to reference (ArcFace)

Per-seam:
- RAFT flow discontinuity (motion jump)
- LPIPS at boundary (visual jarring)

Per-sequence:
- VideoScore2 multi-dim evaluation (if time permits)
- NISQA decomposition
- Total duration verification

## Run Order

1. **Tonight (test fire):** Exp 1 (V11 + 12-frame xfade) — K=1, ~16 min. Quick check if longer crossfade masks the reset.
2. **Overnight Exp 2-3:** Original-ref-only and mild id_guidance reduction — K=1 each, ~16 min each.
3. **Overnight Exp 4:** Beam K=3 with best settings from 1-3 — all 3 products, ~75 min each.
4. **Overnight Exp 5:** health-ad focus — reproduce + improve high water mark.
5. **Morning Exp 6:** Alternative segmentation if time remains.

Total overnight: ~6-8 hours depending on which experiments are queued.

## Success Criteria

- At least one clip the human reviewer rates as "postable"
- Seam flow_disc < 0.5 for all transitions
- Identity similarity > 0.6 for all segments
- No face collapse or distortion in final segments
- Duration 24-27s
