# UGC Autoresearch Leaderboard

Generated: 2026-04-05
Total experiments: 22 (14 Phase 1 + 8 Phase 2)

Multi-clip experiment labels were generalized for publication (recruiting / health / seo
name the ad vertical, not the client); every score is as recorded.

## Key Findings

### 1. Identity Guidance is the most impactful parameter
- **1.0-1.5 = best balance** — natural expressions, clean teeth, good face stability
- **0.5 = too low** — mouth distortion, over-expressive
- **2.0 = good** (our previous best), slight face rigidity
- **3.0 = too high** — flat expressions, overly stable but lifeless

### 2. LoRA strength: lower is better
- **0.1/0.3 = cleanest natural look** — best skin texture, most natural motion
- **0.25/0.5 (baseline) = good** — solid balance
- **0.5/0.6 = too heavy** — compressed, flatter skin, less natural

### 3. Video rescale: 0.45 is the sweet spot
- **0.3 = too aggressive** — visible mouth/teeth distortion
- **0.45 (baseline) = best** — clean, no overbaking
- **0.6 = similar distortion to 0.3** — mouth area unstable

### 4. Audio guidance: 7.0 is correct
- **3.0 = weaker lip-sync**, less mouth movement
- **5.0 = moderate**, decent
- **7.0 (baseline) = strongest lip-sync**, most natural speech animation

### 5. Resolution: 480x864 (960x1728) is the sweet spot
- **320x576 (640x1152)** — faster but noticeably softer
- **480x864 (960x1728)** — best detail-to-quality ratio
- **704x384 (768x1408)** — wider aspect, similar quality

### 6. Frame count: 121-161 is optimal
- **121 frames (5s)** — sharpest per-frame quality, speech too fast
- **161 frames (6.7s)** — best balance of quality + pacing
- **201 frames (8.4s)** — slight quality drop, better content density

### 7. Multi-clip quality degrades after 3-4 clips
- **30s (3 clips)** — quality holds well across all clips
- **60s (6 clips)** — significant degradation in clips 5-6 (skin distortion, arm artifacts, color shift)
- **Root cause:** last-frame conditioning chain accumulates errors
- **Fix needed:** reset to original reference image every 3-4 clips, or use IAMCCS latent overlap

### 8. The selfie arm problem
- The reference photo has an extended arm (selfie pose)
- The model reproduces this arm in every clip
- Later clips distort the arm severely
- **Fix:** crop the reference image to head-and-shoulders only

## Recommended Settings

```json
{
  "identity_guidance": 1.5,
  "video_guidance": 3.0,
  "audio_guidance": 7.0,
  "video_rescale": 0.45,
  "distilled_s1": 0.1,
  "distilled_s2": 0.3,
  "height": 864,
  "width": 480,
  "num_frames": 161,
  "quantize": false,
  "precision": "bf16"
}
```

## Ranked Results — Single Clips

| Rank | Experiment | Teeth | Skin | Face Stability | Sharpness | Motion | Overall |
|------|-----------|-------|------|---------------|-----------|--------|---------|
| 1 | id_guidance_1.0 | 8 | 8 | 7 | 8 | 9 | 8.0 |
| 2 | lora_0.1_0.3 | 8 | 9 | 7 | 8 | 8 | 8.0 |
| 3 | id_guidance_1.5 | 8 | 8 | 8 | 8 | 8 | 8.0 |
| 4 | baseline_best (id_g 2.0) | 7 | 8 | 8 | 8 | 7 | 7.6 |
| 5 | audio_guidance_5.0 | 7 | 8 | 8 | 8 | 7 | 7.6 |
| 6 | combo_best_predicted | 7 | 8 | 8 | 8 | 7 | 7.6 |
| 7 | frames_121 (5s) | 7 | 8 | 8 | 9 | 7 | 7.8 |
| 8 | audio_guidance_3.0 | 7 | 8 | 8 | 8 | 6 | 7.4 |
| 9 | id_guidance_3.0 | 7 | 8 | 9 | 8 | 5 | 7.4 |
| 10 | lora_0.5_0.6 | 6 | 7 | 8 | 7 | 7 | 7.0 |
| 11 | video_rescale_0.3 | 5 | 7 | 7 | 7 | 8 | 6.8 |
| 12 | video_rescale_0.6 | 5 | 7 | 7 | 7 | 8 | 6.8 |
| 13 | id_guidance_0.5 | 5 | 7 | 5 | 7 | 9 | 6.6 |

## Ranked Results — Multi-Clip

| Rank | Experiment | Duration | Clips | Quality Consistency | Seam Quality | Overall |
|------|-----------|----------|-------|-------------------|-------------|---------|
| 1 | recruiting_30s_longer_clips | 16s | 2 | 8 | 8 | 8.0 |
| 2 | recruiting_30s | 19s | 3 | 7 | 7 | 7.0 |
| 3 | health_30s | 19s | 3 | 7 | 7 | 7.0 |
| 4 | seo_30s | 19s | 3 | 7 | 7 | 7.0 |
| 5 | recruiting_60s | 39s | 6 | 4 | 6 | 5.0 |
| 6 | health_60s | 39s | 6 | 4 | 6 | 5.0 |

## Action Items

1. **Update pipeline defaults** to identity_guidance 1.5, LoRA 0.1/0.3
2. **Crop reference images** to head-and-shoulders (remove selfie arm)
3. **Cap multi-clip videos at 30s (3 clips)** until seam quality improves
4. **For 60s videos:** reset reference image every 3 clips to prevent degradation
5. **Test with cropped reference** — the selfie arm is causing significant distortion
