---
title: Metrics Reference — What We Track and Why
type: overview
created: 2026-04-06
updated: 2026-04-06
---

# Metrics Reference

## Audio Metrics

### WER (Word Error Rate) — `score_av.py`
- **What:** Percentage of words the model got wrong vs the target script
- **How:** WhisperX transcription compared to `[SPEECH]` tag content
- **Range:** 0.000 (perfect) to 1.0 (completely wrong)
- **Impact:** **HIGHEST IMPACT.** V1 experiments had WER 0.571 (model said completely different words). After prompt optimization, single clips hit WER 0.000. Multi-clip still degrades (0.09-0.39).
- **Weight in composite:** 25%

### UTMOS (Neural MOS) — `score_av.py`
- **What:** Perceptual naturalness score trained on human MOS ratings
- **How:** `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`
- **Range:** 1.0 (robotic) to 5.0 (indistinguishable from human)
- **Impact:** **CRITICAL.** Our old heuristic MOS couldn't distinguish OmniVoice (3.703) from native LTX (2.500) — both scored ~3.3. UTMOS sees what human ears hear.
- **Weight in composite:** 25% (replaced heuristic MOS as primary naturalness signal)

### MOS Estimate (Heuristic) — `score_av.py`
- **What:** Rule-based naturalness estimate from spectral analysis
- **How:** RMS, spectral flatness, pitch variation, HF ratio → heuristic score
- **Range:** 1.0-5.0
- **Impact:** Low — can't distinguish AI voice quality differences. Kept as fallback when UTMOS unavailable.
- **Weight:** Only used if UTMOS fails

### Pitch Variation — `score_av.py`
- **What:** How much the pitch varies during speech (monotone detection)
- **How:** Autocorrelation-based pitch tracking, standard deviation / mean
- **Range:** 0.0 (perfectly monotone) to 1.0+ (highly variable)
- **Impact:** Moderate. All our clips show >0.4 (not monotone). "Natural conversational" [SOUNDS] tag produces highest variation (0.563).
- **Threshold:** <0.08 = monotone flag

### HF Ratio (Wiriness) — `score_av.py`
- **What:** High-frequency energy as percentage of total — detects metallic/wiry AI voice
- **How:** DFT, ratio of >6kHz energy to total
- **Range:** 0.0-1.0
- **Impact:** Moderate. Short clips (<15 words) consistently wiry (HF >0.25). Longer clips and OmniVoice produce lower HF.
- **Threshold:** >0.25 = wiry flag

### Words Per Second — `score_av.py`
- **What:** Speaking rate
- **How:** Word count / duration from WhisperX timestamps
- **Range:** 1.5 (slow) to 4.0+ (rushed)
- **Impact:** Moderate. Natural speech is ~2.5 WPS. Our clips tend toward 2.7-3.0 (slightly fast). Descriptive [SOUNDS] tags help.

### Speaker Similarity — `score_av.py`
- **What:** How similar the generated voice is to the reference audio
- **How:** Spectral embedding cosine similarity (placeholder for ECAPA-TDNN)
- **Range:** 0.0 (completely different) to 1.0 (identical)
- **Impact:** Low with current implementation (~0.2 for all clips). Needs upgrade to ECAPA-TDNN for meaningful discrimination.
- **Weight:** 10%

## Visual Metrics

### Face Consistency — `score_visual.py`
- **What:** How consistent the face looks across all frames in a clip
- **How:** InsightFace ArcFace embeddings, pairwise cosine similarity
- **Range:** 0.0-1.0 (higher = more consistent)
- **Impact:** **HIGH.** Single clips: 0.84-0.91. Multi-clip (3 clips): drops to 0.77-0.82. Multi-clip (6 clips): 0.71-0.75. This is the primary multi-clip quality signal.
- **Weight in visual composite:** 25%

### Identity Match — `score_visual.py`
- **What:** How well the generated face matches the reference photograph
- **How:** ArcFace embedding cosine similarity vs reference image
- **Range:** 0.0-1.0
- **Impact:** **HIGH.** Best single clips: 0.87-0.91. Drops across multi-clip chains. The `always_original` reference strategy maintains this best (0.83 vs 0.75 for chain).
- **Weight in visual composite:** 25%

### Sharpness — `score_visual.py`
- **What:** Image detail / blur level
- **How:** Laplacian variance (higher = sharper)
- **Range:** ~10 (very blurry) to ~70+ (very sharp)
- **Impact:** **HIGH for temporal.** 121-frame clips: sharpness ~50-60. 201-frame clips: starts at 71 but DROPS TO 43 by end (39% degradation). This is why 201+ frames is bad.
- **Weight in visual composite:** 25%

### Mouth Movement — `score_visual.py`
- **What:** How much the mouth region moves between frames (lip animation)
- **How:** Face region extraction + pixel difference tracking
- **Range:** 0 (static) to 30+ (very animated)
- **Impact:** Moderate. Confirms lip-sync is active. Most clips 18-25 range.
- **Weight in visual composite:** 25%

### Eyebrow Drift — `score_visual.py`
- **What:** How much eyebrows deviate from reference image landmarks
- **How:** InsightFace 106-point landmarks, eyebrow region (33-42) comparison
- **Range:** 0.0 (no drift) to 0.1+ (severe drift)
- **Impact:** Added from V6 human review — the reviewer caught eyebrow drift that other metrics missed.

## Temporal Metrics (Intra-Clip Degradation)

### Visual Degradation — `score_temporal.py`
- **What:** Quality drop from start to end of a single clip
- **How:** Compare face embeddings, sharpness, identity match between first 20% and last 20% of frames
- **Range:** 0.0 (no degradation) to 1.0 (severe)
- **Impact:** **CRITICAL DISCOVERY.** 201-frame clips show 0.244 degradation score. 121-frame clips show <0.1. This is why we stay in the 121-161 frame range.
- **Key metrics:** identity_early vs identity_late, sharpness_early vs sharpness_late

### Audio Degradation — `score_temporal.py`
- **What:** Audio quality drop from start to end
- **How:** UTMOS on first 30% vs last 30% of audio
- **Range:** 0.0-1.0
- **Impact:** Moderate. Interestingly, audio sometimes IMPROVES over clip duration (UTMOS 3.84→4.06).

## Sequence Metrics (Multi-Clip)

### Cross-Clip Face Consistency — `score_sequence.py`
- **What:** Face similarity between clips (not within clips)
- **How:** ArcFace embedding from each clip's mid-frame, pairwise cosine similarity
- **Range:** 0.0-1.0
- **Impact:** **PRIMARY multi-clip signal.** chain: 0.781, always_original: 0.818, same_seed: 0.809.

### Global Face Drift — `score_sequence.py`
- **What:** How much the face changed from first clip to last clip
- **How:** Cosine similarity between clip 1 and clip N face embeddings
- **Range:** 0.0-1.0 (higher = less drift = better)
- **Impact:** HIGH. chain: 0.718, always_original: 0.830. Directly measures the "belief degradation" problem.

### Cut Score — `score_sequence.py`
- **What:** Quality of each hard cut boundary
- **How:** Face jump across cut, loudness delta, whether cut lands in pause
- **Range:** 0.0-1.0 (higher = cleaner cut)
- **Impact:** Moderate. Helps identify specific bad seams vs overall quality.

### Combined Sequence Score — `score_sequence.py`
- **Formula:** 55% mean shot score + 25% mean cut score + 20% sequence score
- **Hard penalties:** -1.0 for WER>0.5 in any clip, -0.5 for drift>0.3, -0.5 for face jump>0.3 at cut

## Composite Scores

### Audio Composite — `score_av.py`
- **Formula:** 25% WER + 25% UTMOS + 10% pacing + 10% speaker_sim + 5% has_speech
- **Range:** 0-10

### Visual Composite — `score_visual.py`
- **Formula:** 25% face_consistency + 25% identity_match + 25% sharpness + 25% mouth_movement
- **Range:** 0-10

### Combined — per-experiment
- **Formula:** 50% audio_composite + 50% visual_composite
- **For multi-clip:** 55% shots + 25% cuts + 20% sequence (with hard penalties)
