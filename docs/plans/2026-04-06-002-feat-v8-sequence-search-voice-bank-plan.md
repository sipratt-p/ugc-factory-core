---
title: "feat: V8 — Sequence Search Assembly + Voice Bank + OmniVoice A/B"
type: feat
status: active
date: 2026-04-06
origin: docs/brainstorms/2026-04-06-v8-sequence-search-requirements.md
---

# feat: V8 — Sequence Search + Voice Bank + OmniVoice A/B

## Overview

Shift from single-clip optimization to sequence-level search. Generate multiple candidates per segment, score all, pick the best assembly path. Simultaneously build a voice reference bank and test OmniVoice as a consistent TTS source for the A2Vid pipeline. Fix the audio-aware splice bug.

## Problem Statement

Single clips are near ceiling (UTMOS 3.7-4.2, face 0.85-0.91). Multi-clip 30-60s degrades because: voice identity re-samples per clip, one-shot-per-segment has no selection, and scoring averages hide fatal failures. GPT5 Pro identified the paradigm shift: make sequence selection part of generation. (see origin: `docs/brainstorms/2026-04-06-v8-sequence-search-requirements.md`)

## Technical Approach

### Implementation Units

#### Unit 1: Best-of-K Candidate Generator

**New file: `factory/candidate_gen.py`**

For each script segment, generate K=3 candidates with different seeds:
```python
def generate_candidates(scene, bible, k=3, base_seed=42):
    candidates = []
    for i in range(k):
        clip = generate_single_clip(
            prompt=bible.build_shot_prompt(scene.speech),
            first_frame=bible.reference_image,
            reference_audio=bible.reference_audio,
            seed=base_seed + i,
            # All other params from best settings
        )
        candidates.append(clip)
    return candidates
```

Score each candidate immediately after generation:
- Face identity match vs reference
- UTMOS naturalness
- WER against target script
- Temporal degradation (early vs late)
- Speaker similarity to reference bank centroid (Unit 2)

**GPU budget:** 3 candidates × 15 min = 45 min per segment. For 3 segments (30s): ~2.25 hours. Acceptable for overnight runs.

#### Unit 2: Voice Reference Bank

**New file: `factory/voice_bank.py`**

For the primary persona:
1. Generate 15-20 different 5-second clips with varied speech content but same [SOUNDS]
2. Extract audio from each clip
3. Denoise each with RNNoise or simple spectral subtraction
4. Loudness-normalize to -16 LUFS
5. Compute ECAPA-TDNN embedding for each
6. Build persona centroid = mean of all embeddings
7. Score each reference clip by how well it transfers (generate test clip with each, measure output voice similarity)
8. Rank references by transfer quality

Store in `personas/<slug>/voice_bank/`:
```
voice_bank/
  ref_001.wav   # best transfer
  ref_002.wav
  ...
  centroid.npy  # ECAPA embedding centroid
  rankings.json # transfer quality scores
```

**For scoring generated clips:** compare ECAPA embedding against centroid AND against all other clips in the sequence (cross-clip voice consistency).

#### Unit 3: Frozen Minimal [SOUNDS]

Update `factory/scene_bible.py`:
```python
# OLD (may re-randomize voice per clip):
voice_description = "confident authoritative male speaking voice"
speaking_pace = "measured pace, clear enunciation"
room_tone = "professional studio quality, quiet room"

# NEW (minimal, frozen):
voice_description = "close mic"
speaking_pace = "measured pace"
room_tone = "dry room, neutral studio ambience"
```

Key: the [SOUNDS] tag should describe the recording environment, NOT the voice character. Let the reference audio + identity guidance handle voice identity. Descriptive voice adjectives may fight the reference conditioning.

#### Unit 4: Sequence Search Assembly

**New file: `factory/sequence_search.py`**

Given K candidates per segment and N segments:
1. Score all K×N candidates (Unit 1 scores)
2. For each adjacent candidate pair (i, j), score the transition:
   - Face embedding similarity (ArcFace)
   - Speaker embedding similarity (ECAPA)
   - Audio energy at boundary (silence detection)
   - Pose/lighting similarity (pixel cross-correlation)
3. Build a lattice: K nodes per segment, K² edges between segments
4. Find optimal path:
   - Node cost = shot quality (negative = bad)
   - Edge cost = transition quality (negative = bad)
   - Greedy: for each segment, pick the candidate that maximizes (node_score + best_edge_score_to_previous_winner)
   - Future: full Viterbi / beam search

5. Assemble winning path with audio-aware handle splicing (Unit 5)

For 3 segments × 3 candidates: 9 nodes, 18 edges. Tiny search space, fast.

#### Unit 5: Audio-Aware Splice Fix

Debug the abrupt 2nd seam in current `handle_gen.py`:

1. **Diagnosis:** Run WhisperX on the tail of clip 1 and head of clip 2 separately to check if word boundaries are detected correctly. The issue may be that WhisperX fails on very short audio segments (~0.6s of handle region).

2. **Fix approach:** 
   - Use energy-based silence detection as PRIMARY (always works on short segments)
   - WhisperX word boundaries as SECONDARY (only if segment is long enough)
   - Require BOTH tail and head to be in low-energy region (RMS < 0.015)
   - If no silence found in handle region, extend search to more frames

3. **Validation:** Score the audio at each candidate splice point. The splice should have:
   - RMS < 0.02 on both sides of the cut
   - No word detected within 100ms of the cut point
   - Loudness delta < 0.01 across the cut

#### Unit 6: OmniVoice → A2Vid Sequence A/B

**This is a strategic fork — separate pipeline path.**

1. Generate full 30s script as one continuous OmniVoice audio track
2. Slice into per-segment audio chunks (with 0.5s overlap for handle search)
3. For each segment: feed OmniVoice audio → LTX A2Vid pipeline (NOT ID-LoRA script)
   - This requires ComfyUI with the IA2V workflow
   - Or the `a2vid_two_stage.py` script from LTX-2 repo
4. Assemble with audio-aware handles
5. Score the full sequence on all metrics

**Comparison:** Run the same script through both paths:
- Path A: Native joint AV (ID-LoRA, current approach, with beam search)
- Path B: OmniVoice TTS → A2Vid (consistent voice, lip-sync from audio)

Measure: voice consistency (ECAPA cross-clip spread), face quality, WER, UTMOS, overall belief.

### V8 Autoresearch Experiments

| # | Name | Type | What it tests |
|---|------|------|--------------|
| 1 | v8_single_baseline | single | Control with updated settings |
| 2 | v8_30s_bestof3 | multi(3×3) | Beam search: 3 candidates per segment, greedy path |
| 3 | v8_30s_bestof3_frozen_sounds | multi(3×3) | Same + frozen minimal [SOUNDS] |
| 4 | v8_30s_voice_bank_best | multi(3) | Best reference from voice bank sweep |
| 5 | v8_30s_voice_bank_worst | multi(3) | Worst reference (to measure bank impact) |
| 6 | v8_30s_omnivoice_a2vid | multi(3) | OmniVoice TTS → A2Vid sequence |
| 7 | v8_60s_bestof3 | multi(6×3) | 60s beam search |
| 8 | v8_splice_debug | multi(3) | Audio splice with enhanced silence detection |
| 9 | v8_recruiting_final | multi(3×3) | recruiting ad with all V8 improvements |
| 10 | v8_health_final | multi(3×3) | health ad with all V8 improvements |

## Acceptance Criteria

- [ ] Best-of-3 assembly scores higher than single-candidate on combined metric
- [ ] Voice bank centroid cross-clip spread is measurably tighter than single-reference
- [ ] Frozen [SOUNDS] does not reduce single-clip quality but improves multi-clip voice consistency
- [ ] OmniVoice A2Vid sequence has consistent voice across all clips (ECAPA spread < 0.1)
- [ ] No audio splice lands mid-word (boundary_penalty = 0 for all cuts)
- [ ] At least one 30s sequence is "usable" by human review

## Dependencies

- Voice bank generation: ~3 hours (20 clips × 15 min each, can run overnight)
- ECAPA-TDNN: SpeechBrain installed ✓, needs proper integration (replace spectral fallback)
- OmniVoice: installed ✓, A2Vid pipeline needs testing
- LTX A2Vid script: `<UGC_PROJECTS_DIR>/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py` (patched for Blackwell)

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| 3× generation cost may be too slow | Start with K=3 overnight; reduce to K=2 if needed |
| Voice bank generation itself has quality variance | Generate from best settings, keep only high-UTMOS clips |
| OmniVoice A2Vid may have worse lip-sync than native | This is a hypothesis test — if it fails, we know native + beam search is the path |
| Frozen [SOUNDS] may reduce audio quality | A/B test frozen vs descriptive on single clips first |

## Sources & References

### Origin
- **Origin document:** [docs/brainstorms/2026-04-06-v8-sequence-search-requirements.md](docs/brainstorms/2026-04-06-v8-sequence-search-requirements.md) — Key decisions: beam search (best-of-3), reference audio bank, frozen [SOUNDS], OmniVoice A2Vid A/B

### External
- [ChatGPT5 Pro: Sequence Search Strategy](research/wiki/sources/chatgpt5-sequence-search.md) — paradigm shift recommendation
- [V6 Multi-Clip Results](research/wiki/experiments/2026-04-06-v6-multiclip-strategies.md) — always_original wins
- [V7 Scene Bible Results](research/wiki/experiments/) — bible + per-clip id guidance
- [Human Feedback Log](research/wiki/human-feedback.md) — "abrupt audio splice", "melty 60s", "teeth quality"
