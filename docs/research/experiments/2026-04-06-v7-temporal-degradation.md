---
title: V7 — Temporal Degradation + Scene Bible + Handle Splice
type: experiment
created: 2026-04-06
experiments: ~20
---

# V7: Temporal Degradation + Scene Bible + Handle Splice

## What Changed
- Scene Bible system for consistent prompts across clips
- Temporal degradation scoring (early frames vs late frames)
- Handle-based splice search (audio-aware)
- UTMOS neural MOS integrated
- Hard penalties for WER > 0.5 (catastrophic -3.0)
- Frozen [SOUNDS]: "close mic, measured pace, dry room, neutral studio ambience"
- always_original reference strategy confirmed as best

## Key Results
- Single-clip UTMOS: 3.7-4.2
- Face consistency: 0.85-0.91 for single clips
- Multi-clip (30s, 3x161fr): face consistency 0.77-0.82
- 201 frames: 39% sharpness degradation confirmed
- 121-161 frames: quality holds within clip

## Human Reviews
- "almost usable, teeth and mouth lower quality" (30s always_original)
- "degrades badly, would not be usable" (60s chain)
- Audio-aware splice: "improved but 2nd seam still abrupt"
