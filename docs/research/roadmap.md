# Roadmap — Ideas, Experiments, and Diverging Paths

> **Note:** Some experiments below diverge from the known-good production pipeline.
> The proven path is documented in the V2 section of log.md. Experimental findings
> here are reference points for future exploration, NOT current best practices.

---

## Current Best Practice (Production Pipeline — V2)
- **Image source:** Real photos, AI-upscaled real photos, or Qwen-Image-2512 generated
- **Voice:** a high-energy male reference clip (male), sohee via Qwen3-TTS (female)
- **Settings:** V13 defaults (tuned on the primary persona) as baseline, per-persona audio_guidance tuning
- **Quality gate:** SyncNet LSE-C primary scorer, adaptive retry up to 5 seeds
- **Assembly:** Trim at last silence_start, concat (no crossfade — kills lip sync)
- **Script rules:** Single sentences with commas, ~7 words/segment, strong consonant openers

---

## Implemented & Validated

### SyncNet Quality Gate (2026-04-16)
- Replaced old correlation metric (was lying — scored 0.77 on dead lip sync)
- SyncNet LSE-C/LSE-D as primary scorer via `factory/score_sync.py`
- POOR = reject + reseed, FAIR/GOOD = pass
- Up to 5 seed attempts per segment
- Status: **In production.** See `factory/quality_gate.py`.

### Qwen-Image-2512 for AI Personas (2026-04-15)
- Only AI image generator that produces LTX-compatible output
- 54GB BF16 model at $UGC_MODELS_DIR/qwen-image-2512/
- Prompt style: "UGC creator, real photograph, natural skin with pores"
- Status: **Validated.** Characters Drew, Elena, Sophia, Marcus, James, Aisha created with it.

### Multi-Angle Reference Photos (2026-04-16)
- `generate_multiangle_qwen.py` creates 6 camera-facing angle variants per character
- Angles stored in `personas/<slug>/qwen_multiangle/`
- Per-segment angle selection not yet automated in video_producer
- Status: **Images generated, pipeline integration pending.**

### Character Voice System (2026-04-17)
- The male reference clip for all male characters — 11/12 GOOD first try
- sohee (Qwen3-TTS) for female — consistent FAIR, occasional GOOD
- VoiceDesign API available for custom voice creation
- Key insight: reference audio ENERGY matters more than timbre
- Status: **In production.**

---

## Ready to Implement

### Mouth Click Detection (2026-04-18)
- Three-layer scorer: Essentia ClickDetector + Spectral Flux + Median Filter
- All dependencies installed, ~55ms per clip on CPU
- Thresholds calibrated: >0.30 = reject
- Integration point: `score_mouth_clicks()` → `quality_gate.py`
- Status: **Code designed, not integrated.** See log.md Apr 18 entry.

### Brand Name WER Whitelist
- Whisper splits compound brand names ("BrandName" → "Brand Name") → false 50% WER
- Need whitelist of brand names to normalize before WER comparison
- Status: **Known bug, easy fix.**

---

## Experimental — Diverging from Known Good

> These are experiments that produced interesting results but are NOT part of
> the production pipeline. They're documented as reference for future exploration.

### Second Real-Photo Persona: Per-Persona Tuning (2026-04-13)
- A second persona (AI-upscaled real photo) needed ag=5.0, idg=1.2 (vs the primary persona's ag=3.5, idg=1.5)
- At the primary persona's settings, lip sync dropped off after segment 1-2
- Seed variance caused ~30% failure rate — K=2 beam search needed
- **Takeaway:** V13 settings aren't universal. Each persona may need audio_guidance tuning.
- **NOT best practice yet** — needs systematic A/B across more personas.

### Alternative Energetic Reference Audio (2026-04-18)
- Tested non-standard voice references with poor quality results
- A different energetic film-dialogue reference didn't translate to better lip sync
- Quality was noticeably below the male reference baseline
- **Takeaway:** Reference point only. The original male clip remains the proven male voice reference.
- **Do NOT adopt as best practice.** Continue experimenting but don't replace the proven reference.

### VBVR LoRA Chaining (2026-04-13, Archived)
- VBVR LoRA improves seam flow (0.06-0.25 vs 2.0-3.1 baseline)
- But chaining still causes identity collapse by seg 4-5 (57% face preservation)
- Hybrid reset: 66% but creates massive seam at reset point (flow 3.6)
- **Verdict:** Not worth pursuing for chaining. Always-original + hard cuts is better.

### Anne — Gemini Nano Banana Pro Image (2026-04-13, Failed)
- AI-generated image with very photorealistic appearance
- Face animated (blinks, head movement) but NO lip sync
- Tested: audio_guidance 3.5/5.0/7.0, the primary persona's voice reference, Edge-TTS, neutral expression
- Heavy noise preprocessing (sigma=25) made it WORSE (poster mode)
- **Verdict:** Gemini images don't work for lip sync. Use Qwen-Image-2512 instead.

### CodeFormer Inter-Segment Enhancement (2026-04-13, Not Tested)
- Concept: face-restore last frame before feeding to next segment during chaining
- CodeFormer installed at $UGC_PROJECTS_DIR/CodeFormer/ with weights
- Never tested because always-original strategy made it unnecessary
- **May revisit** if chaining becomes viable again.

### First+Last Frame Keyframe Anchoring (2026-04-13, Research Only)
- ComfyUI LTXVLoopingSampler supports temporal tile conditioning
- Would require switching from direct ID-LoRA inference to ComfyUI — major change
- ID-LoRA `inference_two_stage_hq.py` only supports `--first-frame`, not `--last-frame`
- **Too large an architecture change for current pipeline.**

---

## Audio Guidance Comparison (2026-04-18, Pending Review)
- Testing ag=2.5 vs ag=3.5 to reduce mouth clicking artifacts
- Samples exported for human review
- Lower ag may reduce artifacts but could also reduce lip sync quality
- **Awaiting subjective review before changing defaults.**
