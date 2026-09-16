# ugc-factory-core

A talking-head video pipeline built on LTX-2.3 (22B, joint audio+video generation) with an identity LoRA, wrapped in measurement gates: a SyncNet lip-sync gate, a transcription (WER) gate, a pre-generation script linter, best-of-K seed banks, and an autoresearch loop that sweeps generation parameters and ranks the results. It was built to turn one reference photo plus a short voice reference into 20 to 90 second UGC-style vertical clips, and it is published so you can run the same gates against your own persona.

Everything here is the generic pipeline. The personas, voice references, music, rendered outputs and client scripts it was developed with have been removed (see "What is not here").

## What it does

```
persona dir ──► script lint ──► LTX-2.3 + ID-LoRA ──► SyncNet + WER gate ──► best-of-K pick ──► trim + concat + music
(photo, wav,    (blocks lip-     (one segment per      (reject POOR sync,     (disjoint seed      (audio hard cuts,
 scene bible)    sync killers)    ~5 s sentence)         retry with a fix)      banks per segment)  15 Mbps final)
                                                              │
                                              scoring fleet: WER, UTMOS, NISQA, ECAPA,
                                              ArcFace, DINOv2, RAFT, LPIPS, TOPIQ, SyncNet
                                                              │
                                              autoresearch loop + leaderboard (docs/)
```

The model generates speech and video jointly from a `[VISUAL] ... [SPEECH] ... [SOUNDS] ...` prompt conditioned on a first frame and a reference voice clip. There is no external TTS overlay in the main path; the reference audio steers voice character, the `[SPEECH]` text is what gets said, and every generated segment is measured before it is allowed into the final cut.

## Measured results

All numbers below come from files in this repository (`docs/`) or, where marked, from the run notes kept alongside the project. Nothing was re-run for publication.

### Autoresearch leaderboard (22 experiments, `docs/leaderboard.md`)

Single-clip parameter sweep, scored 1 to 10 on teeth, skin, face stability, sharpness and motion (visual-only scoring, V1). Baseline was identity guidance 2.0, LoRA strengths 0.25/0.5, rescale 0.45, audio guidance 7.0, 161 frames at 864x480 (960x1728 after the 2x stage).

| Rank | Experiment | Overall |
|------|-----------|---------|
| 1 | id_guidance_1.0 | 8.0 |
| 2 | lora_0.1_0.3 | 8.0 |
| 3 | id_guidance_1.5 | 8.0 |
| 4 | baseline_best (id_g 2.0) | 7.6 |
| 7 | frames_121 | 7.8 |
| 9 | id_guidance_3.0 | 7.4 |
| 11 | video_rescale_0.3 | 6.8 |
| 13 | id_guidance_0.5 | 6.6 |

Findings that became defaults: identity guidance 1.0 to 1.5 (0.5 distorts the mouth, 3.0 goes lifeless), distilled LoRA 0.1/0.3 (0.5/0.6 flattens skin), video rescale 0.45 (0.3 and 0.6 both distort teeth), 121 to 161 frames per segment (201 drops sharpness). Multi-clip: 3 clips hold quality (7.0), 6 clips degrade badly in clips 5 and 6 (5.0), and a 2-clip 201-frame build scored 8.0 with fewer seams. Generation time at full bf16 precision on one GPU was 236 s for a 161-frame clip and 707 s for a 3-clip 30 s build (`docs/research/experiments/results.tsv`).

### Later sweeps (`docs/research/experiments/`)

| Finding | Value | Source |
|---------|-------|--------|
| Word error rate on short prompts after audio guidance + full precision fix | 0.000 for under 20 words; 0.29 at 35 words | audio-sweep-v2, v3 |
| Descriptive `[SOUNDS]` tag vs baseline (heuristic MOS) | 3.3 vs 2.0 to 2.8 | audio-sweep-v2 |
| Single-clip UTMOS | 3.7 to 4.2 | v7 |
| ArcFace face consistency, single clip vs 3-clip vs 6-clip | 0.85 to 0.91 vs 0.77 to 0.82 vs 0.71 to 0.75 | metrics.md, v5, v7 |
| Sharpness loss within a 201-frame clip | 39 % | v6, v7 |
| Best 30 s multi-clip strategy | always re-condition on the original photo (combined 7.33, drift 0.830) vs last-frame chaining (7.19, 0.718) | v6 |
| Seam flow discontinuity, plain hard concat (V6, V7) vs handle-trim + re-encode (V8 to V9b) | 0.10 to 0.30 vs 1.4 to 10.8 | fleet-scoring-cross-version |
| First seam with last-frame conditioning | 0.08 to 0.24 (10x to 31x better than V9b) | fleet-scoring-cross-version |
| TOPIQ-face after full precision + tuned guidance | 0.548 (V6) to 0.755 (V8) | fleet-scoring-cross-version |

### Production pass rates through the SyncNet gate (from run notes, overnight batch 2026-04-18)

Four synthetic personas (Qwen-Image-2512 reference stills) run through `produce_video()` with the gate and adaptive retry. "Passed" means the segment cleared the SyncNet and transcription checks; the rest were kept as best-effort takes.

| Persona | Voice reference | Segments | Passed | Final length |
|---------|-----------------|----------|--------|--------------|
| A (male) | high-energy male reference clip | 12 | 11/12 | 39.6 s |
| B (female) | Qwen3-TTS `sohee` | 14 | 12/14 | not recorded |
| C (female) | Qwen3-TTS `sohee` | 14 | 12/14 | 40.2 s |
| D (male) | high-energy male reference clip | 14 | 13/14 | 42.9 s |

The same notes rank voice references by the SyncNet confidence they produce: the male reference clip averaged LSE-C 6.5 to 7.5 (GOOD), `sohee` 4.5 to 5.0 (FAIR), the Qwen3-TTS `aiden` voice 3.0 to 5.0, and a "sexy casual" TTS voice 1.0 to 4.0 with a high reseed rate. Reference audio energy and cadence mattered more than timbre. Settings are not universal either: a second real-photo persona needed audio guidance 5.0 and identity guidance 1.2 instead of the 3.5 / 1.5 defaults (`docs/research/roadmap.md`).

### Best-of-K (from run notes, 2026-06-10)

`produce_video(best_of=2)` generated two gated candidates per segment from disjoint seed banks and kept the higher SyncNet confidence. On a 4-segment reel the selected takes scored LSE-C 10.12 / 9.03 / 9.43 / 9.36 against a first-pass average of about 9.1, and the discarded candidates showed that single-pass luck spans a full LSE-C point. Cost was about 30 minutes instead of 17 for the same reel.

### Aggregate audio scores

`nisqa_ecapa_scores.json` holds per-clip NISQA decomposition (MOS, noisiness, discontinuity, coloration, loudness) and ECAPA speaker-similarity centroids for nine early multi-clip experiments, produced by `score_all_clips.py`. Clip ids are hashes; no media is included.

## Architecture: stages and gates

**Persona preparation** (`factory/persona_manager.py`, `factory/audio_utils.py`). A persona is a directory with `persona.json`, `reference.jpg` and `reference.wav`, optionally `scene_bible.json`. Reference images that are not portrait JPEGs are center-cropped to 3:4, given deterministic Gaussian noise (sigma 10) and saved at JPEG quality 70, because clean AI stills otherwise freeze; portrait JPEGs pass through untouched. Reference audio is normalized to 48 kHz stereo and trimmed to its loudest 5 s window. A scene bible (character, wardrobe, setting, camera, frozen `[SOUNDS]` tag) can be auto-described from the photo with Claude vision when `ANTHROPIC_API_KEY` is set; otherwise a generic bible is used.

**Script lint** (`factory/script_lint.py`, on by default in `produce_video`). Errors block generation before any GPU time: more than one sentence per segment, URLs or domains, "link in bio", and speech longer than the frame window at 2.5 words per second (121 frames at 24 fps is about 12 words; number words count double). Warnings: promotional phrasing, digits, all-caps tokens, more than 20 words.

**Generation** (`factory/autoresearch.py::generate_test_clip`, `factory/video_engine.py`). Each segment is one subprocess call to the ID-LoRA `inference_two_stage_hq.py` script with the V13 knobs from `factory/config.py`: identity guidance 1.5, audio guidance 3.5, video guidance 3.0, rescale 0.45, distilled LoRA 0.1 / 0.3, 864x480 base with 2x latent upscale, 24 fps, 129 frames (the producer defaults to 121). Output is re-encoded to 15 Mbps H.264. `factory/comfyui_workflow.py` builds the equivalent ComfyUI graphs (initial segment with audio conditioning, and plain talking head) for people who prefer that route.

**SyncNet scorer** (`factory/score_sync.py`), three levels:
1. Level 1, under a second: InsightFace 106-point landmarks give per-frame inner-lip aperture; a static mouth is flagged when the normalized aperture standard deviation is below 0.02.
2. Level 2, a few seconds: SyncNet v2 LSE-D / LSE-C computed on 224x224 face crops tracked from InsightFace keyframe detections, bypassing the slow S3FD detector. Labels: GOOD when LSE-C > 5.0 and LSE-D < 7.0, FAIR when > 3.0 and < 8.0, MEDIOCRE when > 1.5 and < 10.0, else POOR.
3. Level 3: both, plus the per-frame confidence curve and the worst contiguous region where sync breaks.

**Quality gate** (`factory/quality_gate.py`). `assess_segment` runs Level 3 sync scoring, transcribes the clip (an HTTP whisper server at `WHISPER_URL`, default `http://localhost:8765`, with local whisper-base as fallback), and computes WER against the script. Declared thresholds: `MIN_SYNCNET_LSE_C = 2.0`, `MAX_SYNCNET_LSE_D = 9.0`, `MAX_WER_THRESHOLD = 0.4`, `MIN_MOUTH_MOVEMENT = 0.05`. In the current code path the sync decision is label based: only a POOR label rejects (so MEDIOCRE passes); WER above 0.4, a truncated ending (last two script words missing) or a ghost echo (early speech repeated later) also reject. `generate_with_quality_gate` then diagnoses the failure and retries up to 5 times over the seeds `[42, 137, 256, 512, 1024]`: shorten the script by two words if truncated with more than 8 words, else add 32 frames (up to 161); restructure the sentence on ghost echo; reseed for low movement, low correlation or high WER. If nothing passes, the best attempt is kept and labeled best-effort.

**Best-of-K and assembly** (`factory/video_producer.py::produce_video`). With `best_of=K`, each segment runs the gate K times from disjoint seed banks (`[42, 137, 256]`, `[1337, 2025, 777]`, `[9001, 31337, 4242]`), and the winner is ranked by pass status, then SyncNet label (GOOD over FAIR), then raw LSE-C. Segments are trimmed with `trim`/`atrim` at the last detected silence start (never at an inter-word pause), hard-concatenated at 15 Mbps with the slow preset, and background music is added with `amerge` and a stream-copied video track so the final encode is the assembly encode.

**Scoring fleet** (`factory/score_av.py`, `score_visual.py`, `score_seam.py`, `score_advanced.py`, `score_temporal.py`, `score_sequence.py`, `endings.py`). Audio: WER, UTMOS (`tarepan/SpeechMOS`), NISQA decomposition, ECAPA-TDNN speaker similarity and cross-clip centroid spread, pacing; audio composite weights WER 0.20, UTMOS 0.20, NISQA 0.15, pacing 0.10, speaker 0.15, has-speech 0.05. Visual: ArcFace face consistency and identity match, Laplacian sharpness, DFT artifact ratio, mouth-region motion, eyebrow drift. Seams: face embedding delta, LPIPS, head-pose snap, loudness delta, NISQA delta, RAFT optical-flow discontinuity. Temporal: early vs late identity, sharpness and UTMOS. Sequence: 50 % mean shot + 25 % mean seam + 25 % cross-clip face consistency, with hard penalties (WER above 0.8 is -3.0, wiry NISQA coloration below 3.0 is -1.0, face drift above 0.3 is -1.0, and so on).

**Autoresearch loop** (`factory/autoresearch.py`, `factory/candidate_gen.py`, `factory/sequence_search.py`, `factory/wiki_hook.py`). `run_autoresearch.py` and `run_autoresearch_phase2.py` are the two phases that produced the 22-experiment leaderboard: a `TestConfig` grid, one clip per config, frames and audio metrics extracted for scoring, results written per test id and appended to a progress file. `run_autoresearch_v13.py` is the last versioned runner: K candidates per segment generated in parallel across two GPUs, a greedy lattice search (0.7 shot score + 0.3 transition score, with transition scored on the actual cut-point face embeddings and loudness), 6-frame video cross-fade with an audio hard cut, fade-out ending, full fleet scoring, and an append to the markdown log in `docs/research/`. The intermediate runners v2 to v12 are not included; the README and docs only depend on these three.

**Older ungated path** (`run.py`, `factory/orchestrator.py`, `factory/assembler.py`, `factory/subtitles.py`). The original CLI: persona + script JSON, sequential generation with the persona's bible, crossfade assembly, optional karaoke subtitles via faster-whisper and pysubs2. Kept because it is the simplest end-to-end example, but it does not run the gate.

## Running it with your own persona

1. Install the external dependencies below and `pip install -r requirements.txt`.
2. Create a persona directory:

   ```
   personas/my_persona/
     persona.json      copy from examples/persona_template/persona.json
     reference.jpg     a real, well-lit head-and-shoulders photo (portrait, 3:4 or taller)
     reference.wav     3 to 15 s of one speaker, 48 kHz stereo (the code normalizes if not)
     scene_bible.json  optional; copy from examples/persona_template/ and describe the photo
   ```

   `python -c "from factory.persona_manager import create_persona; ..."` does the same from a raw photo and audio file, including the optional Claude-vision bible.

3. Point the code at your machine with environment variables (all have defaults, see `factory/config.py`):

   | Variable | Meaning | Default |
   |----------|---------|---------|
   | `UGC_MODELS_DIR` | weights root (LTX checkpoint, ID-LoRA weights, SyncNet weights) | `~/models` |
   | `UGC_PROJECTS_DIR` | sibling checkouts: `ID-LoRA/`, `ComfyUI/` model folders, `NISQA/` | `~/projects` |
   | `UGC_ML_ENV_PYTHON` | interpreter used for the GPU subprocesses | `~/ml-env/bin/python` if present, else the current interpreter |
   | `UGC_VIDEO_GPU`, `UGC_VIDEO_GPU_2`, `UGC_EVAL_GPU`, `UGC_TTS_GPU` | CUDA indices; set all to 0 on one GPU | 0, 1, 1, 1 |
   | `UGC_RESULTS_DIR` | where autoresearch and the producer write | `outputs/autoresearch` |
   | `WHISPER_URL` | transcription server for the gate and scorer | `http://localhost:8765` |
   | `UGC_SYNCNET_WEIGHTS` | path to `syncnet_v2.model` | `$UGC_MODELS_DIR/syncnet_v2.model`, else Hugging Face download |
   | `ANTHROPIC_API_KEY` | optional, enables scene-bible description and script generation | unset |
   | `UGC_LLM_API_URL`, `UGC_LLM_MODEL` | optional local OpenAI-compatible LLM for `script_generator.generate_with_gemma` | `http://localhost:8000/v1/chat/completions` |

4. Produce a gated reel:

   ```bash
   python examples/run_persona_example.py --persona-dir personas/my_persona --best-of 2 --frames 121
   ```

   The example lints its four placeholder lines, runs `produce_video`, and prints the per-segment SyncNet label, LSE-C, transcript and pass/fail. Pass `--scripts my_lines.json` (a JSON list of strings, one sentence each) for your own copy, `--music path.wav` for a bed, `--gpu 0` to pick the generation GPU.

5. Sweep parameters instead: export `UGC_REF_IMAGE` and `UGC_REF_AUDIO`, then `bash run_autoresearch_all.sh` (phases 1 and 2), or set `UGC_PERSONA_DIR` and run `python run_autoresearch_v13.py` for the K=2 beam runner. Results land under `UGC_RESULTS_DIR/<test_id>/` with `result.json`, extracted frames and the re-encoded clip.

Script rules that the lint enforces and the experiments back up: one flowing sentence per segment (periods between phrases freeze the lips), about 7 to 12 words for a 121 to 129 frame window, no URLs or "link in bio", strong consonant openers, no negations in the visual prompt (they bleed into speech), spell numbers out.

## Requirements and external dependencies

Python packages are in `requirements.txt`. The heavy pieces are not pip installs:

- **LTX-2.3 22B dev checkpoint** (Lightricks), bf16, about 44 GB of VRAM at generation time in these runs (run notes). int8 quantization was tried and rejected for visible quality loss (`docs/research/overview.md`).
- **ID-LoRA for LTX-2.3**: the `ID-LoRA-2.3/scripts/inference_two_stage_hq.py` script and the `id-lora-talkvid` LoRA weights (a `celebvhq` variant is also referenced). This repo shells out to that script; identity guidance, audio guidance and CFG rescale come from it.
- **ComfyUI model folders** for the Gemma-3 12B text encoder, the LTX-2.3 spatial upscaler x2 1.1 and the 22B distilled LoRA 384 (paths in `factory/config.py`). ComfyUI itself is only needed if you use `factory/comfyui_workflow.py`.
- **SyncNet v2**: `syncnet_v2.model` weights plus the `syncnet_python` package from the original SyncNet repository; `score_sync.py` will try a Hugging Face download of the weights if none are found.
- **InsightFace `buffalo_l`** (with `onnxruntime-gpu`), **NISQA** (git checkout plus `weights/nisqa.tar`), **SpeechBrain ECAPA** (`spkrec-ecapa-voxceleb`), **UTMOS** via `torch.hub` (`tarepan/SpeechMOS`), **LPIPS**, torchvision **RAFT** and **DINOv2** weights, optional **pyiqa** (TOPIQ-face, MUSIQ) and **laion-clap**.
- **Transcription**: any HTTP server exposing `POST /transcribe` (multipart `audio`, form fields `target_text` and `language`, JSON back with `transcript`, `wer`, `word_count`, `word_timestamps`), or the local `openai-whisper` / `whisperx` fallbacks. The runs used faster-whisper large-v3-turbo behind such a server so the gate never loads a model next to the 22B video model (run notes).
- **Persona stills**: real photos work best. Of the AI image generators tried, only Qwen-Image-2512 produced faces that lip-synced under LTX; a Gemini-generated still animated (blinks, head motion) but never synced (`docs/research/roadmap.md`), and SDXL and Flux stills failed the same way (run notes).
- **Voice references**: any 3 to 15 s clip of one speaker. OmniVoice (zero-shot cloning), Qwen3-TTS voices and edge-tts were used to manufacture references when no recording existed; `factory/audio_utils.py::generate_voice_reference` wraps edge-tts.
- **ffmpeg** with libx264 and the `xfade`, `silencedetect`, `amerge` filters.

Hardware context: developed on a two-GPU workstation with generation on GPU 0 and all scorers on GPU 1. Nothing prevents a single-GPU run; set the four `UGC_*_GPU` variables to 0 and expect K=2 candidates to run sequentially.

## Lessons learned

- **The correlation lip-sync metric was lying.** The original mouth-motion vs audio-energy correlation scored 0.77 on segments with dead lip sync. SyncNet LSE-C/LSE-D replaced it as the primary gate in April 2026 (run notes; `docs/research/roadmap.md`).
- **Selecting best-of-K on SyncNet alone brings back lip smacking** (run notes, 2026-06-12). LTX generates a fresh audio track per take, and SyncNet only measures mouth-motion to audio correspondence. Mouth noise correlates with that correspondence, so a smacky take can outrank a clean one and did: a single-take seed 42 reel was clean, the `best_of=2` rebuild swapped two segments to smacky takes. The GOOD/FAIR label tiebreak does not fix it because a smacky take can be GOOD. Never rank takes on lip-sync score alone when audio is regenerated per take; the pending fix is a transient/click penalty measured in non-speech regions. Because every candidate take stays on disk under `UGC_RESULTS_DIR`, a bad selection can be repaired with pure ffmpeg reassembly, no regeneration.
- **Generative Foley models leak voices** (run notes, 2026-05-01). MMAudio and Audio-Omni were tried for background sound effects; both produced audible breath, mumbled speech or siren-like artifacts because their training corpora are speech-adjacent, and negative prompts could not suppress it. The recommendation is curated SFX libraries or a text-to-SFX API rather than more generative Foley variants.
- **Script structure is a lip-sync parameter.** Multi-sentence segments, "link in bio", URLs and negation words in the prompt all measurably broke sync or bled into speech; they are now lint errors.
- **Re-encoding around cuts costs more than the cut.** Plain hard concat kept seam flow discontinuity at 0.1 to 0.3; handle-trim plus re-encode pushed it to 2 to 11. Video cross-fade is acceptable, audio must be a hard cut, and the last encode should be the assembly encode.
- **Always re-condition on the original photo.** Last-frame chaining accumulates identity drift; a 60 s chain degrades visibly by clips 5 and 6.
- **Upscalers do not help.** SeedVR2 and CodeFormer added texture without moving sync or realism scores; the quality ceiling was image statistics, not resolution (run notes).
- **Shared environments rot.** A transformers major-version bump in a shared venv silently broke the vendored ID-LoRA encoder for weeks; the fix was a pinned interpreter passed through `UGC_ML_ENV_PYTHON` (run notes, 2026-06-10).

## What is not here

Removed for publication, deliberately:

- `personas/`: every reference photo, voice WAV and scene bible, including the author's own and the synthetic personas the pass rates above refer to.
- `voice_samples/`, `music/`, `outputs/`, `temp/`, the SQLite job database, model weights and partial downloads. No media files of any kind are in this repository.
- `creative_ads/`: client campaign scripts and unrelated creative side projects.
- About 30 one-off `test_*.py` persona and product scripts, the client production script and its ffmpeg rebuild, and the intermediate autoresearch runners v2 to v12. `examples/run_persona_example.py` is the generic replacement.
- The Flask dashboard (`web/`), which was hardcoded to the author's persona and product list.
- The internal research diary (`log.md`, about 1200 lines), the human feedback log and the creative side-project notes. The metric definitions, experiment write-ups and roadmap that back the numbers above are kept under `docs/research/`.
- Agent instruction files.

Experiment labels that named a client were generalized to the ad vertical (recruiting, health, seo); no score was altered. Placeholder scripts in the runners and the example are brand-free stand-ins for the originals.

## Repository layout

```
factory/            pipeline package (generation wrapper, gates, scorers, producer)
run.py              original CLI (ungated path)
run_autoresearch.py, run_autoresearch_phase2.py   the two leaderboard phases
run_autoresearch_v13.py                            K=2 beam runner
run_autoresearch_all.sh
rescore_all.py, score_all_clips.py                 batch re-scoring over past results
examples/           generic gated example + persona template
templates/          V1-era script templates (they predate the lint and will trip it; kept as format examples)
docs/leaderboard.md         22-experiment autoresearch leaderboard
docs/brainstorms/, docs/plans/   seam-experiment plan and the V8 sequence-search design
docs/research/      metrics reference, scoring stack, experiment write-ups, roadmap, results table
nisqa_ecapa_scores.json     aggregate NISQA + ECAPA scores for nine early experiments
```

Known rough edges: `factory/script_generator.py::generate_from_template` passes a `sounds` field that `ScriptScene` no longer accepts (the web UI had its own template path); `factory/quality_gate.py` declares LSE thresholds that the label-based decision does not read directly.

## License

MIT, see `LICENSE`.
