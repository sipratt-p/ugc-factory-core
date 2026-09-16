---
title: V11 Scoring Stack — Full Fleet
type: reference
created: 2026-04-07
---

# V11 Scoring Stack

Complete evaluation fleet for AI-generated talking head UGC videos.
Organized by category. All run locally on RTX 5090 (32GB) + RTX PRO 6000 (98GB).

## Visual Quality

| Metric | Model/Tool | What | VRAM | Status |
|---|---|---|---|---|
| VideoScore2 | Qwen2.5-VL-7B fine-tune | Multi-dim AIGC quality + CoT reasoning | ~16GB | Downloading |
| TOPIQ-face | pyiqa | Face-specific learned IQA (0-1) | ~1GB | ✅ Installed |
| MUSIQ | pyiqa | General image quality (0-100) | ~1GB | ✅ Installed |
| DOVER | dover pip | Fast technical + aesthetic score | <2GB | Installing |
| Laplacian sharpness | OpenCV | Blur detection | CPU | ✅ Integrated |
| DFT artifact ratio | numpy | Compression artifacts | CPU | ✅ Integrated |

## Face / Identity

| Metric | Model/Tool | What | VRAM | Status |
|---|---|---|---|---|
| ArcFace consistency | InsightFace buffalo_l | Face embedding drift across frames | CPU (onnx) | ✅ Integrated |
| ArcFace identity match | InsightFace | Generated vs reference face | CPU | ✅ Integrated |
| Face regions/eyebrow | InsightFace 106-point | Specific feature drift | CPU | ✅ Integrated |
| DINOv2 frame-to-frame | torchvision | Semantic drift (beyond face) | ~2GB | **Add next** |

## Audio Quality

| Metric | Model/Tool | What | VRAM | Status |
|---|---|---|---|---|
| NISQA decomposition | NISQA clone | MOS + Noisiness + Coloration + Discontinuity + Loudness | CPU | ✅ Integrated |
| UTMOS | SpeechMOS | Neural perceptual MOS | CPU | ✅ Integrated |
| ECAPA-TDNN | SpeechBrain | 192-dim speaker embeddings + centroid | ~1GB | ✅ Integrated |
| WhisperX WER | WhisperX | Word error rate | CPU | ✅ Integrated |
| CLAPScore | laion/clap | Text-audio semantic alignment | ~1GB | **Add next** |
| FAD | fadtk | Fréchet Audio Distance (realism) | CPU | Installing |

## AV Sync / Lip Sync

| Metric | Model/Tool | What | VRAM | Status |
|---|---|---|---|---|
| Landmark-audio correlation | InsightFace + scipy | Mouth movement vs audio energy | CPU | ✅ Integrated |
| SyncNet LSE-C/D | syncnet-python | Direct lip sync lag/confidence | ~1GB | Deferred (deps) |
| AGAV-Rater | VideoLLaMA2-Qwen2 | Joint AV quality + consistency | ~16GB | Available on HF |

## Seam / Transition Quality

| Metric | Model/Tool | What | VRAM | Status |
|---|---|---|---|---|
| ArcFace at boundary | InsightFace | Face identity jump at cuts | CPU | ✅ Integrated |
| LPIPS at boundary | lpips (AlexNet) | Perceptual visual distance | ~500MB | ✅ Integrated |
| Head pose delta | InsightFace kps | Head position snap | CPU | ✅ Integrated |
| Loudness delta | soundfile + numpy | Audio level jump | CPU | ✅ Integrated |
| NISQA delta | NISQA | Audio quality change at cuts | CPU | ✅ Integrated |
| Optical flow (RAFT) | torchvision | Motion discontinuity | ~2GB | **Add next** |

## Temporal / Drift

| Metric | Model/Tool | What | VRAM | Status |
|---|---|---|---|---|
| Within-clip identity drop | ArcFace early vs late | Identity degradation over time | CPU | ✅ Integrated |
| Within-clip sharpness slope | Laplacian | Sharpness degradation | CPU | ✅ Integrated |
| LPIPS frame-to-frame | lpips | Temporal flicker/jitter | ~500MB | **Add next** |
| DINOv2 consistency | torchvision | Semantic temporal stability | ~2GB | **Add next** |

## Integration Priority

1. **VideoScore2** — downloading now, highest value single addition
2. **DINOv2 drift** — already in torchvision, trivial to add
3. **RAFT optical flow at seams** — already in torchvision
4. **AGAV-Rater** — joint AV scoring, ~16GB, fits GPU:1
5. **CLAPScore** — text-audio alignment
6. **DOVER** — fast pre-screening gate
