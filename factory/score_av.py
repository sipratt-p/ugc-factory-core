"""Audio-Video scoring tool for autoresearch.

Comprehensive clip evaluation: intelligibility, naturalness, speaker similarity,
AV sync, pacing. Returns structured JSON that Claude can optimize against.

All scoring models run on the eval GPU (config.EVAL_GPU) to avoid
conflicting with video generation on the video GPU.

Karpathy pattern: give the optimizer component scores, not just one scalar.
If quality drops, Claude can diagnose whether the problem was pronunciation,
identity, pacing, naturalness, or sync.

Scoring components:
  - WhisperX WER (intelligibility)
  - UTMOS (perceptual naturalness — neural MOS)
  - NISQA decomposition (Noisiness, Coloration/wiriness, Discontinuity, Loudness)
  - ECAPA-TDNN speaker embeddings (voice identity vs reference)
  - Pacing analysis (words/sec, pauses, rate consistency)
  - Heuristic naturalness (HF ratio, pitch variation, AI voice signals)
"""

import json
import logging
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, asdict

from . import config

logger = logging.getLogger(__name__)

# Force scoring models to the eval GPU
SCORE_GPU = config.EVAL_GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(SCORE_GPU))

# Lazy-loaded singletons for expensive models
_ecapa_classifier = None
_nisqa_model = None


def extract_audio_wav(video_path: Path, output_path: Path) -> Path:
    """Extract audio from video as 16kHz mono WAV."""
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path


# Transcription server: any HTTP service exposing POST /transcribe that accepts
# multipart 'audio' + form fields 'target_text'/'language' and returns JSON with
# transcript, wer, word_count, word_timestamps. Override with WHISPER_URL.
WHISPER_API_URL = os.environ.get("WHISPER_URL", "http://localhost:8765")


def score_intelligibility(audio_path: Path, target_script: str) -> dict:
    """Transcription + WER via the whisper HTTP server (WHISPER_URL).

    Falls back to local WhisperX if the server is unreachable.
    """
    # Try the whisper server first (fast, and no model load in this process)
    try:
        import requests
        resp = requests.post(
            f"{WHISPER_API_URL}/transcribe",
            files={"audio": open(str(audio_path), "rb")},
            data={"target_text": target_script, "language": "en"},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "wer": data.get("wer"),
                "transcript": data.get("transcript", "")[:200],
                "missing_words": [],
                "repeated_words": [],
                "word_count": data.get("word_count", 0),
                "word_timestamps": data.get("word_timestamps", []),
                "transcription_method": "whisper_server",
            }
    except Exception as e:
        logger.warning(f"whisper server unavailable ({e}), falling back to local WhisperX")

    # Fallback: local WhisperX
    try:
        import whisperx

        model = whisperx.load_model("base", device="cpu", compute_type="int8")
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, language="en")

        transcript = " ".join(seg["text"].strip() for seg in result["segments"])

        ref_words = target_script.lower().split()
        hyp_words = transcript.lower().split()

        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, ref_words, hyp_words)
        matches = sum(block.size for block in matcher.get_matching_blocks())
        wer = 1.0 - (matches / max(len(ref_words), 1))

        word_timestamps = []
        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                word_timestamps.append({
                    "word": word.get("word", ""),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                })

        ref_set = set(ref_words)
        hyp_set = set(hyp_words)
        missing = list(ref_set - hyp_set)[:5]
        repeated = [w for w in hyp_words if hyp_words.count(w) > ref_words.count(w)][:5]

        return {
            "wer": round(wer, 3),
            "transcript": transcript[:200],
            "missing_words": missing,
            "repeated_words": list(set(repeated)),
            "word_count": len(hyp_words),
            "word_timestamps": word_timestamps,
        }
    except Exception as e:
        logger.warning(f"WhisperX failed: {e}")
        return {"wer": None, "error": str(e)}


def score_pacing(word_timestamps: list, audio_duration: float) -> dict:
    """Compute pacing metrics from word timestamps."""
    try:
        if not word_timestamps:
            return {"words_per_sec": None}

        word_count = len(word_timestamps)
        wps = word_count / max(audio_duration, 0.1)

        # Pause detection from word gaps
        pauses = []
        for i in range(1, len(word_timestamps)):
            gap = word_timestamps[i].get("start", 0) - word_timestamps[i-1].get("end", 0)
            if gap > 0.15:  # 150ms+ = a pause
                pauses.append(gap)

        mean_pause = sum(pauses) / max(len(pauses), 1) if pauses else 0

        # Rate consistency (variance in word duration)
        durations = [w.get("end", 0) - w.get("start", 0) for w in word_timestamps if w.get("end", 0) > w.get("start", 0)]
        if durations:
            import statistics
            rate_variance = statistics.variance(durations) if len(durations) > 1 else 0
            # Lower variance = more consistent pacing (better)
            rate_score = max(0, 1.0 - rate_variance * 10)
        else:
            rate_score = 0

        # Pause score: penalize zero pauses (rushed) and excessive pauses
        ideal_pauses = max(1, word_count // 15)  # roughly 1 pause per 15 words
        pause_score = 1.0 - min(1.0, abs(len(pauses) - ideal_pauses) / max(ideal_pauses, 1))

        return {
            "words_per_sec": round(wps, 2),
            "pause_count": len(pauses),
            "mean_pause_ms": round(mean_pause * 1000, 0),
            "rate_score": round(rate_score, 3),
            "pause_score": round(pause_score, 3),
        }
    except Exception as e:
        logger.warning(f"Pacing analysis failed: {e}")
        return {"words_per_sec": None, "error": str(e)}


def score_utmos(audio_path: Path) -> dict:
    """Neural MOS prediction using UTMOS — captures perceptual naturalness
    that spectral analysis misses. Trained on human MOS ratings."""
    try:
        import torch
        import soundfile as sf

        data, sr = sf.read(str(audio_path))
        # Resample to 16kHz if needed
        if sr != 16000:
            import subprocess
            resampled = audio_path.parent / "utmos_16k.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", str(resampled),
            ], check=True, capture_output=True)
            data, sr = sf.read(str(resampled))
            resampled.unlink(missing_ok=True)

        wav_tensor = torch.from_numpy(data).float().unsqueeze(0)
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

        with torch.no_grad():
            score = predictor(wav_tensor, sr)

        return {"utmos": round(score.item(), 3)}
    except Exception as e:
        logger.warning(f"UTMOS failed: {e}")
        return {"utmos": None, "utmos_error": str(e)}


def _get_nisqa_model():
    """Lazy-load NISQA model singleton."""
    global _nisqa_model
    if _nisqa_model is None:
        import sys
        nisqa_path = str(config.NISQA_DIR)
        if nisqa_path not in sys.path:
            sys.path.insert(0, nisqa_path)
        from nisqa.NISQA_model import nisqaModel
        _nisqa_model = nisqaModel({
            "mode": "predict_file",
            "pretrained_model": str(config.NISQA_DIR / "weights" / "nisqa.tar"),
            "deg": "/dev/null",  # placeholder, overridden per-call
            "output_dir": None,
            "tr_bs_val": 1,
            "tr_num_workers": 0,
            "ms_channel": None,
        })
    return _nisqa_model


def score_nisqa(audio_path: Path) -> dict:
    """NISQA audio quality decomposition.

    Returns 5 scores (1-5 scale, higher = better):
      - mos_pred: overall predicted MOS
      - noi_pred: Noisiness (low noise = high score)
      - dis_pred: Discontinuity (smooth = high score)
      - col_pred: Coloration — THIS IS THE "WIRINESS" SIGNAL
      - loud_pred: Loudness appropriateness
    """
    try:
        import sys
        # Force local NISQA clone over pip version (pip version has dependency issues)
        nisqa_path = str(config.NISQA_DIR)
        if nisqa_path not in sys.path:
            sys.path.insert(0, nisqa_path)
        # Remove cached pip nisqa module if loaded
        for key in list(sys.modules.keys()):
            if key.startswith("nisqa") and nisqa_path not in str(sys.modules[key].__file__ if hasattr(sys.modules[key], '__file__') and sys.modules[key].__file__ else ""):
                del sys.modules[key]
        from nisqa.NISQA_model import nisqaModel
        import pandas as pd

        # NISQA needs to re-initialize datasets for each file, so we create
        # a fresh instance per call (the model weights are cached by PyTorch)
        model = nisqaModel({
            "mode": "predict_file",
            "pretrained_model": str(config.NISQA_DIR / "weights" / "nisqa.tar"),
            "deg": str(audio_path),
            "output_dir": None,
            "tr_bs_val": 1,
            "tr_num_workers": 0,
            "ms_channel": None,
        })
        df = model.predict()

        return {
            "nisqa_mos": round(float(df["mos_pred"].values[0]), 3),
            "nisqa_noisiness": round(float(df["noi_pred"].values[0]), 3),
            "nisqa_discontinuity": round(float(df["dis_pred"].values[0]), 3),
            "nisqa_coloration": round(float(df["col_pred"].values[0]), 3),
            "nisqa_loudness": round(float(df["loud_pred"].values[0]), 3),
        }
    except Exception as e:
        logger.warning(f"NISQA scoring failed: {e}")
        return {"nisqa_mos": None, "nisqa_error": str(e)}


def _get_ecapa_classifier():
    """Lazy-load ECAPA-TDNN speaker encoder singleton."""
    global _ecapa_classifier
    if _ecapa_classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        # CUDA_VISIBLE_DEVICES is already set to SCORE_GPU at module level,
        # so the only visible GPU is device 0 in this process.
        _ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda:0"},
        )
    return _ecapa_classifier


def get_ecapa_embedding(audio_path: Path):
    """Extract ECAPA-TDNN speaker embedding from audio file.

    Returns a 192-dim embedding tensor (or None on failure).
    Useful for building persona centroids and cross-clip comparison.
    """
    try:
        import torch
        import soundfile as sf

        data, sr = sf.read(str(audio_path))
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampled = audio_path.parent / "_ecapa_16k.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", str(resampled),
            ], check=True, capture_output=True)
            data, sr = sf.read(str(resampled))
            resampled.unlink(missing_ok=True)

        import numpy as np
        signal = torch.from_numpy(np.array(data, dtype=np.float32)).unsqueeze(0)
        classifier = _get_ecapa_classifier()
        embedding = classifier.encode_batch(signal)
        return embedding.squeeze()  # [192]
    except Exception as e:
        logger.warning(f"ECAPA embedding extraction failed: {e}")
        return None


def score_speaker_ecapa(audio_path: Path, reference_path: Path) -> dict:
    """Speaker similarity using ECAPA-TDNN embeddings (replaces spectral fallback).

    Compares generated clip audio against reference audio using cosine similarity
    of 192-dim ECAPA speaker embeddings. This is a proper speaker verification
    metric, not a spectral heuristic.
    """
    try:
        import torch

        gen_emb = get_ecapa_embedding(audio_path)
        ref_emb = get_ecapa_embedding(reference_path)

        if gen_emb is None or ref_emb is None:
            return {"speaker_cos": None, "method": "ecapa_failed"}

        cos_sim = torch.nn.functional.cosine_similarity(
            gen_emb.unsqueeze(0), ref_emb.unsqueeze(0)
        ).item()

        return {
            "speaker_cos": round(cos_sim, 4),
            "method": "ecapa_tdnn",
        }
    except Exception as e:
        logger.warning(f"ECAPA speaker similarity failed: {e}")
        return {"speaker_cos": None, "method": "ecapa_error", "error": str(e)}


def score_speaker_consistency(audio_paths: list[Path]) -> dict:
    """Cross-clip speaker consistency using ECAPA centroid analysis.

    Builds a centroid from all clip embeddings and measures how tightly
    clustered they are. Lower spread = more consistent voice identity.

    Returns:
      - centroid_distances: per-clip cosine to centroid
      - pairwise_min: worst pairwise similarity (the "weakest link")
      - spread: std of centroid distances (lower = better)
    """
    try:
        import torch
        import numpy as np

        embeddings = []
        for ap in audio_paths:
            emb = get_ecapa_embedding(ap)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < 2:
            return {"speaker_consistency": None, "error": "need >=2 clips"}

        stack = torch.stack(embeddings)  # [N, 192]
        centroid = stack.mean(dim=0, keepdim=True)  # [1, 192]

        # Centroid distances
        centroid_cos = torch.nn.functional.cosine_similarity(stack, centroid.expand_as(stack))
        centroid_dists = centroid_cos.tolist()

        # Pairwise similarities
        pairwise = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = torch.nn.functional.cosine_similarity(
                    embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
                ).item()
                pairwise.append(sim)

        return {
            "centroid_distances": [round(d, 4) for d in centroid_dists],
            "centroid_mean": round(float(np.mean(centroid_dists)), 4),
            "centroid_spread": round(float(np.std(centroid_dists)), 4),
            "pairwise_min": round(min(pairwise), 4) if pairwise else None,
            "pairwise_mean": round(float(np.mean(pairwise)), 4) if pairwise else None,
            "method": "ecapa_centroid",
        }
    except Exception as e:
        logger.warning(f"Speaker consistency analysis failed: {e}")
        return {"speaker_consistency": None, "error": str(e)}


def score_naturalness(audio_path: Path) -> dict:
    """Audio quality metrics including AI-voice detection signals."""
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(audio_path))

        # Basic quality metrics
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
        silence_ratio = float(np.mean(np.abs(data) < 0.01))

        # Spectral flatness (closer to 1 = noise, closer to 0 = tonal/speech)
        from scipy.fft import fft
        spectrum = np.abs(fft(data[:sr * 2]))[:sr]
        log_spectrum = np.log(spectrum + 1e-10)
        spectral_flatness = float(np.exp(np.mean(log_spectrum)) / (np.mean(spectrum) + 1e-10))

        # Speech band energy
        freqs_per_bin = sr / len(spectrum)
        speech_start = int(100 / freqs_per_bin)
        speech_end = int(4000 / freqs_per_bin)
        speech_energy = float(np.sum(spectrum[speech_start:speech_end]**2))
        total_energy = float(np.sum(spectrum**2))
        speech_ratio = speech_energy / max(total_energy, 1e-10)

        # === AI-voice detection signals ===

        # High-frequency energy ratio (AI voices often have excessive high freq)
        hf_start = int(6000 / freqs_per_bin)
        hf_energy = float(np.sum(spectrum[hf_start:]**2))
        hf_ratio = hf_energy / max(total_energy, 1e-10)
        # Natural speech: hf_ratio < 0.15, AI/wiry: hf_ratio > 0.25

        # Pitch variation (monotone = AI-like)
        # Simple autocorrelation-based pitch tracking
        frame_size = int(sr * 0.03)  # 30ms frames
        hop_size = int(sr * 0.01)    # 10ms hop
        pitches = []
        for start_idx in range(0, len(data) - frame_size, hop_size):
            frame = data[start_idx:start_idx + frame_size]
            if np.max(np.abs(frame)) < 0.01:
                continue  # skip silence
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            # Find first peak after 2ms (500Hz max)
            min_lag = int(sr / 500)
            max_lag = int(sr / 60)  # 60Hz min
            if max_lag > len(autocorr):
                continue
            search = autocorr[min_lag:max_lag]
            if len(search) > 0:
                peak_idx = np.argmax(search) + min_lag
                if autocorr[peak_idx] > 0.3 * autocorr[0]:
                    pitch = sr / peak_idx
                    if 60 < pitch < 500:
                        pitches.append(pitch)

        if len(pitches) > 5:
            import statistics
            pitch_mean = statistics.mean(pitches)
            pitch_std = statistics.stdev(pitches)
            pitch_variation = pitch_std / max(pitch_mean, 1)
            # Natural speech: pitch_variation > 0.15
            # Monotone/AI: pitch_variation < 0.08
        else:
            pitch_mean = 0
            pitch_std = 0
            pitch_variation = 0

        # Composite naturalness score
        has_speech = speech_ratio > 0.3 and rms > 0.01
        mos_estimate = 2.0 if has_speech else 1.0

        # Reward good characteristics
        if rms > 0.03 and silence_ratio < 0.5:
            mos_estimate += 0.5
        if spectral_flatness < 0.3:
            mos_estimate += 0.3
        if pitch_variation > 0.12:
            mos_estimate += 0.5  # natural pitch variation
        if hf_ratio < 0.2:
            mos_estimate += 0.3  # not wiry/metallic
        # Penalize bad characteristics
        if hf_ratio > 0.3:
            mos_estimate -= 0.5  # wiry/metallic sound
        if pitch_variation < 0.05 and has_speech:
            mos_estimate -= 0.5  # monotone

        return {
            "rms_level": round(rms, 4),
            "peak_level": round(peak, 4),
            "silence_ratio": round(silence_ratio, 3),
            "speech_ratio": round(speech_ratio, 3),
            "spectral_flatness": round(spectral_flatness, 4),
            "hf_ratio": round(hf_ratio, 4),
            "pitch_mean_hz": round(pitch_mean, 1),
            "pitch_variation": round(pitch_variation, 4),
            "has_speech": has_speech,
            "mos_estimate": round(max(1.0, min(5.0, mos_estimate)), 2),
            "ai_voice_signals": {
                "wiry": bool(hf_ratio > 0.25),
                "monotone": bool(pitch_variation < 0.08 and has_speech),
                "rushed": False,
            },
        }
    except Exception as e:
        logger.warning(f"Naturalness analysis failed: {e}")
        return {"mos_estimate": None, "error": str(e)}


def score_speaker_similarity(audio_path: Path, reference_path: Path) -> dict:
    """Speaker similarity — delegates to ECAPA-TDNN.

    This is the main entry point used by score_clip(). Uses proper speaker
    verification embeddings instead of the old spectral heuristic.
    """
    return score_speaker_ecapa(audio_path, reference_path)


def score_clip(
    video_path: Path,
    target_script: str,
    reference_audio: Path = None,
) -> dict:
    """Score a video clip on all audio-video quality metrics.

    Returns structured JSON with component scores for Claude to optimize against.
    """
    video_path = Path(video_path)
    work_dir = video_path.parent / "scoring"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Extract audio
    audio_path = work_dir / "audio.wav"
    try:
        extract_audio_wav(video_path, audio_path)
    except Exception as e:
        return {"error": f"Audio extraction failed: {e}"}

    # Get duration
    try:
        dur_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        duration = float(dur_result.stdout.strip())
    except Exception:
        duration = 0

    # Run all scoring components
    scores = {}

    # 1. Intelligibility (WhisperX)
    intel = score_intelligibility(audio_path, target_script)
    scores.update({k: v for k, v in intel.items() if k != "word_timestamps"})

    # 2. Pacing
    pacing = score_pacing(intel.get("word_timestamps", []), duration)
    scores.update(pacing)

    # 3. Naturalness (heuristic)
    natural = score_naturalness(audio_path)
    scores.update(natural)

    # 3b. UTMOS neural MOS (perceptual naturalness — catches what heuristics miss)
    utmos = score_utmos(audio_path)
    scores.update(utmos)

    # 3c. NISQA decomposition (Noisiness, Coloration/wiriness, Discontinuity, Loudness)
    # NISQA needs 48kHz audio (its mel-spec is tuned for that rate)
    audio_48k = work_dir / "audio_48k.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", str(audio_48k),
        ], check=True, capture_output=True)
        nisqa = score_nisqa(audio_48k)
        audio_48k.unlink(missing_ok=True)
    except Exception as e:
        nisqa = {"nisqa_mos": None, "nisqa_error": str(e)}
    scores.update(nisqa)

    # 4. Speaker similarity via ECAPA-TDNN (if reference provided)
    if reference_audio and Path(reference_audio).exists():
        speaker = score_speaker_similarity(audio_path, Path(reference_audio))
        scores.update(speaker)

    # 5. Composite score — UTMOS + NISQA + ECAPA
    components = []
    if scores.get("wer") is not None:
        components.append(("intelligibility", max(0, 1.0 - scores["wer"]) * 10, 0.20))
    if scores.get("utmos") is not None:
        components.append(("naturalness_utmos", scores["utmos"] * 2, 0.20))
    elif scores.get("mos_estimate") is not None:
        components.append(("naturalness_heuristic", scores["mos_estimate"] * 2, 0.10))
    if scores.get("nisqa_mos") is not None:
        # NISQA overall MOS (1-5 scale → 0-10)
        components.append(("nisqa_quality", scores["nisqa_mos"] * 2, 0.15))
    if scores.get("rate_score") is not None:
        components.append(("pacing", scores["rate_score"] * 10, 0.10))
    if scores.get("speaker_cos") is not None:
        components.append(("speaker_sim", scores["speaker_cos"] * 10, 0.15))
    if scores.get("has_speech"):
        components.append(("has_speech", 10.0, 0.05))

    if components:
        weighted_sum = sum(score * weight for _, score, weight in components)
        total_weight = sum(weight for _, _, weight in components)
        scores["audio_composite"] = round(weighted_sum / max(total_weight, 0.01), 2)
    else:
        scores["audio_composite"] = 0

    scores["duration_s"] = round(duration, 2)

    # Cleanup
    audio_path.unlink(missing_ok=True)

    return scores


def score_boundary(video_path: Path, cut_time_s: float, word_timestamps: list) -> dict:
    """Score the quality of a hard cut at the given time."""
    if not word_timestamps:
        return {"boundary_penalty": 0.5}

    # Find nearest word to the cut point
    min_dist = float("inf")
    nearest_word = None
    in_word = False

    for w in word_timestamps:
        start = w.get("start", 0)
        end = w.get("end", 0)
        if start <= cut_time_s <= end:
            in_word = True
            nearest_word = w
            break
        dist = min(abs(cut_time_s - start), abs(cut_time_s - end))
        if dist < min_dist:
            min_dist = dist
            nearest_word = w

    if in_word:
        # Cut landed inside a word — bad
        return {"boundary_penalty": 1.0, "issue": f"Cut chops word: {nearest_word.get('word', '?')}"}
    elif min_dist < 0.1:
        # Cut very close to a word boundary — marginal
        return {"boundary_penalty": 0.3, "issue": "Cut near word boundary"}
    else:
        # Cut in silence — good
        return {"boundary_penalty": 0.0, "issue": None}
