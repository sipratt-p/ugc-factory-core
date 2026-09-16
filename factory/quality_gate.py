"""Quality gate for UGC segment generation.

Auto-detects lip sync failures, speech truncation, ghost echoes,
and low mouth movement. Regenerates failed segments with different seeds.
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("quality_gate")


@dataclass
class SegmentQuality:
    """Quality assessment for a single generated segment."""
    clip_path: Path
    lip_sync_corr: float
    mouth_movement: float
    transcript: str
    script: str
    trailing_silence: float
    ghost_echo: bool
    speech_truncated: bool
    passes: bool
    failure_reason: str = ""
    syncnet_lse_c: float = 0.0       # higher = better sync (best-of-K selector)
    syncnet_quality: str = "UNKNOWN"


# Thresholds
MIN_LIP_SYNC_CORR = 0.1      # old metric, kept as secondary
MIN_MOUTH_MOVEMENT = 0.05     # 106-point lip aperture
MIN_SYNCNET_LSE_C = 2.0       # SyncNet confidence (higher = better sync)
MAX_SYNCNET_LSE_D = 9.0       # SyncNet distance (lower = better sync)
MAX_WER_THRESHOLD = 0.4       # word error rate
MIN_TRAILING_SILENCE = 0.05
MAX_SEEDS_TO_TRY = 5
SEED_CANDIDATES = [42, 137, 256, 512, 1024]
USE_SYNCNET = True             # Use SyncNet as primary lip sync scorer


def _get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _transcribe(clip_path: Path) -> dict:
    """Transcribe a clip via the whisper HTTP server (default
    http://localhost:8765, env WHISPER_URL to relocate); falls back to
    local whisper-base if the server is down.

    The server beats the old per-call whisper-base load on both accuracy
    (WER decisions) and VRAM (no model load contending with LTX mid-batch).
    """
    import os

    # Extract audio
    audio_path = clip_path.parent / "qg_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(clip_path), "-vn", "-ar", "16000", "-ac", "1",
         str(audio_path)],
        capture_output=True,
    )

    server = os.environ.get("WHISPER_URL", "http://localhost:8765")
    try:
        import requests
        resp = requests.post(
            f"{server}/transcribe",
            files={"audio": open(str(audio_path), "rb")},
            data={"language": "en"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Adapt to the openai-whisper result shape assess_segment expects:
        # ghost-echo detection reads segments[{start, end, text}] — synthesize
        # one per word from the server's word_timestamps.
        result = {
            "text": data.get("transcript", ""),
            "segments": [
                {"start": w["start"], "end": w["end"], "text": w["word"]}
                for w in data.get("word_timestamps", [])
            ],
        }
        audio_path.unlink(missing_ok=True)
        return result
    except Exception as e:
        logger.warning(f"    whisper server unavailable ({e}); using local whisper-base")

    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path), word_timestamps=True)
    audio_path.unlink(missing_ok=True)
    return result


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Simple WER calculation."""
    ref_words = reference.lower().strip().strip('"').replace(".", "").replace(",", "").split()
    hyp_words = hypothesis.lower().strip().strip('"').replace(".", "").replace(",", "").split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Simple Levenshtein at word level
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / max(m, 1)


def _detect_ghost_echo(whisper_result: dict) -> bool:
    """Detect if speech appears before 1.5s that matches text from later."""
    segments = whisper_result.get("segments", [])
    if len(segments) < 2:
        return False

    early_text = ""
    late_text = ""
    for seg in segments:
        if seg["start"] < 1.5:
            early_text += seg["text"].strip().lower() + " "
        elif seg["start"] >= 2.0:
            late_text += seg["text"].strip().lower() + " "

    if not early_text.strip() or not late_text.strip():
        return False

    # Check if early text appears as a substring of late text
    early_words = early_text.strip().split()
    if len(early_words) >= 2:
        early_phrase = " ".join(early_words)
        if early_phrase in late_text:
            return True

    return False


def _detect_truncation(script: str, transcript: str) -> bool:
    """Detect if the script was truncated — last words missing."""
    script_words = script.lower().strip().strip('"').replace(".", "").replace(",", "").split()
    trans_words = transcript.lower().strip().replace(".", "").replace(",", "").split()

    if not script_words or not trans_words:
        return False

    # Check if the last 2 words of the script appear in the transcript
    last_words = " ".join(script_words[-2:])
    trans_text = " ".join(trans_words)
    if last_words not in trans_text:
        # Last words missing — likely truncated
        return True

    return False


def _get_trailing_silence(clip_path: Path) -> float:
    """Get trailing silence duration in seconds."""
    dur = _get_duration(clip_path)
    result = subprocess.run(
        ["ffmpeg", "-i", str(clip_path), "-af", "silencedetect=n=-25dB:d=0.2",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = re.findall(r"silence_start: ([\d.]+)", result.stderr)
    if starts:
        return dur - float(starts[-1])
    return 0.0


def assess_segment(clip_path: Path, script: str, scores: dict) -> SegmentQuality:
    """Run all quality checks on a generated segment.

    Uses SyncNet as the primary lip sync scorer (if available),
    falling back to correlation-based scoring.
    """
    lip_sync_corr = scores.get("lip_sync_corr", 0)
    mouth_movement = scores.get("mouth_movement_range", 0)
    trailing = _get_trailing_silence(clip_path)

    # SyncNet scoring (primary)
    syncnet_lse_c = 0
    syncnet_lse_d = 99
    syncnet_quality = "UNKNOWN"
    mouth_static = False
    if USE_SYNCNET:
        try:
            from .score_sync import score_sync_full
            sync = score_sync_full(clip_path)
            syncnet_lse_c = sync.lse_c
            syncnet_lse_d = sync.lse_d
            syncnet_quality = sync.quality
            mouth_static = sync.is_static
            mouth_movement = sync.mouth_open_std  # use 106-point measurement
            logger.info(
                f"    SyncNet: LSE-C={syncnet_lse_c:.2f} LSE-D={syncnet_lse_d:.2f} "
                f"quality={syncnet_quality} mouth_static={mouth_static}"
            )
        except Exception as e:
            logger.warning(f"    SyncNet failed: {e}, falling back to correlation")

    # Transcribe
    whisper_result = _transcribe(clip_path)
    transcript = whisper_result.get("text", "").strip()

    # Detect issues
    ghost = _detect_ghost_echo(whisper_result)
    truncated = _detect_truncation(script, transcript)
    wer = _word_error_rate(script, transcript)

    # Determine pass/fail
    failures = []

    # Primary: SyncNet check
    if USE_SYNCNET and syncnet_quality != "UNKNOWN":
        if syncnet_quality == "POOR":
            failures.append(f"syncnet_POOR(LSE-C={syncnet_lse_c:.1f})")
            if mouth_static:
                failures.append("mouth_static")
        # FAIR and GOOD pass regardless of mouth_static — subtle movement is acceptable
    else:
        # Fallback: old metrics
        if lip_sync_corr < MIN_LIP_SYNC_CORR:
            failures.append(f"low_corr={lip_sync_corr:.3f}")
        if mouth_movement < MIN_MOUTH_MOVEMENT:
            failures.append(f"low_mouth={mouth_movement:.4f}")

    if ghost:
        failures.append("ghost_echo")
    if truncated:
        failures.append("speech_truncated")
    if wer > MAX_WER_THRESHOLD:
        failures.append(f"high_wer={wer:.2f}")

    passes = len(failures) == 0
    failure_reason = ", ".join(failures) if failures else ""

    return SegmentQuality(
        clip_path=clip_path,
        lip_sync_corr=lip_sync_corr,
        mouth_movement=mouth_movement,
        transcript=transcript,
        script=script,
        trailing_silence=trailing,
        ghost_echo=ghost,
        speech_truncated=truncated,
        passes=passes,
        failure_reason=failure_reason,
        syncnet_lse_c=syncnet_lse_c,
        syncnet_quality=syncnet_quality,
    )


@dataclass
class RetryStrategy:
    """What to change on retry based on failure diagnosis."""
    action: str  # "reseed", "shorten_script", "rewrite_script", "increase_frames"
    new_script: str | None = None
    new_num_frames: int | None = None
    reason: str = ""


def _diagnose_and_fix(quality: SegmentQuality, script: str, num_frames: int) -> RetryStrategy:
    """Diagnose WHY a segment failed and determine the fix.

    Returns a RetryStrategy with the appropriate corrective action.
    """
    # Priority 1: Speech truncated → script too long for frame window
    if quality.speech_truncated:
        word_count = len(script.strip().strip('"').split())
        if word_count > 8:
            # Shorten: remove the last 2 words and rephrase
            words = script.strip().strip('"').rstrip('.').split()
            shortened = " ".join(words[:-2]) + "."
            new_script = f'"{shortened}"'
            return RetryStrategy(
                action="shorten_script",
                new_script=new_script,
                reason=f"truncated at {word_count} words → shortened to {word_count - 2}",
            )
        elif num_frames < 161:
            # Already short script — increase frame count
            new_frames = min(161, num_frames + 32)  # bump by 32 (next 8n+1 step)
            return RetryStrategy(
                action="increase_frames",
                new_num_frames=new_frames,
                reason=f"script short ({word_count}w) but still truncated → {num_frames}f→{new_frames}f",
            )

    # Priority 2: Ghost echo → sentence structure issue
    if quality.ghost_echo:
        # Rewrite: reverse the sentence structure
        words = script.strip().strip('"').rstrip('.').split()
        if len(words) > 4:
            # Simple rewrite: move the first half to the end
            mid = len(words) // 2
            rewritten = " ".join(words[mid:] + words[:mid]) + "."
            new_script = f'"{rewritten}"'
            return RetryStrategy(
                action="rewrite_script",
                new_script=new_script,
                reason=f"ghost echo detected → restructured sentence",
            )

    # Priority 3: Low mouth movement → bad seed (just reseed)
    if quality.mouth_movement < MIN_MOUTH_MOVEMENT:
        return RetryStrategy(
            action="reseed",
            reason=f"low mouth movement ({quality.mouth_movement:.4f}) → trying different seed",
        )

    # Priority 4: Low correlation but decent mouth → timing/sync issue
    if quality.lip_sync_corr < MIN_LIP_SYNC_CORR:
        return RetryStrategy(
            action="reseed",
            reason=f"low sync correlation ({quality.lip_sync_corr:.3f}) → trying different seed",
        )

    # Priority 5: High WER → LTX garbled the speech
    if _word_error_rate(quality.script, quality.transcript) > MAX_WER_THRESHOLD:
        # Simplify vocabulary — replace multi-syllable words
        return RetryStrategy(
            action="reseed",
            reason=f"high WER → trying different seed (may need vocabulary simplification)",
        )

    # Default: just reseed
    return RetryStrategy(action="reseed", reason="general failure → reseed")


def generate_with_quality_gate(
    generate_fn,
    script: str,
    test_id_base: str,
    score_fn,
    num_frames: int = 129,
    seeds: list[int] = None,
    max_attempts: int = MAX_SEEDS_TO_TRY,
) -> tuple[Path, SegmentQuality, list[dict]]:
    """Generate a segment with adaptive quality gating.

    Diagnoses failures and adjusts parameters (script, frames, seed) before retrying.

    Args:
        generate_fn: callable(seed, test_id, script, num_frames) -> Path to clip
        script: the speech text for this segment
        test_id_base: base name for test IDs
        score_fn: callable(clip_path) -> dict with lip_sync_corr, mouth_movement_range
        num_frames: initial frame count
        seeds: list of seeds to try (default: SEED_CANDIDATES)
        max_attempts: max number of attempts total

    Returns:
        (best_clip_path, quality_assessment, attempt_log)
    """
    if seeds is None:
        seeds = list(SEED_CANDIDATES[:max_attempts])

    best_result = None
    best_quality = None
    attempt_log = []
    current_script = script
    current_frames = num_frames
    seed_idx = 0

    for attempt in range(max_attempts):
        seed = seeds[seed_idx % len(seeds)]
        test_id = f"{test_id_base}_a{attempt}_s{seed}"
        logger.info(f"  Attempt {attempt + 1}/{max_attempts} (seed {seed}, {len(current_script.split())}w, {current_frames}f)")

        clip_path = generate_fn(seed, test_id, current_script, current_frames)
        scores = score_fn(clip_path)
        quality = assess_segment(clip_path, current_script, scores)

        log_entry = {
            "attempt": attempt + 1,
            "seed": seed,
            "script": current_script,
            "num_frames": current_frames,
            "corr": quality.lip_sync_corr,
            "mouth": quality.mouth_movement,
            "passes": quality.passes,
            "failure_reason": quality.failure_reason,
            "transcript": quality.transcript,
        }

        logger.info(
            f"    corr={quality.lip_sync_corr:.3f} mouth={quality.mouth_movement:.4f} "
            f"pass={quality.passes} {quality.failure_reason}"
        )

        if quality.passes:
            log_entry["action"] = "PASSED"
            attempt_log.append(log_entry)
            logger.info(f"  PASSED on attempt {attempt + 1}")
            return clip_path, quality, attempt_log

        # Diagnose and determine fix
        strategy = _diagnose_and_fix(quality, current_script, current_frames)
        log_entry["action"] = f"{strategy.action}: {strategy.reason}"
        attempt_log.append(log_entry)

        logger.info(f"    Diagnosis: {strategy.action} — {strategy.reason}")

        # Apply the fix
        if strategy.action == "shorten_script" and strategy.new_script:
            current_script = strategy.new_script
            logger.info(f"    New script: {current_script}")
        elif strategy.action == "rewrite_script" and strategy.new_script:
            current_script = strategy.new_script
            logger.info(f"    Rewritten: {current_script}")
        elif strategy.action == "increase_frames" and strategy.new_num_frames:
            current_frames = strategy.new_num_frames
            logger.info(f"    Frames: {current_frames}")
        elif strategy.action == "reseed":
            seed_idx += 1  # move to next seed

        # Track best
        if best_quality is None or quality.lip_sync_corr > best_quality.lip_sync_corr:
            best_quality = quality
            best_result = clip_path

    logger.warning(
        f"  No attempt passed after {max_attempts} tries. "
        f"Using best: corr={best_quality.lip_sync_corr:.3f} ({best_quality.failure_reason})"
    )
    return best_result, best_quality, attempt_log
