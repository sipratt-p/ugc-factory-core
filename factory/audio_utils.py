"""Audio and image validation/normalization for the UGC pipeline.

ID-LoRA reads reference audio at native sample rate and computes
mel-spectrograms from it. 48kHz stereo produces the best lip sync
and voice quality. This module ensures all reference audio meets
that standard before it reaches the generation pipeline.
"""

import logging
import shutil
import subprocess
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 2
TARGET_DURATION_MIN = 3.0
TARGET_DURATION_MAX = 15.0
TARGET_IMAGE_RATIO = 3 / 4
IMAGE_RATIO_TOLERANCE = 0.05
DEFAULT_VOICE_FILES = (
    "female_voice_48k_jenny.wav",
    "female_voice_48k.wav",
    "male_voice_48k.wav",
)


def _probe_audio(path: Path) -> dict:
    """Get audio stream metadata via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries",
         "stream=sample_rate,channels,codec_name,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    import json
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    return streams[0] if streams else {}


def validate_reference_audio(path: Path) -> dict:
    """Check reference audio meets ID-LoRA requirements.

    Returns dict with 'valid' bool, 'warnings' list, and metadata.
    """
    warnings = []

    if not path.exists():
        return {"valid": False, "warnings": ["File does not exist"], "path": str(path)}

    info = _probe_audio(path)
    if not info:
        return {"valid": False, "warnings": ["Could not probe audio file"], "path": str(path)}

    sr = int(info.get("sample_rate", 0))
    channels = int(info.get("channels", 0))
    duration = float(info.get("duration", 0))

    if sr != TARGET_SAMPLE_RATE:
        warnings.append(f"Sample rate {sr}Hz, expected {TARGET_SAMPLE_RATE}Hz — will normalize")
    if channels != TARGET_CHANNELS:
        warnings.append(f"{channels} channel(s), expected {TARGET_CHANNELS} (stereo) — will normalize")
    if duration < TARGET_DURATION_MIN:
        warnings.append(f"Duration {duration:.1f}s is short (min {TARGET_DURATION_MIN}s)")
    if duration > TARGET_DURATION_MAX:
        warnings.append(f"Duration {duration:.1f}s is long (max {TARGET_DURATION_MAX}s)")

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "sample_rate": sr,
        "channels": channels,
        "duration": duration,
        "needs_normalization": sr != TARGET_SAMPLE_RATE or channels != TARGET_CHANNELS,
        "path": str(path),
    }


def normalize_reference_audio(input_path: Path, output_path: Path = None) -> Path:
    """Ensure reference audio is 48kHz stereo PCM WAV.

    If already correct format, copies as-is. Returns output path.
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_normalized.wav"

    info = validate_reference_audio(input_path)
    if not info.get("needs_normalization", True):
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        logger.info(f"Audio already 48kHz stereo: {input_path}")
        return output_path

    logger.info(f"Normalizing audio to 48kHz stereo: {input_path} → {output_path}")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-c:a", "pcm_s16le",
        str(output_path),
    ], check=True, capture_output=True)

    return output_path


def validate_reference_image(path: Path) -> dict:
    """Check reference image is suitable for ID-LoRA conditioning."""
    warnings = []

    if not path.exists():
        return {"valid": False, "warnings": ["File does not exist"], "path": str(path)}

    suffix = path.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        warnings.append(f"Unusual format {suffix} — .jpg or .png recommended")

    meta = _get_image_metadata(path)
    if meta:
        w = meta["width"]
        h = meta["height"]
        if w < 256 or h < 256:
            warnings.append(f"Image too small ({w}x{h}) — at least 512x512 recommended")
        if w == h and w in (512, 1024, 2048):
            warnings.append(f"Square {w}x{w} image — may be AI-generated. Real photos work better.")

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "path": str(path),
    }


def _get_image_metadata(path: Path) -> dict:
    """Read image size/format via PIL."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": (img.format or "").upper(),
            }
    except Exception:
        try:
            import json
            from . import config

            script = (
                "from PIL import Image; import json, sys; "
                "img = Image.open(sys.argv[1]); "
                "print(json.dumps({'width': img.width, 'height': img.height, "
                "'format': (img.format or '').upper()}))"
            )
            result = subprocess.run(
                [str(config.ML_ENV_PYTHON), "-c", script, str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout)
        except Exception:
            return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_image_needs_preprocess(path: Path) -> bool:
    """Return True when a first frame should be repaired before generation.

    We only preprocess wide/square or non-JPEG inputs. Already-good portrait JPEGs
    are copied through unchanged so we don't keep adding noise/compression.
    """
    meta = _get_image_metadata(path)
    if not meta:
        return True

    if meta["format"] != "JPEG":
        return True

    ratio = meta["width"] / max(meta["height"], 1)
    return ratio > TARGET_IMAGE_RATIO + IMAGE_RATIO_TOLERANCE


def prepare_reference_image(input_path: Path, output_path: Path) -> Path:
    """Materialize a generation-ready first frame at output_path."""
    if reference_image_needs_preprocess(input_path):
        return preprocess_reference_image(input_path, output_path)
    if input_path != output_path:
        shutil.copy2(input_path, output_path)
    return output_path


def preprocess_reference_image(input_path: Path, output_path: Path) -> Path:
    """Crop to portrait + add noise + JPEG compression for ID-LoRA first frame.

    Three transformations:
    1. Center-crop to 3:4 portrait if image is square/landscape (ID-LoRA works
       best with portrait-oriented close-ups matching the 480x864 generation size)
    2. Add Gaussian noise (sigma=10) — prevents frozen frames with AI-generated
       images which are too clean for the model (trained on real video frames)
    3. Save as JPEG quality 70 — adds compression artifacts the model expects

    Real photos are minimally affected. The added noise is deterministic per input
    image, so repeated repair of the same raw image does not introduce per-clip drift.

    See: https://github.com/Lightricks/LTX-2/issues/11
         https://civitai.com/articles/11073/ltx-image-to-video-tips
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(input_path).convert("RGB")
        w, h = img.size

        # Center-crop to 3:4 portrait if needed (matches the successful persona refs)
        target_ratio = TARGET_IMAGE_RATIO  # w/h
        current_ratio = w / h
        if current_ratio > target_ratio + IMAGE_RATIO_TOLERANCE:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
            logger.info(f"Cropped to portrait: {w}x{h} → {img.size[0]}x{img.size[1]}")

        # Add deterministic Gaussian noise so legacy raw images animate, but repeated
        # processing of the same input stays stable across clips.
        arr = np.array(img).astype(np.float32)
        seed = int.from_bytes(hashlib.sha256(input_path.read_bytes()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 10, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        Image.fromarray(arr).save(str(output_path), format="JPEG", quality=70)
        logger.info(f"Preprocessed image (crop+noise+JPEG): {input_path} → {output_path}")
        return output_path
    except Exception as e:
        try:
            from . import config

            script = """
from PIL import Image
import hashlib
import numpy as np
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
target_ratio = 3 / 4
tolerance = 0.05

img = Image.open(input_path).convert("RGB")
w, h = img.size
ratio = w / h
if ratio > target_ratio + tolerance:
    new_w = int(h * target_ratio)
    left = (w - new_w) // 2
    img = img.crop((left, 0, left + new_w, h))

arr = np.array(img).astype(np.float32)
seed = int.from_bytes(hashlib.sha256(open(input_path, "rb").read()).digest()[:8], "big")
rng = np.random.default_rng(seed)
noise = rng.normal(0, 10, arr.shape)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
Image.fromarray(arr).save(output_path, format="JPEG", quality=70)
"""
            subprocess.run(
                [str(config.ML_ENV_PYTHON), "-c", script, str(input_path), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Preprocessed image via ml-env fallback: {input_path} → {output_path}")
            return output_path
        except Exception as fallback_error:
            logger.warning(
                f"Image preprocessing failed ({e}); ml-env fallback failed ({fallback_error}), using original"
            )
            shutil.copy2(input_path, output_path)
            return output_path


def generate_voice_reference(
    gender: str = "female",
    output_path: Path = None,
    text: str = None,
) -> Path:
    """Generate a voice reference clip via edge-tts for personas without one.

    Args:
        gender: "male" or "female"
        output_path: Where to save the 48kHz stereo WAV
        text: What the voice should say (generic speech sample if None)
    """
    import asyncio

    voices = {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
    }
    voice = voices.get(gender, voices["female"])

    if text is None:
        text = (
            "Hello, welcome to my channel. Today I want to share something "
            "important with you about health and wellness."
        )

    if output_path is None:
        output_path = Path("/tmp/voice_reference.wav")

    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

    mp3_path = output_path.with_suffix(".mp3")

    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate="-5%")
        await communicate.save(str(mp3_path))

    asyncio.run(_generate())

    # Convert to 48kHz stereo WAV, trim to 5s
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", str(TARGET_SAMPLE_RATE), "-ac", str(TARGET_CHANNELS),
        "-c:a", "pcm_s16le", "-t", "5",
        str(output_path),
    ], check=True, capture_output=True)
    mp3_path.unlink(missing_ok=True)

    logger.info(f"Generated {gender} voice reference: {output_path}")
    return output_path


def resolve_reference_audio_path(persona_dir: Path, preferred: Path = None) -> Path | None:
    """Pick the best available speaker reference for a persona.

    If the configured reference is just a generic/default placeholder, prefer any
    persona-specific alternative audio that differs from the defaults.
    """
    from . import config

    defaults_dir = config.PERSONAS_DIR / "_defaults"
    default_hashes = set()
    for name in DEFAULT_VOICE_FILES:
        candidate = defaults_dir / name
        if candidate.exists():
            try:
                default_hashes.add(_sha256_file(candidate))
            except OSError:
                pass

    ordered = []
    for candidate in (
        Path(preferred) if preferred else None,
        persona_dir / "reference_original_48k.wav",
        persona_dir / "voice_reference_clean.wav",
        persona_dir / "reference.wav",
    ):
        if candidate and candidate.exists():
            ordered.append(candidate)

    unique_candidates = []
    seen = set()
    for candidate in ordered:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        try:
            if _sha256_file(candidate) not in default_hashes:
                return candidate
        except OSError:
            continue

    if unique_candidates:
        return unique_candidates[0]

    for name in DEFAULT_VOICE_FILES:
        candidate = defaults_dir / name
        if candidate.exists():
            return candidate

    return None


def trim_to_sweet_spot(
    input_path: Path,
    output_path: Path = None,
    target_duration: float = 5.0,
) -> Path:
    """Trim audio to the best window of target_duration seconds.

    Selects the loudest/most speech-active segment. If already shorter
    than target, returns as-is with a warning.
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_trimmed.wav"

    info = _probe_audio(input_path)
    duration = float(info.get("duration", 0))

    if duration <= target_duration + 0.5:
        if duration < 3.0:
            logger.warning(f"Audio very short ({duration:.1f}s) — may affect voice quality")
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return output_path

    # Find loudest window using RMS energy
    try:
        import numpy as np
        import soundfile as sf
        data, sr = sf.read(str(input_path))
        if data.ndim > 1:
            data = data.mean(axis=1)  # mono for analysis

        window_samples = int(target_duration * sr)
        if len(data) <= window_samples:
            shutil.copy2(input_path, output_path)
            return output_path

        # Sliding window RMS
        step = sr // 4  # 0.25s steps
        best_start = 0
        best_rms = 0
        for start in range(0, len(data) - window_samples, step):
            window = data[start:start + window_samples]
            rms = float(np.sqrt(np.mean(window ** 2)))
            if rms > best_rms:
                best_rms = rms
                best_start = start

        start_time = best_start / sr
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-ss", str(start_time), "-t", str(target_duration),
            "-c:a", "pcm_s16le",
            str(output_path),
        ], check=True, capture_output=True)
        logger.info(f"Trimmed audio to {target_duration}s starting at {start_time:.1f}s (RMS={best_rms:.4f})")
    except Exception as e:
        # Fallback: just take the first N seconds
        logger.warning(f"Smart trim failed ({e}), taking first {target_duration}s")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-t", str(target_duration), "-c:a", "pcm_s16le",
            str(output_path),
        ], check=True, capture_output=True)

    return output_path
