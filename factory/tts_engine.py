"""TTS engine — generates a single continuous audio track from the full script.

One TTS pass = consistent voice identity across all scenes.
The audio is then sliced per-segment during video generation.
"""

import subprocess
import logging
from pathlib import Path

from . import config
from .models import Persona, Script

logger = logging.getLogger(__name__)


def generate_full_audio(
    script: Script,
    persona: Persona,
    job_dir: Path,
) -> Path:
    """Generate a single continuous audio track for the entire script.

    Concatenates all scene speech into one text block, generates with
    Chatterbox TTS in one pass for consistent voice identity.
    Returns path to the full audio WAV file.
    """
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Combine all scene speech into one text
    # Add natural pauses between scenes
    full_text = " ... ".join(scene.speech for scene in script.scenes)

    output_path = audio_dir / "full_script.wav"

    logger.info(f"Generating full TTS audio ({len(full_text)} chars, {script.num_scenes} scenes)...")

    voice_ref = persona.voice_ref_path
    if not voice_ref or not voice_ref.exists():
        # Use a default reference or skip voice cloning
        voice_ref = None

    _generate_with_chatterbox(full_text, voice_ref, output_path)

    logger.info(f"Full audio generated: {output_path}")
    return output_path


def _generate_with_chatterbox(
    text: str,
    voice_ref: Path | None,
    output_path: Path,
    gpu_id: int = config.TTS_GPU,
) -> Path:
    """Generate speech with Chatterbox Turbo."""
    voice_ref_arg = f"audio_prompt_path={repr(str(voice_ref))}," if voice_ref else ""

    script_code = f"""
import soundfile as sf
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
wav = model.generate(
    {repr(text)},
    {voice_ref_arg}
)
sf.write({repr(str(output_path))}, wav.cpu().squeeze().numpy(), model.sr)
print(f"Generated: {{wav.shape}}, sr={{model.sr}}, duration={{wav.shape[-1]/model.sr:.1f}}s")
"""
    env = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PATH": str(config.ML_ENV_PYTHON.parent) + ":/usr/bin:/bin",
        "HOME": str(Path.home()),
    }
    result = subprocess.run(
        [str(config.ML_ENV_PYTHON), "-c", script_code],
        env=env, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Chatterbox TTS failed: {result.stderr[-500:]}")
    logger.info(result.stdout.strip())
    return output_path
