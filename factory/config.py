"""Centralized configuration for the UGC Factory pipeline.

Every machine-specific location is an environment variable with a sensible
default, so the code carries no absolute paths:

  UGC_MODELS_DIR      model weights root                     (default ~/models)
  UGC_PROJECTS_DIR    sibling repos: ID-LoRA, ComfyUI, NISQA  (default ~/projects)
  UGC_ML_ENV_PYTHON   interpreter for GPU subprocesses        (default: ~/ml-env
                      if that venv exists, else the running interpreter)
  UGC_VIDEO_GPU / UGC_VIDEO_GPU_2 / UGC_EVAL_GPU / UGC_TTS_GPU
                      CUDA device indices (defaults 0 / 1 / 1 / 1). Set them
                      all to 0 on a single-GPU machine.
  UGC_RESULTS_DIR     autoresearch output root (default <repo>/outputs/autoresearch)
  NISQA_DIR           NISQA checkout (default <UGC_PROJECTS_DIR>/NISQA)
  WHISPER_URL         transcription server (default http://localhost:8765)
"""

import os
import sys
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = Path(os.environ.get("UGC_MODELS_DIR", Path.home() / "models")).expanduser()
PROJECTS_DIR = Path(os.environ.get("UGC_PROJECTS_DIR", Path.home() / "projects")).expanduser()

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"
PERSONAS_DIR = PROJECT_ROOT / "personas"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
MUSIC_DIR = PROJECT_ROOT / "music"
AUTORESEARCH_DIR = Path(os.environ.get("UGC_RESULTS_DIR", OUTPUTS_DIR / "autoresearch")).expanduser()

# GPU assignments. Developed on a two-GPU workstation: generation on GPU 0,
# scoring models on GPU 1 so they never contend with the 22B video model.
VIDEO_GPU = int(os.environ.get("UGC_VIDEO_GPU", "0"))      # Primary generation GPU
VIDEO_GPU_2 = int(os.environ.get("UGC_VIDEO_GPU_2", "1"))  # Secondary generation GPU (parallel beam search)
EVAL_GPU = int(os.environ.get("UGC_EVAL_GPU", "1"))        # Scoring/eval models when not generating
TTS_GPU = int(os.environ.get("UGC_TTS_GPU", "1"))          # TTS generation GPU
ALL_GPUS = sorted({VIDEO_GPU, VIDEO_GPU_2})                # GPUs used for parallel candidates

# Video generation settings (9:16 portrait for TikTok/Instagram)
# Stage 1 base resolution -> Stage 2 is 2x
VIDEO_HEIGHT_STAGE1 = 480   # -> 960 after 2x upscale
VIDEO_WIDTH_STAGE1 = 288    # -> 576 after 2x upscale (roughly 9:16)
# Two-stage: Stage 1 generates at base, Stage 2 upscales 2x
TWO_STAGE = True           # Enable latent upscale (1088x1920 output)
NUM_FRAMES = 129           # 5.4s at 24fps -- V13 proven (8n+1 rule: 129 = 16*8+1)
FRAME_RATE = 24.0
CLIP_DURATION = 5.375      # seconds per clip (129 frames at 24fps)
SEED = 42                  # consistent seed for visual continuity

# V13 proven generation settings -- locked after 88+ experiments
IDENTITY_GUIDANCE = 1.5
AUDIO_GUIDANCE = 3.5
VIDEO_GUIDANCE = 3.0
VIDEO_RESCALE = 0.45
DISTILLED_S1 = 0.1
DISTILLED_S2 = 0.3
GENERATION_HEIGHT = 864    # stage 1 height (stage 2 = 1728)
GENERATION_WIDTH = 480     # stage 1 width (stage 2 = 960)

# Audio settings -- ID-LoRA reads at native SR, 48kHz stereo produces best lip sync
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2         # stereo
REFERENCE_AUDIO_DURATION = 5.0  # seconds, proven sweet spot

# ID-LoRA model paths
IDLORA_REPO = PROJECTS_DIR / "ID-LoRA"
IDLORA_INFERENCE_SCRIPT = IDLORA_REPO / "ID-LoRA-2.3" / "scripts" / "inference_two_stage_hq.py"
IDLORA_TALKVID_WEIGHTS = MODELS_DIR / "id-lora-talkvid" / "lora_weights.safetensors"
IDLORA_CELEBVHQ_WEIGHTS = MODELS_DIR / "id-lora-celebvhq" / "lora_weights.safetensors"
IDLORA_DEFAULT_WEIGHTS = IDLORA_TALKVID_WEIGHTS  # can be overridden per persona

# LTX 2.3 model paths
LTX_CHECKPOINT = MODELS_DIR / "ltx-2.3" / "ltx-2.3-22b-dev.safetensors"
LTX_TEXT_ENCODER = PROJECTS_DIR / "ComfyUI" / "models" / "text_encoders" / "comfy_gemma_3_12B_it"
LTX_UPSAMPLER = PROJECTS_DIR / "ComfyUI" / "models" / "upscale_models" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX_DISTILLED_LORA = PROJECTS_DIR / "ComfyUI" / "models" / "loras" / "ltx-2.3-22b-distilled-lora-384.safetensors"

# Scoring model checkouts
NISQA_DIR = Path(os.environ.get("NISQA_DIR", PROJECTS_DIR / "NISQA")).expanduser()

# Enhancement tools
REALESRGAN_MODEL = MODELS_DIR / "realesrgan" / "RealESRGAN_x4plus.pth"

# TTS
FISH_SPEECH_URL = "http://127.0.0.1:8080/v1/tts"

# API keys (from environment)
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")


def _resolve_ml_env_python() -> Path:
    """Interpreter for GPU subprocesses (ID-LoRA inference, LoRA training, TTS).

    Order: UGC_ML_ENV_PYTHON, then a ~/ml-env virtualenv if one exists, then
    the interpreter running this code.
    """
    override = os.environ.get("UGC_ML_ENV_PYTHON")
    if override:
        return Path(override).expanduser()

    candidate = Path.home() / "ml-env" / "bin" / "python"
    if candidate.exists():
        return candidate

    return Path(sys.executable)


# ML environment
ML_ENV_PYTHON = _resolve_ml_env_python()


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUTS_DIR, TEMP_DIR, PERSONAS_DIR, TEMPLATES_DIR, MUSIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
