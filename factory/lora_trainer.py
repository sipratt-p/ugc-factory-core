"""LoRA training integration with ai-toolkit.

Generates training configs and launches LoRA training jobs for persona
face consistency. Runs on config.VIDEO_GPU.
"""

import json
import subprocess
import logging
import yaml
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

AI_TOOLKIT_DIR = config.PROJECTS_DIR / "ai-toolkit"


def generate_training_config(
    persona_slug: str,
    dataset_dir: Path,
    output_name: str = None,
    steps: int = 1500,
    rank: int = 128,
    alpha: int = 64,
    lr: float = 4e-5,
    batch_size: int = 4,
    resolution: int = 1024,
) -> Path:
    """Generate an ai-toolkit YAML config for face LoRA training.

    Uses research-backed settings for 8+/10 face consistency:
    - Rank 128 (enough capacity for face geometry)
    - LR 4e-5 (prevents face averaging)
    - Batch 4 (more stable gradients; needs a large-VRAM GPU)
    - 1500 steps (~50 epochs with 25-30 images)
    """
    if not output_name:
        output_name = f"{persona_slug}_lora_v1"

    output_dir = AI_TOOLKIT_DIR / "output" / output_name

    training_config = {
        "job": "extension",
        "config": {
            "name": output_name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": str(output_dir),
                "device": "cuda:0",
                "network": {
                    "type": "lora",
                    "linear": rank,
                    "linear_alpha": alpha,
                },
                "save": {
                    "dtype": "float16",
                    "save_every": 250,
                    "max_step_saves_to_keep": 3,
                },
                "datasets": [{
                    "folder_path": str(dataset_dir),
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [resolution],
                }],
                "train": {
                    "batch_size": batch_size,
                    "steps": steps,
                    "gradient_accumulation_steps": 1,
                    "lr": lr,
                    "optimizer": "adamw8bit",
                    "lr_scheduler": "cosine",
                    "warmup_steps": 100,
                    "max_grad_norm": 1.0,
                    "dtype": "bf16",
                    "noise_scheduler": "flowmatch",
                },
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",
                    "quantize": True,
                },
                "trigger_word": "ohwx person",
                "sample": {
                    "sample_every": 500,
                    "prompts": [
                        f"ohwx person, sitting at a desk, natural lighting, speaking to camera",
                        f"ohwx person, close-up portrait, warm studio lighting, slight smile",
                    ],
                },
            }],
        },
    }

    config_path = AI_TOOLKIT_DIR / "config" / f"{output_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    logger.info(f"Training config saved: {config_path}")
    return config_path


def start_training(config_path: Path) -> subprocess.Popen:
    """Launch LoRA training as a background process."""
    cmd = [
        str(config.ML_ENV_PYTHON),
        str(AI_TOOLKIT_DIR / "run.py"),
        str(config_path),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": str(config.VIDEO_GPU),
        "PATH": str(config.ML_ENV_PYTHON.parent) + ":/usr/bin:/bin",
        "HOME": str(Path.home()),
    }

    log_path = config_path.with_suffix(".log")
    log_file = open(log_path, "w")

    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(AI_TOOLKIT_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    logger.info(f"Training started: PID {process.pid}, log: {log_path}")
    return process


def get_training_status(config_path: Path) -> dict:
    """Check training status from log file."""
    log_path = config_path.with_suffix(".log")
    if not log_path.exists():
        return {"status": "not_started"}

    with open(log_path) as f:
        lines = f.readlines()

    if not lines:
        return {"status": "starting"}

    last_line = lines[-1].strip()

    # Parse step progress
    for line in reversed(lines):
        if "step" in line.lower() and "/" in line:
            return {"status": "running", "last_line": line.strip(), "total_lines": len(lines)}
        if "error" in line.lower() or "exception" in line.lower():
            return {"status": "error", "error": line.strip()}
        if "saved" in line.lower() and "safetensors" in line.lower():
            return {"status": "complete", "last_line": line.strip()}

    return {"status": "running", "last_line": last_line, "total_lines": len(lines)}
