"""Persona management: create, list, update, delete personas.

Personas are stored as JSON + reference files on disk at personas/<slug>/.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Optional

from . import config
from .models import Persona

logger = logging.getLogger(__name__)

# Preset templates for quick persona creation
PERSONA_PRESETS = {
    "health_expert": {
        "description": "Health and wellness influencer, warm and approachable, knowledgeable",
        "style_prompt": "A young woman content creator at a modern desk with ring light and microphone, cozy home studio, warm natural lighting, speaking to camera with confidence",
    },
    "product_reviewer": {
        "description": "Tech and product reviewer, casual and energetic, relatable",
        "style_prompt": "A young person at a home office desk with products on display, casual setup, natural lighting, speaking to camera with enthusiasm and gesturing",
    },
    "recruiter": {
        "description": "Professional recruiter, confident and trustworthy, corporate-casual",
        "style_prompt": "A professional person at a clean modern desk, corporate casual attire, soft studio lighting, speaking to camera with warmth and authority",
    },
    "educator": {
        "description": "Educational content creator, clear and engaging, expert vibe",
        "style_prompt": "A knowledgeable presenter at a desk with a whiteboard or screen behind them, bright studio lighting, speaking to camera with clarity and passion",
    },
}


def list_personas() -> list[Persona]:
    """List all personas from disk."""
    config.ensure_dirs()
    personas = []
    for d in sorted(config.PERSONAS_DIR.iterdir()):
        pf = d / "persona.json"
        if pf.exists():
            personas.append(_load_from_dir(d))
    return personas


def get_persona(slug: str) -> Optional[Persona]:
    """Get a single persona by slug."""
    d = config.PERSONAS_DIR / slug
    pf = d / "persona.json"
    if not pf.exists():
        return None
    return _load_from_dir(d)


def create_persona(
    slug: str,
    name: str,
    face_ref_source: Path,
    description: str = "",
    style_prompt: str = "",
    voice_ref_source: Optional[Path] = None,
    preset: Optional[str] = None,
) -> Persona:
    """Create a new persona from provided data.

    If no voice reference is provided, uses a bundled default.
    Auto-generates a scene bible from the reference image using Claude vision.
    """
    from .audio_utils import (
        normalize_reference_audio, validate_reference_image,
        prepare_reference_image, generate_voice_reference, trim_to_sweet_spot,
    )
    from .scene_bible import auto_bible_from_image

    persona_dir = config.PERSONAS_DIR / slug
    if persona_dir.exists():
        raise ValueError(f"Persona '{slug}' already exists")

    persona_dir.mkdir(parents=True)

    # Apply preset defaults if specified
    if preset and preset in PERSONA_PRESETS:
        p = PERSONA_PRESETS[preset]
        if not description:
            description = p["description"]
        if not style_prompt:
            style_prompt = p["style_prompt"]

    # Store a generation-ready portrait reference once so the runtime path stays stable.
    face_ref = persona_dir / "reference.jpg"
    prepare_reference_image(face_ref_source, face_ref)

    img_check = validate_reference_image(face_ref_source)
    for w in img_check.get("warnings", []):
        logger.warning(f"Image warning: {w}")

    # Handle voice reference
    voice_ref_name = None
    if voice_ref_source and Path(voice_ref_source).exists():
        voice_out = persona_dir / "reference.wav"
        normalize_reference_audio(Path(voice_ref_source), voice_out)
        trim_to_sweet_spot(voice_out, voice_out)
        voice_ref_name = "reference.wav"
    else:
        # Auto-generate voice reference via edge-tts
        # Gender detection happens in auto_bible_from_image (below)
        # For now generate female as default, update after bible
        voice_out = persona_dir / "reference.wav"
        try:
            generate_voice_reference(gender="female", output_path=voice_out)
            voice_ref_name = "reference.wav"
            logger.info("Auto-generated female voice reference via edge-tts")
        except Exception as e:
            logger.warning(f"Voice generation failed ({e}), trying bundled default")
            defaults_dir = config.PERSONAS_DIR / "_defaults"
            for fallback_name in ("female_voice_48k_jenny.wav", "female_voice_48k.wav", "male_voice_48k.wav"):
                fallback = defaults_dir / fallback_name
                if fallback.exists():
                    shutil.copy2(fallback, voice_out)
                    voice_ref_name = "reference.wav"
                    break

    # Auto-generate scene bible from image
    bible = auto_bible_from_image(
        image_path=face_ref,
        reference_audio=str(persona_dir / "reference.wav") if voice_ref_name else "",
        persona_name=name,
        cache_dir=persona_dir,
    )

    # Use bible's character description if no description provided
    if not description:
        description = bible.character_description
    if not style_prompt:
        style_prompt = bible.build_visual_prompt()

    # Save persona JSON
    data = {
        "name": name,
        "description": description,
        "face_ref": face_ref.name,
        "style_prompt": style_prompt,
    }
    if voice_ref_name:
        data["voice_ref"] = voice_ref_name

    with open(persona_dir / "persona.json", "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Created persona: {name} ({slug})")
    return _load_from_dir(persona_dir)


def update_persona(slug: str, **kwargs) -> Persona:
    """Update persona fields."""
    persona_dir = config.PERSONAS_DIR / slug
    pf = persona_dir / "persona.json"
    if not pf.exists():
        raise FileNotFoundError(f"Persona '{slug}' not found")

    with open(pf) as f:
        data = json.load(f)

    for key in ("name", "description", "style_prompt"):
        if key in kwargs and kwargs[key] is not None:
            data[key] = kwargs[key]

    # Handle new face reference — normalize once and regenerate the bible
    if "face_ref_source" in kwargs and kwargs["face_ref_source"]:
        from .audio_utils import prepare_reference_image
        from .scene_bible import auto_bible_from_image
        src = Path(kwargs["face_ref_source"])
        face_ref = persona_dir / "reference.jpg"
        prepare_reference_image(src, face_ref)
        data["face_ref"] = face_ref.name
        # Regenerate scene bible for new face
        voice_ref = persona_dir / "reference.wav"
        auto_bible_from_image(
            image_path=face_ref,
            reference_audio=str(voice_ref) if voice_ref.exists() else "",
            persona_name=data.get("name", slug),
            cache_dir=persona_dir,
        )

    with open(pf, "w") as f:
        json.dump(data, f, indent=2)

    return _load_from_dir(persona_dir)


def delete_persona(slug: str):
    """Delete a persona and all its files."""
    persona_dir = config.PERSONAS_DIR / slug
    if persona_dir.exists():
        shutil.rmtree(persona_dir)
        logger.info(f"Deleted persona: {slug}")


def _load_from_dir(persona_dir: Path) -> Persona:
    """Load a Persona from a directory."""
    with open(persona_dir / "persona.json") as f:
        data = json.load(f)
    return Persona(
        name=data["name"],
        slug=persona_dir.name,
        description=data.get("description", ""),
        face_ref_path=persona_dir / data["face_ref"],
        style_prompt=data.get("style_prompt", ""),
        voice_ref_path=Path(persona_dir / data["voice_ref"]) if data.get("voice_ref") else None,
        lora_path=Path(data["lora_path"]) if data.get("lora_path") else None,
    )
