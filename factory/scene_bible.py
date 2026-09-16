"""Scene Bible — persistent world state for multi-clip video consistency.

From ChatGPT5 Pro research: treat multi-clip as a graph of anchored shots,
not a chain of independent generations. The scene bible maintains the
"world model" across all clips: character appearance, setting, lighting,
camera grammar, voice characteristics, and pacing targets.

Every clip prompt is generated FROM the bible, ensuring consistency.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# English heteronyms — same spelling, different pronunciation depending on meaning.
# LTX has no pronunciation dictionary and will guess wrong on these.
# Use phonetic alternatives or simpler synonyms instead.
HETERONYMS = {
    "resume": "CV, application (not rez-oo-may vs reh-ZOO-may)",
    "resumes": "CVs, applications",
    "live": "use 'in real time' or 'currently' instead",
    "read": "use past tense 'reviewed' or present 'check' instead",
    "lead": "use 'guide' or 'top candidate' instead",
    "wound": "use 'injury' or 'wrapped' instead",
    "close": "use 'near' or 'shut' — context dependent",
    "tear": "use 'rip' or 'crying' instead",
    "minute": "use 'tiny' or 'sixty seconds' instead",
    "bass": "use 'low frequency' or 'fish' — context dependent",
    "bow": "use 'ribbon' or 'bend forward' instead",
    "present": "use 'gift' or 'current' or 'show' instead",
    "record": "use 'log' or 'best ever' instead",
    "project": "use 'plan' or 'forecast' instead",
    "produce": "use 'make' or 'fruits and vegetables' instead",
    "refuse": "use 'reject' or 'garbage' instead",
    "content": "use 'satisfied' or 'material' — context dependent",
    "desert": "use 'abandon' or 'sandy wasteland' instead",
    "object": "use 'thing' or 'oppose' instead",
    "invalid": "use 'not valid' or 'disabled person' instead",
    "buffet": "use 'food spread' — avoid, often mispronounced",
    "analyses": "use 'analysis results' or 'reviews' instead",
}


def check_pronunciation_safety(speech: str) -> list[str]:
    """Check speech text for words LTX is likely to mispronounce.

    Returns list of warnings. Empty list = safe.
    """
    warnings = []
    words = speech.lower().replace('"', '').replace('.', '').replace(',', '').split()
    for word in words:
        if word in HETERONYMS:
            warnings.append(f"'{word}' — {HETERONYMS[word]}")
    return warnings


def validate_script(scenes: list[dict]) -> list[dict]:
    """Validate a multi-scene script for pronunciation safety.

    Returns list of issues found. Empty = all safe.
    """
    issues = []
    for i, scene in enumerate(scenes):
        speech = scene.get("speech", "")
        warnings = check_pronunciation_safety(speech)
        for w in warnings:
            issues.append({"segment": i, "warning": w, "speech": speech[:80]})
            logger.warning(f"  Pronunciation risk in segment {i}: {w}")
    return issues


@dataclass
class SceneBible:
    """Persistent world state for a multi-clip video sequence."""

    # Character identity
    character_name: str = ""
    character_description: str = ""
    wardrobe: str = ""
    hair: str = ""
    distinguishing_features: str = ""

    # Setting
    location: str = ""
    location_details: str = ""
    lighting: str = ""
    background: str = ""
    props: list[str] = field(default_factory=list)

    # Camera
    camera_angle: str = "head-and-shoulders close-up"
    camera_movement: str = "fixed camera, clean composition"
    framing: str = ""

    # Audio
    # Minimal frozen [SOUNDS] — avoid timbre adjectives that re-randomize
    # voice per clip. Let reference audio handle voice character.
    voice_description: str = "close mic"
    speaking_pace: str = "measured pace"
    emotional_band: str = ""
    room_tone: str = "dry room, neutral studio ambience"
    target_wps: float = 2.5  # words per second target

    # Technical
    reference_image: str = ""
    reference_audio: str = ""

    def build_visual_prompt(self) -> str:
        """Build consistent [VISUAL] tag from the bible."""
        parts = []
        if self.camera_angle:
            parts.append(self.camera_angle)
        if self.character_description:
            parts.append(f"of {self.character_description}")
        if self.wardrobe:
            parts.append(self.wardrobe)
        if self.location:
            parts.append(self.location)
        if self.lighting:
            parts.append(self.lighting)
        if self.camera_movement:
            parts.append(self.camera_movement)
        parts.append("Speaking directly to camera.")
        return ", ".join(parts)

    def build_sounds_prompt(self) -> str:
        """Build consistent [SOUNDS] tag from the bible."""
        parts = []
        if self.voice_description:
            parts.append(self.voice_description)
        if self.speaking_pace:
            parts.append(self.speaking_pace)
        if self.room_tone:
            parts.append(self.room_tone)
        return ", ".join(parts)

    def build_shot_prompt(self, speech: str) -> str:
        """Build a complete prompt for one shot from the bible + speech."""
        visual = self.build_visual_prompt()
        sounds = self.build_sounds_prompt()
        return f"[VISUAL]: {visual} [SPEECH]: {speech} [SOUNDS]: {sounds}"

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SceneBible":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# Generic bible — works with any image because ID-LoRA's --first-frame dominates
GENERIC_BIBLE = SceneBible(
    character_description="a person",
    camera_angle="Head-and-shoulders close-up",
    camera_movement="fixed camera, clean composition, subtle eye blinks, slight natural head motion",
    lighting="soft natural lighting",
    voice_description="close mic",
    speaking_pace="measured pace",
    room_tone="dry room, neutral studio ambience",
)


def auto_bible_from_image(
    image_path: Path,
    reference_audio: str = "",
    persona_name: str = "",
    cache_dir: Path = None,
) -> SceneBible:
    """Auto-generate a scene bible from any reference image using Claude vision.

    ID-LoRA's --first-frame conditioning is very strong — the visual prompt
    just needs to not contradict the image. This function describes the image
    so the prompt aligns with what the model sees.

    Falls back to GENERIC_BIBLE if no API key or vision call fails.
    """
    # Check cache first
    if cache_dir:
        cached = cache_dir / "scene_bible.json"
        if cached.exists():
            try:
                bible = SceneBible.load(cached)
                bible.reference_image = str(image_path)
                if reference_audio:
                    bible.reference_audio = reference_audio
                logger.info(f"Loaded cached scene bible from {cached}")
                return bible
            except Exception:
                pass

    # Try Claude vision
    from . import config
    api_key = config.CLAUDE_API_KEY
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY — using generic bible")
        bible = SceneBible(
            **{k: v for k, v in GENERIC_BIBLE.__dict__.items()
               if k not in ("reference_image", "reference_audio")},
            reference_image=str(image_path),
            reference_audio=reference_audio,
        )
        return bible

    try:
        import anthropic
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode()

        # Detect actual image format from file header, not extension
        with open(image_path, "rb") as img_f:
            header = img_f.read(8)
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif header[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        elif header[:4] == b'RIFF':
            media_type = "image/webp"
        else:
            media_type = "image/png"

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": (
                        "Describe this person for a video generation prompt. Return JSON only:\n"
                        '{"character_description": "a [gender] with [hair] [distinguishing features]",\n'
                        ' "wardrobe": "[clothing description]",\n'
                        ' "hair": "[hair color and style]",\n'
                        ' "distinguishing_features": "[notable features]",\n'
                        ' "location": "[setting/background]",\n'
                        ' "lighting": "[lighting description]",\n'
                        ' "apparent_gender": "male" or "female"}\n'
                        "Be factual and concise. No markdown fences."
                    )},
                ],
            }],
        )

        import json
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)

        bible = SceneBible(
            character_name=persona_name or "auto",
            character_description=data.get("character_description", "a person"),
            wardrobe=data.get("wardrobe", ""),
            hair=data.get("hair", ""),
            distinguishing_features=data.get("distinguishing_features", ""),
            location=data.get("location", ""),
            lighting=data.get("lighting", "soft natural lighting"),
            background=data.get("location", ""),
            camera_angle="Head-and-shoulders close-up",
            camera_movement="fixed camera, clean composition, subtle eye blinks, slight natural head motion",
            voice_description="close mic",
            speaking_pace="measured pace",
            room_tone="dry room, neutral studio ambience",
            reference_image=str(image_path),
            reference_audio=reference_audio,
        )

        # Cache it
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            bible.save(cache_dir / "scene_bible.json")

        logger.info(f"Auto-generated bible: {data.get('character_description', '?')}")
        return bible

    except Exception as e:
        logger.warning(f"Claude vision failed ({e}) — using generic bible")
        return SceneBible(
            **{k: v for k, v in GENERIC_BIBLE.__dict__.items()
               if k not in ("reference_image", "reference_audio")},
            reference_image=str(image_path),
            reference_audio=reference_audio,
        )


# ---------------------------------------------------------------------------
# Loading a bible for a persona directory
# ---------------------------------------------------------------------------

REFERENCE_IMAGE_NAMES = ("reference.jpg", "reference.jpeg", "reference.png", "reference.webp")


def bible_from_persona_dir(persona_dir: Path, character_name: str = "") -> SceneBible:
    """Build a SceneBible for personas/<slug>/.

    Uses personas/<slug>/scene_bible.json when present (persona_manager and
    auto_bible_from_image write one), otherwise a copy of GENERIC_BIBLE, and
    points reference_image / reference_audio at the persona's reference files
    (reference.jpg|png|webp and reference.wav).
    """
    persona_dir = Path(persona_dir)
    image = next((persona_dir / n for n in REFERENCE_IMAGE_NAMES if (persona_dir / n).exists()), None)
    if image is None:
        raise FileNotFoundError(
            f"No reference image in {persona_dir} (expected one of {', '.join(REFERENCE_IMAGE_NAMES)})")
    audio = persona_dir / "reference.wav"

    bible_path = persona_dir / "scene_bible.json"
    if bible_path.exists():
        bible = SceneBible.load(bible_path)
    else:
        bible = SceneBible(**{k: v for k, v in GENERIC_BIBLE.__dict__.items()
                              if k not in ("reference_image", "reference_audio")})
    if character_name:
        bible.character_name = character_name
    elif not bible.character_name:
        bible.character_name = persona_dir.name
    bible.reference_image = str(image.resolve())
    bible.reference_audio = str(audio.resolve()) if audio.exists() else ""
    return bible
