"""Data models for the UGC Factory pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid


@dataclass
class Persona:
    name: str
    slug: str                          # filesystem-safe identifier
    description: str
    face_ref_path: Path                # reference image for I2V conditioning
    style_prompt: str                  # visual description for prompt prefix
    voice_ref_path: Optional[Path] = None  # optional voice reference
    lora_path: Optional[Path] = None   # custom Flux 2 LoRA (if trained)


@dataclass
class ScriptScene:
    index: int
    visual: str                        # scene/setting description
    speech: str                        # what the person says
    duration: float = 5.0              # seconds

    # V13 proven frozen [SOUNDS] — minimal to avoid re-randomizing voice per clip
    DEFAULT_SOUNDS = "close mic, measured pace, dry room, neutral studio ambience"

    def to_prompt(self, sounds: str = None) -> str:
        """Format as [VISUAL]/[SPEECH]/[SOUNDS] tags required by LTX 2.3.

        The model generates both video and audio jointly. The [SPEECH] tag
        drives lip-sync and audio generation. [SOUNDS] controls vocal tone
        and ambient audio — frozen minimal prompt proven best in V2-V13.
        """
        return (
            f"[VISUAL]: {self.visual} "
            f"[SPEECH]: {self.speech} "
            f"[SOUNDS]: {sounds or self.DEFAULT_SOUNDS}"
        )


@dataclass
class Script:
    title: str
    persona_slug: str
    scenes: list[ScriptScene]
    product: str = "custom"            # free-form product tag stored with the job
    tone: str = "casual"               # educational | promotional | casual

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.scenes)

    @property
    def num_scenes(self) -> int:
        return len(self.scenes)


@dataclass
class ClipResult:
    scene_index: int
    video_path: Optional[Path] = None
    status: str = "pending"            # pending | generating | done | failed
    error: Optional[str] = None


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    script: Optional[Script] = None
    persona_slug: str = ""
    status: str = "pending"            # pending | video | assembly | complete | failed
    progress: int = 0                  # 0-100
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    clips: list[ClipResult] = field(default_factory=list)
    output_path: Optional[Path] = None
    error: Optional[str] = None
    options: dict = field(default_factory=dict)

    @property
    def job_dir(self) -> Path:
        from . import config
        return config.OUTPUTS_DIR / self.id
