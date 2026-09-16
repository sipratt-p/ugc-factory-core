"""Pipeline orchestrator: chains script → video generation → assembly.

Uses ID-LoRA inference_two_stage_hq.py for video generation (proven best quality).
The model generates both video and audio jointly. No external TTS.
Assembly uses hard cuts at natural speech pauses.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from . import config
from .models import Job, Persona, Script
from . import video_engine
from . import assembler
from . import subtitles
from . import tts_engine

logger = logging.getLogger(__name__)


def load_persona(slug: str) -> Persona:
    """Load a persona from the personas directory."""
    persona_dir = config.PERSONAS_DIR / slug
    persona_file = persona_dir / "persona.json"

    if not persona_file.exists():
        raise FileNotFoundError(f"Persona '{slug}' not found at {persona_file}")

    with open(persona_file) as f:
        data = json.load(f)

    return Persona(
        name=data["name"],
        slug=slug,
        description=data["description"],
        face_ref_path=persona_dir / data["face_ref"],
        style_prompt=data["style_prompt"],
        voice_ref_path=Path(persona_dir / data["voice_ref"]) if data.get("voice_ref") else None,
        lora_path=Path(data["lora_path"]) if data.get("lora_path") else None,
    )


def load_script(script_path: Path) -> Script:
    """Load a script from a JSON file."""
    with open(script_path) as f:
        data = json.load(f)

    from .models import ScriptScene
    scenes = [
        ScriptScene(
            index=i,
            visual=s["visual"],
            speech=s["speech"],
            duration=s.get("duration", 5.0),
        )
        for i, s in enumerate(data["scenes"])
    ]

    return Script(
        title=data["title"],
        persona_slug=data["persona_slug"],
        scenes=scenes,
        product=data.get("product", "custom"),
        tone=data.get("tone", "casual"),
    )


def run_pipeline(
    script: Script,
    persona: Persona,
    options: Optional[dict] = None,
    on_status: Optional[Callable[[str, int], None]] = None,
) -> Job:
    """Execute the full video generation pipeline.

    Pipeline stages:
    1. Video: Generate clips sequentially with last-frame chain (GPU:0)
       - Model generates BOTH video and audio jointly
       - Each clip uses last frame of previous for visual continuity
    2. Assembly: Stitch clips with crossfade transitions

    Args:
        script: The video script with scenes
        persona: The persona to use (face reference)
        options: Override options (crossfade, transition, etc.)
        on_status: Callback(status_text, progress_pct) for progress updates
    """
    config.ensure_dirs()
    options = options or {}

    job = Job(
        script=script,
        persona_slug=persona.slug,
        status="pending",
        options=options,
    )
    job.job_dir.mkdir(parents=True, exist_ok=True)

    def update(status: str, progress: int):
        job.status = status
        job.progress = progress
        if on_status:
            on_status(status, progress)

    try:
        job.started_at = datetime.now()

        # Load scene bible for consistent prompts
        from .scene_bible import auto_bible_from_image
        bible_path = config.PERSONAS_DIR / persona.slug / "scene_bible.json"
        if bible_path.exists():
            from .scene_bible import SceneBible
            scene_bible = SceneBible.load(bible_path)
            scene_bible.reference_image = str(persona.face_ref_path)
        else:
            scene_bible = auto_bible_from_image(
                image_path=persona.face_ref_path,
                reference_audio=str(persona.voice_ref_path) if persona.voice_ref_path else "",
                persona_name=persona.name,
                cache_dir=config.PERSONAS_DIR / persona.slug,
            )

        # Video generation — native joint AV (model generates both video + audio)
        # V13 proven settings, scene bible prompts, always-original reference
        update("video", 5)
        logger.info(f"[{job.id}] Generating {script.num_scenes} clips (V13 settings)...")

        def video_progress(done, total):
            pct = 5 + int(85 * done / total)  # 5-90% range
            update("video", pct)

        video_clips = video_engine.generate_clips(
            script, persona, job.job_dir,
            scene_bible=scene_bible,
            on_progress=video_progress,
        )
        update("video", 90)
        logger.info(f"[{job.id}] All {len(video_clips)} clips generated")

        # Stage 2: Assembly — stitch clips with crossfade
        update("assembly", 92)
        logger.info(f"[{job.id}] Assembling final video...")
        output_path = job.job_dir / f"{job.id}_final.mp4"

        raw_output = job.job_dir / f"{job.id}_raw.mp4"
        assembler.assemble_av(
            video_clips=video_clips,
            output_path=raw_output,
            crossfade=options.get("crossfade", 0.5),
            transition=options.get("transition", "fade"),
        )

        # Stage 3 (optional): Add subtitles
        current_video = raw_output
        if options.get("subtitles", False):
            update("subtitles", 94)
            logger.info(f"[{job.id}] Adding subtitles...")
            subtitled = job.job_dir / f"{job.id}_subtitled.mp4"
            subtitles.add_subtitles_to_video(
                video_path=current_video,
                output_path=subtitled,
                scenes=script.scenes,
                style=options.get("subtitle_style", "karaoke"),
                use_whisper=options.get("use_whisper", True),
            )
            current_video = subtitled

        # Stage 4 (optional): Add background music
        if options.get("music") and Path(options["music"]).exists():
            update("music", 96)
            logger.info(f"[{job.id}] Adding background music...")
            with_music = job.job_dir / f"{job.id}_music.mp4"
            assembler.add_background_music(
                video_path=current_video,
                music_path=Path(options["music"]),
                output_path=with_music,
                music_volume_db=options.get("music_volume", -20),
            )
            current_video = with_music

        # Copy final result
        import shutil
        shutil.copy2(current_video, output_path)

        job.output_path = output_path
        job.status = "complete"
        job.progress = 100
        job.completed_at = datetime.now()

        # Save job metadata
        job_meta = {
            "id": job.id,
            "status": job.status,
            "persona": persona.slug,
            "script_title": script.title,
            "product": script.product,
            "scenes": script.num_scenes,
            "duration": script.total_duration,
            "output": str(job.output_path),
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat(),
        }
        with open(job.job_dir / "job.json", "w") as f:
            json.dump(job_meta, f, indent=2)

        logger.info(f"[{job.id}] Pipeline complete! Output: {output_path}")
        return job

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.error(f"[{job.id}] Pipeline failed: {e}")
        raise
