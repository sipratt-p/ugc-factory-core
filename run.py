#!/usr/bin/env python3
"""UGC Factory CLI — generate videos from the command line.

Usage:
    python run.py generate --persona my_persona --topic "vitamin D benefits" --duration 30
    python run.py generate --persona my_persona --script scripts/test_script.json
    python run.py list-personas
    python run.py list-jobs
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from factory import config
from factory.models import Script, ScriptScene
from factory.orchestrator import load_persona, load_script, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ugc-factory")


def make_test_script(persona_slug: str, topic: str, product: str, duration: int) -> Script:
    """Create a test script with unique speech per scene, each ending at a natural pause."""
    clip_duration = config.CLIP_DURATION
    num_scenes = max(1, int(duration / clip_duration))

    # Load persona style prompt for visual consistency
    from factory.orchestrator import load_persona as _load
    persona = _load(persona_slug)
    style = persona.style_prompt

    # Each scene uses the persona's specific visual description
    scene_library = [
        {
            "visual": f"{style}, home studio setup with microphone visible.",
            "speech": f"Hey everyone welcome back to the channel. Today we need to talk about {topic}. This has completely changed my routine and I think it can help you too.",
        },
        {
            "visual": f"{style}, leaning forward, making eye contact with camera, gesturing naturally.",
            "speech": f"So here is what most people get wrong about {topic}. The research is actually really clear on this but somehow nobody talks about it.",
        },
        {
            "visual": f"{style}, animated expressions, counting on fingers, speaking with passion.",
            "speech": f"I started paying attention to {topic} about six months ago and honestly the results have been incredible. My energy levels are completely different now.",
        },
        {
            "visual": f"{style}, warm smile, pointing at camera, call to action energy, leaning toward camera.",
            "speech": f"If this was helpful hit follow and drop a comment about your experience with {topic}. I read every single one. See you next time!",
        },
    ]

    scenes = []
    for i in range(num_scenes):
        s = scene_library[i % len(scene_library)]
        scenes.append(ScriptScene(
            index=i,
            visual=s["visual"],
            speech=s["speech"],
            duration=clip_duration,
        ))

    return Script(
        title=f"{topic} - UGC Video",
        persona_slug=persona_slug,
        scenes=scenes,
        product=product,
        tone="casual",
    )


def cmd_generate(args):
    persona = load_persona(args.persona)
    logger.info(f"Loaded persona: {persona.name} ({persona.slug})")

    if args.script:
        script = load_script(Path(args.script))
    else:
        script = make_test_script(
            persona_slug=args.persona,
            topic=args.topic,
            product=args.product,
            duration=args.duration,
        )

    logger.info(f"Script: {script.title} ({script.num_scenes} scenes, {script.total_duration}s)")
    for scene in script.scenes:
        logger.info(f"  Scene {scene.index}: {scene.speech[:60]}...")

    options = {}
    if args.subtitles:
        options["subtitles"] = True
        options["subtitle_style"] = args.subtitle_style
    if args.music:
        music_path = config.MUSIC_DIR / args.music
        if music_path.exists():
            options["music"] = str(music_path)
        else:
            logger.warning(f"Music file not found: {music_path}")

    def on_status(status, progress):
        logger.info(f"  [{progress}%] {status}")

    job = run_pipeline(script, persona, options=options, on_status=on_status)
    logger.info(f"\nVideo generated: {job.output_path}")
    logger.info(f"Job ID: {job.id}")


def cmd_list_personas(args):
    config.ensure_dirs()
    found = False
    for d in sorted(config.PERSONAS_DIR.iterdir()):
        pf = d / "persona.json"
        if pf.exists():
            with open(pf) as f:
                data = json.load(f)
            print(f"  {d.name}: {data['name']}")
            found = True
    if not found:
        print("  No personas found. Create one in personas/<slug>/persona.json")


def cmd_list_jobs(args):
    config.ensure_dirs()
    found = False
    for d in sorted(config.OUTPUTS_DIR.iterdir(), reverse=True):
        jf = d / "job.json"
        if jf.exists():
            with open(jf) as f:
                data = json.load(f)
            print(f"  [{data['status']}] {data['id']} - {data.get('script_title', '?')} ({data.get('duration', '?')}s)")
            found = True
    if not found:
        print("  No jobs found.")


def main():
    parser = argparse.ArgumentParser(description="UGC Factory CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate a video")
    gen.add_argument("--persona", required=True, help="Persona slug")
    gen.add_argument("--topic", default="AI video generation", help="Video topic")
    gen.add_argument("--product", default="custom", help="Free-form product tag stored with the job (default: custom)")
    gen.add_argument("--duration", type=int, default=30, help="Target duration in seconds")
    gen.add_argument("--script", help="Path to script JSON file")
    gen.add_argument("--subtitles", action="store_true", help="Add animated subtitles")
    gen.add_argument("--subtitle-style", default="karaoke", choices=["karaoke", "pop"])
    gen.add_argument("--music", help="Background music file from music/ directory")
    gen.set_defaults(func=cmd_generate)

    lp = subparsers.add_parser("list-personas", help="List available personas")
    lp.set_defaults(func=cmd_list_personas)

    lj = subparsers.add_parser("list-jobs", help="List completed jobs")
    lj.set_defaults(func=cmd_list_jobs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
