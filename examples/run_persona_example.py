#!/usr/bin/env python3
"""Produce a gated talking-head reel from your own persona.

Persona directory layout (see examples/persona_template/):

    personas/<slug>/
        persona.json        name, description, style prompt
        reference.jpg       head-and-shoulders photo (portrait)
        reference.wav       3 to 15 s of one speaker, 48 kHz stereo preferred
        scene_bible.json    optional: character / setting / camera description

Usage:
    python examples/run_persona_example.py --persona-dir personas/my_persona
    python examples/run_persona_example.py --persona-dir personas/my_persona \
        --scripts my_lines.json --best-of 2 --frames 121 --music beds/lofi.wav

Every segment goes through script lint, then generation, then the SyncNet +
transcription quality gate with adaptive retry (factory/quality_gate.py).
With --best-of K each segment is generated K times from disjoint seed banks
and the best gated take is kept. Read the README's "Lessons learned" before
trusting best-of selection on lip-sync score alone.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factory.scene_bible import bible_from_persona_dir  # noqa: E402
from factory.script_lint import lint_scripts  # noqa: E402
from factory.video_producer import produce_video  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Placeholder lines: one flowing sentence each, no URLs, no digits, no
# "link in bio", about ten words for a 121-frame (5 s) window.
DEFAULT_SCRIPTS = [
    "Most advice you see online is marketing dressed up as science.",
    "Before you buy anything, look up the actual clinical evidence.",
    "A few minutes of reading saves you months of guessing.",
    "Check the evidence first and decide with a clear head.",
]


def main():
    parser = argparse.ArgumentParser(description="Gated talking-head production for one persona")
    parser.add_argument("--persona-dir", required=True, help="personas/<slug> with reference.jpg + reference.wav")
    parser.add_argument("--scripts", help="JSON file: list of speech strings, one per segment")
    parser.add_argument("--frames", type=int, default=121, help="frames per segment (121 or 129 recommended)")
    parser.add_argument("--best-of", type=int, default=1, help="gated candidates per segment (1 = volume, 2+ = hero)")
    parser.add_argument("--music", help="optional background music WAV")
    parser.add_argument("--music-db", type=float, default=-22, help="music level in dB")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA index for generation (default config.VIDEO_GPU)")
    parser.add_argument("--output", help="output directory (default UGC_RESULTS_DIR/produce_<id>)")
    parser.add_argument("--skip-lint", action="store_true", help="bypass script lint errors")
    args = parser.parse_args()

    scripts = DEFAULT_SCRIPTS
    if args.scripts:
        scripts = json.loads(Path(args.scripts).read_text())
        assert isinstance(scripts, list) and all(isinstance(s, str) for s in scripts), \
            "--scripts must be a JSON list of strings"

    bible = bible_from_persona_dir(Path(args.persona_dir))
    print(f"Persona: {bible.character_name}")
    print(f"  image: {bible.reference_image}")
    print(f"  audio: {bible.reference_audio or '(none: the model will improvise a voice)'}")
    print(f"  [VISUAL]: {bible.build_visual_prompt()}")
    print(f"  [SOUNDS]: {bible.build_sounds_prompt()}")

    errors, warnings = lint_scripts(scripts, num_frames=args.frames)
    if errors and not args.skip_lint:
        sys.exit("fix the lint errors above or pass --skip-lint")

    result = produce_video(
        bible=bible,
        scripts=scripts,
        output_dir=Path(args.output) if args.output else None,
        num_frames=args.frames,
        music_path=Path(args.music).resolve() if args.music else None,
        music_volume_db=args.music_db,
        gpu_id=args.gpu,
        skip_lint=True,  # already linted above
        best_of=args.best_of,
    )

    print(f"\nFINAL: {result.final_path}")
    print(f"duration={result.duration:.1f}s  gen_time={result.gen_time:.0f}s  "
          f"passed={result.passed_count}/{result.total_count}")
    for i, q in enumerate(result.segments):
        status = "PASS" if q.passes else f"BEST-EFFORT ({q.failure_reason})"
        print(f"  seg{i}: {q.syncnet_quality:8s} LSE-C={q.syncnet_lse_c:.2f}  {status}")
        print(f"         said: {q.transcript!r}")


if __name__ == "__main__":
    main()
