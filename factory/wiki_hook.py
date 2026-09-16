"""Auto-update research wiki after each experiment.

Call update_wiki_after_experiment(result) at the end of any autoresearch run
to append findings to the wiki log and update the leaderboard.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Markdown knowledge base the runners append to (env UGC_WIKI_DIR)
WIKI_DIR = Path(os.environ.get("UGC_WIKI_DIR", Path(__file__).parent.parent / "docs" / "research"))
LOG_FILE = WIKI_DIR / "log.md"


def update_wiki_after_experiment(result: dict):
    """Append experiment result to wiki log.

    Args:
        result: dict with keys like name, version, actual_duration_s,
                sequence_scores, generation_time_s, settings, etc.
    """
    try:
        name = result.get("name", "unknown")
        version = result.get("version", "?")
        dur = result.get("actual_duration_s", "?")
        gen_time = result.get("generation_time_s", "?")
        k = result.get("k", 1)
        seq = result.get("sequence_scores", {})
        final = seq.get("final_score", "?")
        penalties = len(seq.get("penalties", []))
        settings = result.get("settings", {})
        test_id = result.get("test_id", "?")

        # Build log entry
        date = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n## [{date}] experiment | {name}\n"
        entry += f"- Version: V{version}, K={k}, test_id={test_id}\n"
        entry += f"- Duration: {dur}s, Generation: {gen_time}s\n"
        entry += f"- Final score: {final}, Penalties: {penalties}\n"

        # Shot summary
        shots = seq.get("shot_scores", [])
        if shots:
            wers = [s.get("wer") for s in shots if s.get("wer") is not None]
            utmos_vals = [s.get("utmos") for s in shots if s.get("utmos") is not None]
            nisqa_vals = [s.get("nisqa_coloration") for s in shots if s.get("nisqa_coloration") is not None]
            if wers:
                entry += f"- WER: mean={sum(wers)/len(wers):.3f}, worst={max(wers):.3f}\n"
            if utmos_vals:
                entry += f"- UTMOS: mean={sum(utmos_vals)/len(utmos_vals):.3f}\n"
            if nisqa_vals:
                entry += f"- NISQA Coloration: mean={sum(nisqa_vals)/len(nisqa_vals):.3f}\n"

        # Seam summary
        cuts = seq.get("cut_scores", [])
        if cuts:
            seam_scores = [c.get("seam_score") for c in cuts if c.get("seam_score") is not None]
            face_deltas = [c.get("face_delta") for c in cuts if c.get("face_delta") is not None]
            lpips_deltas = [c.get("lpips_delta") for c in cuts if c.get("lpips_delta") is not None]
            if seam_scores:
                entry += f"- Seam scores: mean={sum(seam_scores)/len(seam_scores):.3f}, worst={min(seam_scores):.3f}\n"
            if face_deltas:
                entry += f"- Face deltas: mean={sum(face_deltas)/len(face_deltas):.3f}, worst={max(face_deltas):.3f}\n"
            if lpips_deltas:
                entry += f"- LPIPS at seams: mean={sum(lpips_deltas)/len(lpips_deltas):.4f}, worst={max(lpips_deltas):.4f}\n"

        # Penalties
        for p in seq.get("penalties", []):
            entry += f"- PENALTY: {p['type']} ({p['amount']:+.1f}) — {p['reason']}\n"

        # Settings
        if settings:
            key_settings = {k: v for k, v in settings.items()
                          if k in ["first_frame", "identity_guidance", "audio_guidance",
                                   "xfade_frames", "handle_trim", "preset"]}
            if key_settings:
                entry += f"- Settings: {json.dumps(key_settings)}\n"

        # Append to log
        with open(LOG_FILE, "a") as f:
            f.write(entry)

        logger.info(f"Wiki log updated: {name}")

    except Exception as e:
        logger.warning(f"Wiki update failed: {e}")


def update_wiki_leaderboard(results: list):
    """Update the leaderboard page with latest results."""
    try:
        leaderboard_file = WIKI_DIR / "experiments" / "leaderboard.md"

        # Read existing content up to the auto-generated section
        existing = ""
        if leaderboard_file.exists():
            content = leaderboard_file.read_text()
            marker = "<!-- AUTO-GENERATED BELOW -->"
            if marker in content:
                existing = content[:content.index(marker)]
            else:
                existing = content + "\n\n"

        # Build leaderboard table
        table = "<!-- AUTO-GENERATED BELOW -->\n\n"
        table += "## Latest Results (Auto-Updated)\n\n"
        table += "| Experiment | V | K | Duration | Final | Mean Shot | Penalties |\n"
        table += "|---|---|---|---|---|---|---|\n"

        for r in sorted(results, key=lambda x: x.get("sequence_scores", {}).get("final_score", 0), reverse=True):
            seq = r.get("sequence_scores", {})
            table += (f"| {r.get('name', '?')} | {r.get('version', '?')} | {r.get('k', '?')} | "
                     f"{r.get('actual_duration_s', '?')}s | {seq.get('final_score', '?')} | "
                     f"{r.get('mean_shot_score', '?')} | {len(seq.get('penalties', []))} |\n")

        with open(leaderboard_file, "w") as f:
            f.write(existing + table)

        logger.info(f"Leaderboard updated with {len(results)} results")

    except Exception as e:
        logger.warning(f"Leaderboard update failed: {e}")
