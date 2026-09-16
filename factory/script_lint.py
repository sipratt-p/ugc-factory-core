"""Pre-generation script lint for UGC segments.

Codifies the known script-level lip-sync killers and editorial rules so they
block BEFORE GPU time instead of surfacing as failed quality gates after
~3 min/segment of generation:

- Multi-sentence segments (periods between phrases freeze LTX lip-sync)
- "link in bio" / URLs (freeze lip-sync AND violate the educational-tone rule)
- Promotional phrasing ("buy now") — editorial rule: educational content with
  a subtle product mention, not a pitch
- Segment length vs the frame window (121fr @ 24fps ≈ 5s ≈ 12-13 words at
  the ~2.5 wps bible pace; >20 words measurably degrades WER)
- Digits / all-caps acronyms (TTS misreads — spell numbers out)

Usage (library):
    from factory.script_lint import lint_scripts
    errors, warnings = lint_scripts(scripts, num_frames=121)

Usage (CLI):
    python -m factory.script_lint "sentence one" "sentence two"
"""
from __future__ import annotations

import re
import sys

RE_SENTENCE_END = re.compile(r"[.!?]+(?:\s+)")  # sentence break mid-line
RE_URL = re.compile(r"https?://\S+|www\.\S+|\S+\.(com|ai|io|net)\b", re.I)
RE_LINK_IN_BIO = re.compile(r"\blink\s+in\s+(the\s+)?bio\b", re.I)
RE_PROMO = re.compile(r"\b(buy now|order now|act fast|limited time|don't miss)\b", re.I)
RE_DIGITS = re.compile(r"\d")
RE_ALLCAPS = re.compile(r"\b[A-Z]{3,}\b")

WPS = 2.5  # bible speaking pace
FPS = 24


def lint_segment(script: str, num_frames: int = 121) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one segment's speech text."""
    errors, warnings = [], []
    s = script.strip().strip('"').strip()

    if not s:
        return ["empty segment"], []

    if RE_SENTENCE_END.search(s):
        n = len(RE_SENTENCE_END.findall(s)) + 1
        errors.append(f"{n} sentences in one segment — periods between phrases "
                      f"freeze LTX lip-sync; use one flowing sentence with commas, "
                      f"or split into separate segments")
    if RE_URL.search(s):
        errors.append("URL/domain in speech — freezes lip-sync; say the name, not the address")
    if RE_LINK_IN_BIO.search(s):
        errors.append("'link in bio' — freezes lip-sync AND violates the "
                      "educational-tone rule")
    if RE_PROMO.search(s):
        warnings.append("promotional phrasing — editorial rule is educational "
                        "content with subtle product mention")
    if RE_DIGITS.search(s):
        warnings.append("digit in speech — spell numbers out for reliable TTS")
    for m in RE_ALLCAPS.finditer(s):
        if m.group(0) not in ("AI",):
            warnings.append(f"all-caps '{m.group(0)}' may be spelled out letter-by-letter")

    window_s = num_frames / FPS
    words = len(s.split())
    # Number words speak slower than the average wps — "five hundred" takes
    # ~2x a normal word (caught live 2026-06-10: a 10-word segment with
    # "five hundred" truncated despite fitting the naive estimate).
    NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "twenty", "thirty", "forty",
                    "fifty", "hundred", "thousand", "million", "percent"}
    n_num = sum(1 for w in s.lower().split() if w.strip(".,!?") in NUMBER_WORDS)
    est_s = (words + n_num) / WPS
    if est_s > window_s:
        errors.append(f"{words} words ≈ {est_s:.1f}s > {window_s:.1f}s frame window "
                      f"({num_frames}fr) — speech will truncate; trim to "
                      f"~{int(window_s * WPS)} words")
    elif est_s > window_s * 0.9:
        warnings.append(f"{words} words ≈ {est_s:.1f}s is tight against the "
                        f"{window_s:.1f}s window")
    if words > 20:
        warnings.append(f"{words} words — WER degrades measurably above ~20 words")

    return errors, warnings


def lint_scripts(scripts: list[str], num_frames: int = 121,
                 verbose: bool = True) -> tuple[int, int]:
    """Lint all segments. Returns (total_errors, total_warnings)."""
    total_e = total_w = 0
    for i, s in enumerate(scripts, 1):
        errors, warnings = lint_segment(s, num_frames)
        for e in errors:
            if verbose:
                print(f"  [ERROR] seg {i}: {e}\n          {s!r}")
        for w in warnings:
            if verbose:
                print(f"  [WARN]  seg {i}: {w}\n          {s!r}")
        total_e += len(errors)
        total_w += len(warnings)
    if verbose:
        print(f"script lint: {total_e} errors, {total_w} warnings "
              f"across {len(scripts)} segments")
    return total_e, total_w


if __name__ == "__main__":
    e, _ = lint_scripts(sys.argv[1:])
    sys.exit(1 if e else 0)
