"""Script generation using Gemma 4, Claude API, or templates.

Generates structured scripts with [VISUAL]/[SPEECH]/[SOUNDS] tags
for the ID-LoRA pipeline. Each scene is ~5 seconds of speakable content.
"""

import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional

from . import config
from .models import Persona, Script, ScriptScene
from .scene_bible import check_pronunciation_safety, HETERONYMS

logger = logging.getLogger(__name__)

# Local OpenAI-compatible chat endpoint for script generation (vLLM, llama.cpp,
# oMLX, etc.). Developed against Gemma 4 31B; any instruct model works.
import os
GEMMA_API_URL = os.environ.get("UGC_LLM_API_URL", "http://localhost:8000/v1/chat/completions")
GEMMA_MODEL_ID = os.environ.get("UGC_LLM_MODEL", "gemma-4-31b-it")

SYSTEM_PROMPT = """You are a UGC video script writer for short-form social media (TikTok/Instagram Reels).

You write scripts for AI-generated talking-head videos. Each script is broken into 5-second scenes.
The AI model generates both video AND audio jointly — the visual description and speech are combined
into a single natural prompt.

RULES:
- Each scene's speech MUST be speakable in exactly 5 seconds (about 15-20 words)
- The "visual" field describes what the viewer sees — camera angle, lighting, expressions, gestures
- The "speech" field is the exact words spoken — conversational, punchy, engaging
- Start with a hook scene that grabs attention in the first 2 seconds
- End with a clear call-to-action
- Keep it conversational — like talking to a friend, not reading a script
- Use short sentences. No jargon unless the audience expects it.
- Keep the visual description consistent across scenes (same setting, same person)

OUTPUT FORMAT: Return valid JSON only, no markdown fences. Schema:
{
  "title": "Video title",
  "scenes": [
    {
      "visual": "Description of the scene — setting, lighting, gestures, expressions",
      "speech": "Exact words spoken in this 5-second segment"
    }
  ]
}"""


GEMMA_SYSTEM_PROMPT = f"""You are a UGC video script writer for short-form social media ads (TikTok/Instagram Reels).

You write the [SPEECH] text for AI-generated talking-head videos. An AI video model (LTX 2.3) will
generate both video and audio from your text — it speaks the exact words you write.

CRITICAL RULES:
1. Each segment's speech must be speakable in the target duration at ~2.5 words/second.
   - 129 frames (~5.4s): 12-14 words max
   - 161 frames (~6.7s): 15-17 words max
2. Wrap each speech in double quotes: "Like this."
3. PRONUNCIATION SAFETY — the AI model has no pronunciation dictionary. AVOID these words
   which have multiple pronunciations (heteronyms): {', '.join(HETERONYMS.keys())}
   Use the simpler alternatives instead.
4. Spell out URLs phonetically: "example dot com", "my brand dot AI"
5. Avoid technical jargon the model might slur: acronyms, numbers with units, chemical names.
   Instead of "400mg magnesium glycinate" say "four hundred milligrams of magnesium"
6. First segment MUST be a pattern-interrupt hook (question, bold claim, surprising fact, or "stop doing X")
7. Last segment should mention the product naturally — like a friend sharing a useful tool, NOT a hard sell.
   Include the URL but frame it as a recommendation, not an ad CTA.
8. Middle segments teach something genuinely useful. The viewer should learn something real
   even if they never click. Educational value first, product mention is subtle.
9. Keep sentences short. No subordinate clauses. Speak like a friend, not a textbook.

OUTPUT FORMAT: Return ONLY valid JSON, no markdown fences or explanation.
{{
  "title": "Short descriptive title",
  "scenes": [
    {{"speech": "Exact words for segment 1"}},
    {{"speech": "Exact words for segment 2"}}
  ]
}}"""


def _call_gemma(messages: list[dict], max_tokens: int = 1000) -> str:
    """Call the local OpenAI-compatible chat endpoint (UGC_LLM_API_URL)."""
    payload = json.dumps({
        "model": GEMMA_MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        GEMMA_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"Gemma 4 API call failed: {e}")

    choice = data["choices"][0]["message"]
    return choice.get("content", "") or ""


def generate_with_gemma(
    product: str,
    product_description: str,
    num_segments: int = 4,
    num_frames: int = 129,
    tone: str = "casual",
    url: str = "",
) -> list[dict]:
    """Generate script segments using the local OpenAI-compatible LLM.

    Returns list of dicts with "speech" key, ready for autoresearch runner.
    """
    duration_per_seg = num_frames / 24.0
    max_words = int(duration_per_seg * 2.5)

    user_prompt = (
        f"Write a {num_segments}-segment UGC ad script for {product}.\n\n"
        f"Product description: {product_description}\n"
        f"URL (spell phonetically): {url}\n"
        f"Tone: {tone}\n"
        f"Each segment: ~{duration_per_seg:.1f}s, max {max_words} words\n"
        f"Total segments: {num_segments}\n\n"
        f"Remember: first segment = attention-grabbing hook, last = CTA with URL."
    )

    messages = [
        {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Scale tokens with segment count — Gemma's thinking mode eats tokens
    token_budget = max(1000, num_segments * 150 + 500)

    # Try up to 2 times (Gemma occasionally returns empty on first attempt)
    data = None
    for attempt in range(2):
        response_text = _call_gemma(messages, max_tokens=token_budget)
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            data = json.loads(text)
            break
        except json.JSONDecodeError:
            logger.warning(f"  Gemma attempt {attempt+1} returned invalid JSON, retrying...")
            if attempt == 1:
                raise RuntimeError(f"Gemma returned invalid JSON after 2 attempts: {text[:200]}")

    scenes = data.get("scenes", [])

    # Validate pronunciation safety
    for i, scene in enumerate(scenes):
        speech = scene.get("speech", "")
        warnings = check_pronunciation_safety(speech)
        if warnings:
            logger.warning(f"  Segment {i} pronunciation risks: {warnings}")
            # Ask Gemma to fix
            fix_prompt = (
                f"This speech has pronunciation risks for our AI voice model: {speech}\n"
                f"Problems: {warnings}\n"
                f"Rewrite it avoiding those words. Keep the same meaning and length. "
                f"Return ONLY the rewritten speech text in double quotes, nothing else."
            )
            fixed = _call_gemma(
                [{"role": "user", "content": fix_prompt}],
                max_tokens=200,
            ).strip().strip('"')
            if fixed and len(fixed) > 10:
                logger.info(f"  Fixed segment {i}: {fixed[:60]}...")
                scene["speech"] = f'"{fixed}"'

    # Validate word counts
    for i, scene in enumerate(scenes):
        speech = scene["speech"].strip('"')
        word_count = len(speech.split())
        if word_count > max_words + 3:
            logger.warning(f"  Segment {i}: {word_count} words exceeds {max_words} target")

    logger.info(f"LLM generated {len(scenes)} segments for {product}")
    return scenes


def generate_with_claude(
    topic: str,
    persona: Persona,
    product: str = "custom",
    tone: str = "casual",
    duration: int = 30,
) -> Script:
    """Generate a script using Claude API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    if not config.CLAUDE_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    num_scenes = max(2, duration // 5)

    user_prompt = f"""Write a {duration}-second UGC video script ({num_scenes} scenes of 5 seconds each).

Topic: {topic}
Product/Brand: {product}
Tone: {tone}
Persona: {persona.description}
Visual style: {persona.style_prompt}

The presenter ({persona.name}) is speaking directly to camera in a {tone} style.
Make it feel authentic and engaging — not corporate or stiff."""

    client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text.strip()

    # Parse JSON (handle potential markdown fences)
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    response_text = response_text.strip()

    data = json.loads(response_text)

    scenes = [
        ScriptScene(
            index=i,
            visual=s["visual"],
            speech=s["speech"],
            duration=s.get("duration", 5.0),
        )
        for i, s in enumerate(data["scenes"])
    ]

    script = Script(
        title=data.get("title", f"{topic} - UGC Video"),
        persona_slug=persona.slug,
        scenes=scenes,
        product=product,
        tone=tone,
    )

    logger.info(f"Claude generated script: {script.title} ({script.num_scenes} scenes)")
    return script


def generate_from_template(
    template_path: Path,
    variables: dict,
    persona: Persona,
) -> Script:
    """Generate a script from a template with variable substitution."""
    with open(template_path) as f:
        data = json.load(f)

    scenes = []
    for i, s in enumerate(data["scenes"]):
        visual = s["visual"]
        speech = s["speech"]
        for key, value in variables.items():
            visual = visual.replace(f"{{{{{key}}}}}", value)
            speech = speech.replace(f"{{{{{key}}}}}", value)

        scenes.append(ScriptScene(
            index=i,
            visual=visual,
            speech=speech,
            sounds=s.get("sounds", "silent, no sound"),
            duration=s.get("duration", 5.0),
        ))

    return Script(
        title=data.get("name", "Template Video"),
        persona_slug=persona.slug,
        scenes=scenes,
        product=data.get("product", "custom"),
        tone=data.get("tone", "casual"),
    )


def save_script(script: Script, output_path: Path) -> Path:
    """Save a script to JSON file."""
    data = {
        "title": script.title,
        "persona_slug": script.persona_slug,
        "product": script.product,
        "tone": script.tone,
        "scenes": [
            {
                "visual": s.visual,
                "speech": s.speech,
                "duration": s.duration,
            }
            for s in script.scenes
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path
