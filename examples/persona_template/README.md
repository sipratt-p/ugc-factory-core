# Persona template

Copy this directory to `personas/<slug>/`, then add:

- `reference.jpg`: a real, well-lit head-and-shoulders photo, portrait orientation
  (3:4 or taller). Square or landscape images are center-cropped to 3:4 and given
  light noise + JPEG compression automatically (`factory/audio_utils.py`).
- `reference.wav`: 3 to 15 seconds of one speaker. Anything is normalized to
  48 kHz stereo and trimmed to its loudest 5 second window.

`scene_bible.json` is optional. If it is missing, `bible_from_persona_dir()` uses
the generic bible, and `persona_manager.create_persona()` can write one for you
from the photo when `ANTHROPIC_API_KEY` is set. Keep `[SOUNDS]` fields minimal
(`close mic`, `measured pace`, `dry room, neutral studio ambience`): describing the
voice's timbre there re-randomizes the voice per clip. The reference audio, not the
prompt, carries the voice.

`reference_image` and `reference_audio` are filled in at load time; leave them empty.
