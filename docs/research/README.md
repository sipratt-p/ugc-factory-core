# Research notes (curated subset)

The project kept a Karpathy-style knowledge base next to the code: raw sources,
an LLM-maintained wiki with entity / concept / experiment pages, and a TSV of
experiment results. This directory is the generic part of that wiki.

Included:

- `metrics.md`: what every score measures, its range, and why it is weighted the way it is.
- `metrics-v11-scoring-stack.md`: the full evaluation fleet by category (visual, identity, audio, sync, seam, temporal).
- `overview.md`: the V4-era best pipeline and the ranked findings up to 51 experiments.
- `roadmap.md`: what is validated, what is ready to implement, and which experiments diverged from the known-good path.
- `experiments/`: write-ups for the V2 to V10 sweeps plus the 89-experiment cross-version leaderboard and the V1 `results.tsv`.

Not included: the chronological research log, the human feedback log, external
LLM advice transcripts, and creative side projects. Pages referring to
`<UGC_RESULTS_DIR>` point at generated media that is not part of this repository.

Labels that named a client were generalized to the ad vertical (recruiting,
health, seo) before publication; scores are unchanged.

Conventions kept from the original wiki: pages carry YAML frontmatter
(`type`, `created`, `updated`, `tags`); the runners append to `log.md` through
`factory/wiki_hook.py` (env `UGC_WIKI_DIR` relocates it).
