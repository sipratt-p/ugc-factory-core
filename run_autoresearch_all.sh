#!/bin/bash
# Run the two leaderboard phases back to back.
# Activate whatever environment holds the requirements first, and export
# UGC_REF_IMAGE / UGC_REF_AUDIO (see run_autoresearch.py).
set -e
cd "$(dirname "$0")"

echo "=== Phase 1 starting ==="
python run_autoresearch.py
echo "=== Phase 1 done ==="

echo "=== Phase 2 starting ==="
python run_autoresearch_phase2.py
echo "=== Phase 2 done ==="

echo "=== ALL EXPERIMENTS COMPLETE ==="
