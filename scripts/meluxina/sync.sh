#!/usr/bin/env bash
# Copy this repo between the laptop and MeluXina. Run on the laptop.
#
#   scripts/meluxina/sync.sh push   # laptop -> MeluXina (code + patched tau2 submodule)
#   scripts/meluxina/sync.sh pull   # MeluXina -> laptop (results/ only)
#
# rsync rather than git clone, because the tau2 patches and run_tau2.py edits
# aren't necessarily committed/pushed, and the patched submodule state should
# match the laptop exactly. Needs the "meluxina" host in ~/.ssh/config.
set -euo pipefail

REMOTE_HOST=meluxina
REMOTE_REPO=/project/home/p201155/u104856/Misinformation-in-Agentic-Systems
LOCAL_REPO="$(cd "$(dirname "$0")/../.." && pwd)"

case "${1:-}" in
  push)
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO'"
    # .venv holds macOS binaries; the cluster builds its own (setup.sh).
    rsync -az --progress \
      --exclude '.venv/' --exclude '__pycache__/' --exclude '.DS_Store' \
      --exclude '/results/' --exclude '/results_archive/' --exclude '/logs/' \
      --exclude '/third_party/tau2-bench/data/simulations/' \
      "$LOCAL_REPO/" "$REMOTE_HOST:$REMOTE_REPO/"
    ;;
  pull)
    mkdir -p "$LOCAL_REPO/results"
    rsync -az --progress "$REMOTE_HOST:$REMOTE_REPO/results/" "$LOCAL_REPO/results/"
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 1
    ;;
esac
