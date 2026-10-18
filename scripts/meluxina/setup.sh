#!/usr/bin/env bash
# One-time MeluXina setup. Run on a LOGIN node (they have internet; GPU nodes
# may not), from the synced repo:
#
#   ssh meluxina
#   bash /project/home/p201155/$USER/Misinformation-in-Agentic-Systems/scripts/meluxina/setup.sh
#
# Installs, all under /project/home/p201155/$USER (home is capped at 100 GB):
#   - Ollama 0.33.1, the same version as the laptop runs (no root needed: it's
#     a self-contained tarball with its own CUDA libs);
#   - uv, and tau2's Python env (with the `knowledge` extra for banking_knowledge);
#   - the model weights: qwen2.5:72b (Qwen2.5-72B-Instruct, the agent), plus
#     llama3.3:70b and gpt-oss:120b (misinformation generator and judge, see
#     misinfo.sbatch). All pinned.
# Safe to re-run: every step is skipped if already done.
set -euo pipefail

OLLAMA_VERSION=0.33.1
MODEL=qwen2.5:72b  # pinned: ignores any TAU2_MODEL in the environment
MISINFO_MODELS=(llama3.3:70b gpt-oss:120b)  # generator, judge
WORK=/project/home/p201155/$USER
REPO="$WORK/Misinformation-in-Agentic-Systems"
OLLAMA_DIR="$WORK/ollama-$OLLAMA_VERSION"
export OLLAMA_MODELS="$WORK/ollama-models"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$WORK" "$OLLAMA_MODELS" "$REPO/logs/meluxina"  # sbatch --output needs logs/meluxina to exist

if [ ! -d "$REPO/third_party/tau2-bench" ]; then
  echo "Repo not found at $REPO. On the laptop, run: scripts/meluxina/sync.sh push" >&2
  exit 1
fi

# --- uv ---------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# --- Ollama -----------------------------------------------------------------
# Linux releases ship as .tar.zst and the login nodes have no zstd binary, so
# decompress with Python's zstandard package via uv.
if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  echo "Installing Ollama $OLLAMA_VERSION into $OLLAMA_DIR..."
  mkdir -p "$OLLAMA_DIR"
  archive="$WORK/ollama-linux-amd64-$OLLAMA_VERSION.tar.zst"
  curl -fL -o "$archive" \
    "https://github.com/ollama/ollama/releases/download/v$OLLAMA_VERSION/ollama-linux-amd64.tar.zst"
  uv run --no-project --python 3.12 --with zstandard python - "$archive" "$OLLAMA_DIR" <<'EOF'
import sys, tarfile, zstandard
src, dest = sys.argv[1], sys.argv[2]
with open(src, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as zf:
    with tarfile.open(fileobj=zf, mode="r|") as tar:
        tar.extractall(dest)
EOF
  rm -f "$archive"
fi
"$OLLAMA_DIR/bin/ollama" --version

# --- tau2 Python env --------------------------------------------------------
# --inexact keeps the `knowledge` extra (see run_tau2.py / banking_knowledge).
echo "Syncing tau2 Python env..."
uv sync --directory "$REPO/third_party/tau2-bench" --extra knowledge --inexact
# tau2 imports websockets unconditionally (tau2.voice), but the lock only pulls
# it in with the `voice` extra. The laptop env has it installed separately;
# match its version.
uv pip install --python "$REPO/third_party/tau2-bench/.venv" websockets==17.1

# --- Model weights ----------------------------------------------------------
# A short-lived CPU-only server on the login node, just to download weights
# into the shared OLLAMA_MODELS dir. Unusual port to avoid other users' servers.
PORT=$((20000 + RANDOM % 10000))
export OLLAMA_HOST="127.0.0.1:$PORT"
"$OLLAMA_DIR/bin/ollama" serve >"$WORK/ollama-setup.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
for _ in $(seq 60); do
  curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break
  sleep 1
done
echo "Pulling $MODEL (about 47 GB for qwen2.5:72b)..."
"$OLLAMA_DIR/bin/ollama" pull "$MODEL"
echo "Pulling ${MISINFO_MODELS[*]} (about 43 GB + 65 GB)..."
for m in "${MISINFO_MODELS[@]}"; do
  "$OLLAMA_DIR/bin/ollama" pull "$m"
done
"$OLLAMA_DIR/bin/ollama" list

echo
echo "Setup done. Submit a job with:"
echo "  cd $REPO && sbatch scripts/meluxina/tau2_ollama.sbatch retail"
