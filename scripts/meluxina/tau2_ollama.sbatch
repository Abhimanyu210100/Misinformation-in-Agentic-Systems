#!/bin/bash -l
#SBATCH --job-name=tau2-ollama
#SBATCH --account=p201155
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/meluxina/%x.%j.out
#SBATCH --mail-user=abhiswaroop210100@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Run tau2-bench on one MeluXina GPU node (4x A100 40GB) against Ollama.
#
#   sbatch scripts/meluxina/tau2_ollama.sbatch <domain> [<domain>] [-- <tau2 run flags>]
#
# Examples (submit from the repo root on MeluXina, after setup.sh):
#   sbatch scripts/meluxina/tau2_ollama.sbatch airline
#   sbatch scripts/meluxina/tau2_ollama.sbatch retail telecom
#   sbatch scripts/meluxina/tau2_ollama.sbatch airline -- --num-trials 2
# Quick test on two tasks only:
#   TAU2_TASK_IDS="0 1" sbatch --qos=test --time=00:30:00 scripts/meluxina/tau2_ollama.sbatch airline
#
# Every run gets --seed 42 (flags after "--" are added on top; a later --seed wins).
#
# Model: qwen2.5:72b (Qwen2.5-72B-Instruct, Q4_K_M, 47 GB), pinned. With the
# 32K context it needs ~58 GB, so each Ollama server spans 2 GPUs, and up to 2
# domains run in parallel (server 0: GPUs 0,1, port 11434; server 1: GPUs 2,3,
# port 11435). Each run_tau2.py keeps concurrency 1: one Ollama 0.33.1 server,
# one request at a time, temperature 0. The whole node is billed regardless,
# so give it 2 domains when you can.
#
# Env override: TAU2_NUM_CTX (passed through to run_tau2.py).
# Logs: logs/meluxina/<jobid>/{ollama_server<N>,tau2_<domain>}.log
# Results: results/<run-name>/ as usual; copy back with `scripts/meluxina/sync.sh pull`.
set -euo pipefail

OLLAMA_VERSION=0.33.1
WORK=/project/home/p201155/$USER
REPO="$WORK/Misinformation-in-Agentic-Systems"
OLLAMA_BIN="$WORK/ollama-$OLLAMA_VERSION/bin/ollama"
NUM_GPUS=4
GPUS_PER_SERVER=2
MAX_DOMAINS=$((NUM_GPUS / GPUS_PER_SERVER))
BASE_PORT=11434

export OLLAMA_MODELS="$WORK/ollama-models"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export PATH="$HOME/.local/bin:$PATH"
MODEL=qwen2.5:72b  # pinned: ignores any TAU2_MODEL in the environment
NUM_CTX="${TAU2_NUM_CTX:-32768}"

# --- Arguments: domains, then optional "--" and flags for every tau2 run ------
DOMAINS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  DOMAINS+=("$1")
  shift
done
[ "${1:-}" = "--" ] && shift
EXTRA_FLAGS=(--seed 42 "$@")
# Test mode: run only the given task IDs, e.g. TAU2_TASK_IDS="0 1" sbatch ...
if [ -n "${TAU2_TASK_IDS:-}" ]; then
  read -ra task_ids <<<"$TAU2_TASK_IDS"
  EXTRA_FLAGS+=(--task-ids "${task_ids[@]}")
fi

if [ ${#DOMAINS[@]} -eq 0 ] || [ ${#DOMAINS[@]} -gt $MAX_DOMAINS ]; then
  echo "Give 1-$MAX_DOMAINS domains ($GPUS_PER_SERVER GPUs each), e.g.: sbatch $0 airline retail" >&2
  exit 1
fi
if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Ollama not found at $OLLAMA_BIN. Run scripts/meluxina/setup.sh on a login node first." >&2
  exit 1
fi

cd "$REPO"
LOG_DIR="$REPO/logs/meluxina/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) | model $MODEL | ctx $NUM_CTX | domains: ${DOMAINS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# --- Start one Ollama server per domain, each on its own pair of GPUs ---------
SERVER_PIDS=()
cleanup() { kill "${SERVER_PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT

for i in "${!DOMAINS[@]}"; do
  port=$((BASE_PORT + i))
  gpus=$(seq -s, $((i * GPUS_PER_SERVER)) $((i * GPUS_PER_SERVER + GPUS_PER_SERVER - 1)))
  # Pin each server to its GPUs. CUDA_VISIBLE_DEVICES alone isn't enough: Ollama
  # 0.33.1 also probes GPUs via Vulkan (and Slurm sets GPU_DEVICE_ORDINAL to
  # all 4), which let the server see all 4 A100s. Turn Vulkan off and narrow
  # every device list to this server's GPUs.
  CUDA_VISIBLE_DEVICES=$gpus \
  GPU_DEVICE_ORDINAL=$gpus \
  GGML_VK_VISIBLE_DEVICES=$gpus \
  OLLAMA_VULKAN=0 \
  OLLAMA_HOST="127.0.0.1:$port" \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=-1 \
    "$OLLAMA_BIN" serve >"$LOG_DIR/ollama_server$i.log" 2>&1 &
  SERVER_PIDS+=($!)
done

for i in "${!DOMAINS[@]}"; do
  port=$((BASE_PORT + i))
  for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$port/api/tags" >/dev/null && break
    sleep 1
  done
  curl -sf "http://127.0.0.1:$port/api/tags" >/dev/null || {
    echo "Ollama server $i (port $port) did not start; see $LOG_DIR/ollama_server$i.log" >&2
    exit 1
  }
done

# run_tau2.py creates <model>-ctx<N> on every run. All servers share
# OLLAMA_MODELS, so create it once here, before the runs start in parallel,
# to avoid concurrent writes to the same manifest.
curl -sf "http://127.0.0.1:$BASE_PORT/api/create" \
  -d "{\"model\": \"$MODEL-ctx$NUM_CTX\", \"from\": \"$MODEL\", \"parameters\": {\"num_ctx\": $NUM_CTX}, \"stream\": false}" \
  >/dev/null

# --- One run_tau2.py per domain, each against its own server -------------------
RUN_PIDS=()
for i in "${!DOMAINS[@]}"; do
  domain=${DOMAINS[$i]}
  flags=("${EXTRA_FLAGS[@]}")
  # The default retrieval config needs OPENAI_API_KEY (see run_tau2.py notes).
  [ "$domain" = "banking_knowledge" ] && flags=(--retrieval-config bm25 "${flags[@]}")
  OLLAMA_HOST="http://127.0.0.1:$((BASE_PORT + i))" \
  TAU2_MODEL="$MODEL" \
  TAU2_NUM_CTX="$NUM_CTX" \
  TAU2_MAX_CONCURRENCY=1 \
    uv run --no-project --python 3.12 python scripts/run_tau2.py "$domain" "${flags[@]}" \
    >"$LOG_DIR/tau2_$domain.log" 2>&1 &
  RUN_PIDS+=($!)
  echo "Started $domain on Ollama server $i (pid $!), log: $LOG_DIR/tau2_$domain.log"
done

STATUS=0
for i in "${!DOMAINS[@]}"; do
  if wait "${RUN_PIDS[$i]}"; then
    echo "${DOMAINS[$i]}: finished OK"
  else
    echo "${DOMAINS[$i]}: FAILED (exit $?), see $LOG_DIR/tau2_${DOMAINS[$i]}.log"
    STATUS=1
  fi
done
exit $STATUS
