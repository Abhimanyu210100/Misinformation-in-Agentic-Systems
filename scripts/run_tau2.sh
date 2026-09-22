#!/usr/bin/env bash
# Run a tau2-bench (retail/telecom/airline/mock) evaluation against a local
# Ollama model.
#
# Usage:
#   scripts/run_tau2.sh <domain> [extra 'tau2 run' flags...]
#
# Examples:
#   scripts/run_tau2.sh retail
#   scripts/run_tau2.sh telecom --num-tasks 20 --num-trials 2
#   TAU2_MODEL=qwen2.5:14b scripts/run_tau2.sh retail --num-tasks 5
#
# Env overrides:
#   TAU2_MODEL         Ollama model tag (default: qwen2.5:32b)
#   TAU2_NUM_CTX        Context window passed to Ollama (default: 16384)
#   TAU2_MAX_CONCURRENCY Concurrent simulations (default: 1; local Ollama
#                        serves one big model at a time, so keep this low
#                        unless OLLAMA_NUM_PARALLEL is configured)
#   OLLAMA_HOST         Ollama server base URL (default: http://localhost:11434)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <domain: retail|telecom|airline|mock> [extra 'tau2 run' flags...]" >&2
  exit 1
fi

DOMAIN="$1"; shift

MODEL="${TAU2_MODEL:-qwen2.5:32b}"
NUM_CTX="${TAU2_NUM_CTX:-16384}"
MAX_CONCURRENCY="${TAU2_MAX_CONCURRENCY:-1}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU2_DIR="$SCRIPT_DIR/../third_party/tau2-bench"

if [[ ! -d "$TAU2_DIR" ]]; then
  echo "tau2-bench submodule not found at $TAU2_DIR. Run: git submodule update --init" >&2
  exit 1
fi

if ! curl -sf "$OLLAMA_HOST/api/tags" > /dev/null; then
  echo "Ollama server not reachable at $OLLAMA_HOST. Start it with 'ollama serve' (or open the Ollama app)." >&2
  exit 1
fi

if ! curl -sf "$OLLAMA_HOST/api/tags" | grep -q "\"$MODEL\""; then
  echo "Model '$MODEL' not found locally. Pull it first: ollama pull $MODEL" >&2
  exit 1
fi

# NOTE: we deliberately use the "openai/" provider pointed at Ollama's
# OpenAI-compatible endpoint (/v1) rather than litellm's native
# "ollama_chat/" provider. As of litellm's current release, ollama_chat's
# request transformation drops `tool_calls` from assistant messages when
# re-serializing multi-turn history (see BerriAI/litellm#26094 and #18922),
# which silently breaks tau2's tool-calling conversations after a couple of
# turns (agent/user messages come back with empty content AND no tool
# calls). The OpenAI-compatible endpoint does not go through that broken
# code path and was verified to preserve tool_calls correctly.
LLM_ARGS=$(python3 -c "
import json, sys
print(json.dumps({
    'api_base': sys.argv[1] + '/v1',
    'api_key': 'ollama',
    'extra_body': {'options': {'num_ctx': int(sys.argv[2])}},
}))
" "$OLLAMA_HOST" "$NUM_CTX")

RUN_NAME="${DOMAIN}_$(echo "$MODEL" | tr ':/' '__')_$(date +%Y%m%d_%H%M%S)"

echo "Domain: $DOMAIN | Model: $MODEL (via $OLLAMA_HOST) | num_ctx: $NUM_CTX | save-to: $RUN_NAME"

cd "$TAU2_DIR"
exec uv run tau2 run \
  --domain "$DOMAIN" \
  --agent-llm "openai/${MODEL}" \
  --agent-llm-args "$LLM_ARGS" \
  --user-llm "openai/${MODEL}" \
  --user-llm-args "$LLM_ARGS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --save-to "$RUN_NAME" \
  --auto-resume \
  "$@"
