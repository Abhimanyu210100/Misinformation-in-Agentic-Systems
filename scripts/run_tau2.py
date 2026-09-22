#!/usr/bin/env python3
"""Run a tau2-bench (retail/telecom/airline/mock) evaluation against a local
Ollama model.

Usage:
    scripts/run_tau2.py <domain> [extra 'tau2 run' flags...]

Examples:
    scripts/run_tau2.py retail
    scripts/run_tau2.py telecom --num-tasks 20 --num-trials 2
    TAU2_MODEL=qwen2.5:14b scripts/run_tau2.py retail --num-tasks 5

Env overrides:
    TAU2_MODEL            Ollama model tag (default: qwen2.5:32b)
    TAU2_NUM_CTX           Context window passed to Ollama (default: 16384)
    TAU2_MAX_CONCURRENCY   Concurrent simulations (default: 1; local Ollama
                           serves one big model at a time, so keep this low
                           unless OLLAMA_NUM_PARALLEL is configured)
    OLLAMA_HOST            Ollama server base URL (default: http://localhost:11434)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAU2_DIR = os.path.join(SCRIPT_DIR, "..", "third_party", "tau2-bench")


def ollama_models(ollama_host: str) -> list[str]:
    with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=5) as resp:
        data = json.load(resp)
    return [m["name"] for m in data.get("models", [])]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a tau2-bench evaluation against a local Ollama model.",
        epilog="Unrecognized flags are passed through to 'tau2 run' (e.g. --num-tasks, --num-trials).",
    )
    parser.add_argument("domain", help="retail | telecom | airline | mock")
    args, passthrough = parser.parse_known_args()

    model = os.environ.get("TAU2_MODEL", "qwen2.5:32b")
    num_ctx = int(os.environ.get("TAU2_NUM_CTX", "16384"))
    max_concurrency = os.environ.get("TAU2_MAX_CONCURRENCY", "1")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    if not os.path.isdir(TAU2_DIR):
        print(
            f"tau2-bench submodule not found at {TAU2_DIR}. Run: git submodule update --init",
            file=sys.stderr,
        )
        return 1

    try:
        models = ollama_models(ollama_host)
    except Exception as e:
        print(
            f"Ollama server not reachable at {ollama_host} ({e}). "
            "Start it with 'ollama serve' (or open the Ollama app).",
            file=sys.stderr,
        )
        return 1

    if model not in models:
        print(
            f"Model '{model}' not found locally. Pull it first: ollama pull {model}",
            file=sys.stderr,
        )
        return 1

    # NOTE: we deliberately use the "openai/" provider pointed at Ollama's
    # OpenAI-compatible endpoint (/v1) rather than litellm's native
    # "ollama_chat/" provider. As of litellm's current release, ollama_chat's
    # request transformation drops `tool_calls` from assistant messages when
    # re-serializing multi-turn history (see BerriAI/litellm#26094 and #18922),
    # which silently breaks tau2's tool-calling conversations after a couple of
    # turns (agent/user messages come back with empty content AND no tool
    # calls). The OpenAI-compatible endpoint does not go through that broken
    # code path and was verified to preserve tool_calls correctly.
    llm_args = json.dumps(
        {
            "api_base": f"{ollama_host}/v1",
            "api_key": "ollama",
            "extra_body": {"options": {"num_ctx": num_ctx}},
        }
    )

    run_name = "{}_{}_{}".format(
        args.domain,
        model.translate(str.maketrans(":/", "__")),
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    print(
        f"Domain: {args.domain} | Model: {model} (via {ollama_host}) | "
        f"num_ctx: {num_ctx} | save-to: {run_name}"
    )

    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        args.domain,
        "--agent-llm",
        f"openai/{model}",
        "--agent-llm-args",
        llm_args,
        "--user-llm",
        f"openai/{model}",
        "--user-llm-args",
        llm_args,
        "--max-concurrency",
        max_concurrency,
        "--save-to",
        run_name,
        "--auto-resume",
        *passthrough,
    ]

    return subprocess.call(cmd, cwd=TAU2_DIR)


if __name__ == "__main__":
    sys.exit(main())
