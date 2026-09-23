#!/usr/bin/env python3
"""Run a tau2-bench (retail/telecom/airline/mock) evaluation against a local
Ollama model.

Usage:
    scripts/run_tau2.py <domain> [extra 'tau2 run' flags...]

Examples:
    scripts/run_tau2.py retail
    scripts/run_tau2.py telecom --num-tasks 20 --num-trials 2
    TAU2_MODEL=qwen2.5:14b scripts/run_tau2.py retail --num-tasks 5

Results are copied out of the tau2-bench submodule into results/<run-name>/
in this repo after the run finishes, so they're stored alongside the rest of
the project rather than buried in third-party code.

Env overrides:
    TAU2_MODEL            Ollama model tag (default: qwen2.5:32b)
    TAU2_NUM_CTX           Context window passed to Ollama (default: 16384)
    TAU2_MAX_CONCURRENCY   Concurrent simulations (default: 1; local Ollama
                           serves one big model at a time, so keep this low
                           unless OLLAMA_NUM_PARALLEL is configured)
    OLLAMA_HOST            Ollama server base URL (default: http://localhost:11434)
    TAU2_RESULTS_DIR       Where finished runs are copied to (default: results/
                           at the repo root)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TAU2_DIR = os.path.join(REPO_ROOT, "third_party", "tau2-bench")

# tau2's NL-assertions grading judge has no CLI override and defaults to a
# real OpenAI model (see patches/tau2_nl_assertions_local_judge.patch for
# the full story). We patch the vendored config.py to point it at the local
# Ollama model instead. Since third_party/tau2-bench is a submodule, that
# edit lives only in this working copy and is wiped out by a fresh
# `git submodule update`/re-clone, so we reapply it here if needed.
NL_ASSERTIONS_PATCH = os.path.join(
    REPO_ROOT, "patches", "tau2_nl_assertions_local_judge.patch"
)
NL_ASSERTIONS_PATCH_MARKER = "_TAU2_JUDGE_MODEL"


def ensure_nl_assertions_patch() -> None:
    config_path = os.path.join(TAU2_DIR, "src", "tau2", "config.py")
    with open(config_path) as f:
        if NL_ASSERTIONS_PATCH_MARKER in f.read():
            return
    print("Applying local patch: NL-assertions judge -> local Ollama model")
    subprocess.check_call(
        ["git", "apply", NL_ASSERTIONS_PATCH], cwd=TAU2_DIR
    )


def ollama_models(ollama_host: str) -> list[str]:
    with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=5) as resp:
        data = json.load(resp)
    return [m["name"] for m in data.get("models", [])]


def split_per_task(results_path: str, domain: str, dest_dir: str) -> None:
    """Split a tau2 results.json into one <domain>_<task_id>_clean.json per
    task (all trials for that task included), each a self-contained
    results.json-shaped file (same schema, scoped to that task)."""
    with open(results_path) as f:
        results = json.load(f)

    tasks_by_id = {task["id"]: task for task in results["tasks"]}
    sims_by_task: dict[str, list] = {}
    for sim in results["simulations"]:
        sims_by_task.setdefault(sim["task_id"], []).append(sim)

    for task_id, sims in sims_by_task.items():
        task_out = {
            "timestamp": results.get("timestamp"),
            "info": results.get("info"),
            "tasks": [tasks_by_id[task_id]] if task_id in tasks_by_id else [],
            "simulations": sims,
            "simulation_index": None,
        }
        out_path = os.path.join(dest_dir, f"{domain}_{task_id}_clean.json")
        with open(out_path, "w") as f:
            json.dump(task_out, f, indent=2)


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

    ensure_nl_assertions_patch()

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

    returncode = subprocess.call(cmd, cwd=TAU2_DIR)

    run_dir = os.path.join(TAU2_DIR, "data", "simulations", run_name)
    if os.path.isfile(os.path.join(run_dir, "results.json")):
        results_root = os.environ.get(
            "TAU2_RESULTS_DIR", os.path.join(REPO_ROOT, "results")
        )
        dest_dir = os.path.join(results_root, run_name)
        shutil.copytree(run_dir, dest_dir, dirs_exist_ok=True)
        split_per_task(
            os.path.join(dest_dir, "results.json"), args.domain, dest_dir
        )
        print(f"Results copied to: {dest_dir}")
        print(f"Per-task files written as: {dest_dir}/{args.domain}_<task_id>_clean.json")
    else:
        print(
            f"No results.json found at {run_dir}; nothing copied "
            "(run may have failed before producing output).",
            file=sys.stderr,
        )

    return returncode


if __name__ == "__main__":
    sys.exit(main())
