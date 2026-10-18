#!/usr/bin/env python3
"""Run a tau2-bench (retail/telecom/airline/mock) evaluation against a local
Ollama model.

Usage:
    scripts/run_tau2.py <domain> [extra 'tau2 run' flags...]

Examples:
    scripts/run_tau2.py retail
    scripts/run_tau2.py airline_misinfo_user   # a condition from scripts/tau2_cli.py
    scripts/run_tau2.py telecom --num-tasks 20 --num-trials 2
    TAU2_MODEL=qwen2.5:14b scripts/run_tau2.py retail --num-tasks 5

Results are copied out of the tau2-bench submodule into results/<run-name>/
in this repo after the run finishes, so they're stored alongside the rest of
the project rather than buried in third-party code.

Env overrides:
    TAU2_MODEL            Ollama model tag (default: qwen2.5:32b)
    TAU2_NUM_CTX           Context window in tokens (default: 32768), built into a
                           derived Ollama model <model>-ctx<N> that the run uses
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
import re
import shutil
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TAU2_DIR = os.path.join(REPO_ROOT, "third_party", "tau2-bench")

# Local patches to the vendored tau2 (each .patch file explains why):
#   - the NL-assertions grading judge has no CLI override and defaults to a
#     real OpenAI model, so config.py points it at the local Ollama model
#     (and asks Ollama for JSON-constrained output);
#   - tau2 json.loads() the judge's raw reply, so fenced/prose/trailing-comma
#     replies crashed tasks as infrastructure_error; evaluator_nl_assertions.py
#     parses them tolerantly and shows the raw reply if it still can't;
#   - Ollama mishandles Qwen2.5 tool calls (drops calls to tools not in the
#     request, crashing the task), so llm_utils.py handles them client-side.
# Since third_party/tau2-bench is a submodule, these edits live only in this
# working copy and are wiped out by a fresh `git submodule update`/re-clone,
# so we reapply them here if needed. Each entry: (patch, patched file, marker).
PATCHES = [
    ("tau2_nl_assertions_local_judge.patch", "src/tau2/config.py", "_TAU2_JUDGE_MODEL"),
    (
        "tau2_nl_assertions_json_parse.patch",
        "src/tau2/evaluator/evaluator_nl_assertions.py",
        "TAU2_JUDGE_JSON_PARSE",
    ),
    (
        "tau2_qwen_tools_in_prompt.patch",
        "src/tau2/utils/llm_utils.py",
        "TAU2_QWEN_TOOLS_IN_PROMPT",
    ),
]


def ensure_patches() -> None:
    for patch, target, marker in PATCHES:
        with open(os.path.join(TAU2_DIR, target)) as f:
            if marker in f.read():
                continue
        print(f"Applying local patch: {patch}")
        subprocess.check_call(
            ["git", "apply", os.path.join(REPO_ROOT, "patches", patch)], cwd=TAU2_DIR
        )


def ollama_models(ollama_host: str) -> list[str]:
    with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=5) as resp:
        data = json.load(resp)
    return [m["name"] for m in data.get("models", [])]


def ensure_context_model(ollama_host: str, model: str, num_ctx: int) -> str:
    """Create (or refresh) <model>-ctx<num_ctx>: the same weights with num_ctx built in."""
    # Ollama's OpenAI-compatible endpoint ignores per-request num_ctx, so it must live in the model.
    derived = f"{model}-ctx{num_ctx}"
    body = {"model": derived, "from": model, "parameters": {"num_ctx": num_ctx}, "stream": False}
    req = urllib.request.Request(
        f"{ollama_host}/api/create",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = json.load(resp).get("status")
    if status != "success":
        raise RuntimeError(f"Ollama could not create {derived}: {status}")
    return derived


def split_per_task(results_path: str, domain: str, dest_dir: str) -> None:
    """Split a tau2 results.json into one <domain>_<task_id>_<condition>.json
    per task (all trials for that task included), each a self-contained
    results.json-shaped file (same schema, scoped to that task). The condition
    is "clean" for a base domain, or e.g. "misinfo_user" for the condition
    domains registered by tau2_cli.py (airline_misinfo_user -> airline_19_misinfo_user.json)."""
    match = re.fullmatch(r"(.+)_((?:misinfo|true)_(?:user|tool))", domain)
    domain, condition = (match.group(1), match.group(2)) if match else (domain, "clean")
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
        out_path = os.path.join(dest_dir, f"{domain}_{task_id}_{condition}.json")
        with open(out_path, "w") as f:
            json.dump(task_out, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a tau2-bench evaluation against a local Ollama model.",
        epilog="Unrecognized flags are passed through to 'tau2 run' (e.g. --num-tasks, --num-trials).",
    )
    parser.add_argument(
        "domain",
        help="retail | telecom | airline | mock | banking_knowledge, or a misinformation "
        "condition such as airline_misinfo_user (see scripts/tau2_cli.py)",
    )
    args, passthrough = parser.parse_known_args()

    model = os.environ.get("TAU2_MODEL", "qwen2.5:32b")
    num_ctx = int(os.environ.get("TAU2_NUM_CTX", "32768"))
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

    ensure_patches()
    run_model = ensure_context_model(ollama_host, model, num_ctx)

    env = dict(os.environ)
    # The NL-assertions judge patch reads these, so the judge shares the same model and context.
    env["TAU2_MODEL"] = run_model
    env["TAU2_NUM_CTX"] = str(num_ctx)
    if model.startswith("qwen2.5"):
        env["TAU2_QWEN_TOOLS_IN_PROMPT"] = "1"
        tool_mode = "client-side, Qwen2.5 chat template"
    else:
        tool_mode = (
            "Ollama's parser (WARNING: calls to tools not in the request "
            "come back empty and crash the task)"
        )

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
            "temperature": 0.0,  # tau2's default; these args replace its defaults wholesale
        }
    )

    run_name = "{}_{}_{}".format(
        args.domain,
        model.translate(str.maketrans(":/", "__")),
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    print(
        f"Domain: {args.domain} | Model: {run_model} (via {ollama_host}) | "
        f"context: {num_ctx} tokens | temperature: 0 | tools: {tool_mode} | save-to: {run_name}"
    )

    # tau2_cli.py is tau2's CLI plus our misinformation-condition domains.
    cmd = [
        "uv",
        "run",
        "python",
        os.path.join(SCRIPT_DIR, "tau2_cli.py"),
        "run",
        "--domain",
        args.domain,
        "--agent-llm",
        f"openai/{run_model}",
        "--agent-llm-args",
        llm_args,
        "--user-llm",
        f"openai/{run_model}",
        "--user-llm-args",
        llm_args,
        "--max-concurrency",
        max_concurrency,
        "--save-to",
        run_name,
        "--auto-resume",
        *passthrough,
    ]

    returncode = subprocess.call(cmd, cwd=TAU2_DIR, env=env)

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
        print(f"Per-task files written to: {dest_dir}")
    else:
        print(
            f"No results.json found at {run_dir}; nothing copied "
            "(run may have failed before producing output).",
            file=sys.stderr,
        )

    return returncode


if __name__ == "__main__":
    sys.exit(main())
