"""Persist runs: native trace, injection provenance, and a flat index row.

Three artifacts per run, each earning its place:

- ``simulation.json`` -- tau2's ``SimulationRun`` serialized untouched. Lossless,
  so Experiment 2 can ask questions nobody has thought of yet. It already holds
  every tool call, every tool result, the reward, the seed, and the full raw LLM
  response per assistant turn (``raw_data``, set at
  tau2-bench/src/tau2/utils/llm_utils.py:452) -- which is where reasoning content
  from vLLM-served reasoning models lands.
- ``injection.json`` -- what was injected, where, and which claim record it came
  from. This is the one thing that is genuinely unreconstructable later: once the
  run is over, nothing in the trace records what the document originally said.
- a row appended to ``index.jsonl`` -- the flat table analysis actually queries,
  so a sweep-wide question doesn't require re-parsing thousands of full traces.
"""

import json
from pathlib import Path

from .config import INDEX_PATH, RUNS_DIR


def model_slug(model: str) -> str:
    """Filesystem-safe directory name for a litellm model string."""
    return model.replace("/", "__").replace(":", "_")


def run_dir(model: str, task_id: str, condition: str, seed: int) -> Path:
    return RUNS_DIR / model_slug(model) / task_id / condition.replace("/", "_") / str(seed)


def tool_calls_of(sim) -> list[str]:
    return [
        tc.name
        for m in (sim.messages or [])
        for tc in (getattr(m, "tool_calls", None) or [])
    ]


def detect_exposure(sim, injected_text: str | None, channel: str | None) -> bool | None:
    """Did the misinformation actually reach the agent?

    For the document channel this is the crux: the poisoned document sits in the
    knowledge base, but the agent only sees it if one of its own ``grep`` calls
    returns it. Exposure is therefore measured, not assumed, and runs where the
    agent never retrieved the poison have to be analyzed separately -- they say
    nothing about credulity.

    Matching uses a distinctive slice of the injected text rather than the whole
    string, because retrieval tools truncate and reformat what they return.
    """
    if channel is None or not injected_text:
        return None

    needle = " ".join(injected_text.split())[:60].strip()
    if not needle:
        return None

    for message in sim.messages or []:
        role = getattr(message, "role", None)
        if channel == "document" and role != "tool":
            continue
        if channel == "user" and role != "user":
            continue
        content = " ".join((getattr(message, "content", None) or "").split())
        if needle and needle in content:
            return True
    return False


def save_run(
    sim,
    *,
    model: str,
    task_id: str,
    condition: str,
    seed: int,
    provenance: dict,
) -> Path:
    """Write all three artifacts for one run and return its directory."""
    directory = run_dir(model, task_id, condition, seed)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "simulation.json").write_text(
        json.dumps(sim.model_dump(mode="json"), indent=2)
    )
    (directory / "injection.json").write_text(json.dumps(provenance, indent=2))

    exposed = detect_exposure(
        sim, provenance.get("injected_text"), provenance.get("channel")
    )

    row = {
        "model": model,
        "task_id": task_id,
        "condition": condition,
        "channel": provenance.get("channel"),
        "strategy": provenance.get("strategy"),
        "seed": seed,
        "reward": sim.reward_info.reward if sim.reward_info else None,
        "exposed": exposed,
        "termination_reason": str(sim.termination_reason),
        "num_messages": len(sim.messages) if sim.messages else 0,
        "tool_calls": tool_calls_of(sim),
        "duration_s": round(sim.duration, 2) if sim.duration else None,
        "agent_cost": sim.agent_cost,
        "run_dir": str(directory.relative_to(RUNS_DIR)),
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("a") as fp:
        fp.write(json.dumps(row) + "\n")

    return directory
