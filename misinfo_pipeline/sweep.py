"""Run the sweep: every (model, task, condition, seed) cell, resumably.

    python -m misinfo_pipeline.sweep
    python -m misinfo_pipeline.sweep --limit 4        # smoke test
    python -m misinfo_pipeline.sweep --models m1 m2   # override the lineup

Conditions per task: 1 clean + 8 user-channel + 8 document-channel = 17.

tau2 checkpoints and auto-resumes *within* a single run config, but this sweep
spans conditions and models, so the outer loop keeps its own manifest: one line
per completed cell, checked before dispatch. A sweep this size will be
interrupted -- the manifest is what makes restarting cheap rather than
catastrophic.

Models are iterated outermost so each one is loaded and served once, rather
than being swapped in and out per task.
"""

import argparse
import json

from tau2.data_model.simulation import TextRunConfig
from tau2.run import get_tasks, run_single_task

from .config import (
    AGENT_MODEL,
    CLAIMS_VERIFIED_PATH,
    DOMAIN,
    MANIFEST_PATH,
    RETRIEVAL_VARIANT,
    SEEDS,
    USER_MODEL,
)
from .injection import inject_user_claim, pin_default_retrieval_variant, register_domain
from .trace import save_run


def load_claims() -> dict[str, list[dict]]:
    """Approved claims, grouped by task id."""
    if not CLAIMS_VERIFIED_PATH.exists():
        raise SystemExit(
            f"No verified claims at {CLAIMS_VERIFIED_PATH}.\n"
            "Run: python -m misinfo_pipeline.generate && python -m misinfo_pipeline.verify"
        )
    claims = json.loads(CLAIMS_VERIFIED_PATH.read_text())
    grouped: dict[str, list[dict]] = {}
    for claim in claims:
        grouped.setdefault(claim["task_id"], []).append(claim)
    return grouped


def cell_id(model: str, task_id: str, condition: str, seed: int) -> str:
    return f"{model}|{task_id}|{condition}|{seed}"


def load_manifest() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    return {
        json.loads(line)["cell"]
        for line in MANIFEST_PATH.read_text().splitlines()
        if line.strip()
    }


def record_done(cell: str) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a") as fp:
        fp.write(json.dumps({"cell": cell}) + "\n")


def build_conditions(task_claims: list[dict]) -> list[tuple[str, dict | None]]:
    """All conditions for one task: clean, then each channel x strategy."""
    conditions: list[tuple[str, dict | None]] = [("clean", None)]
    for claim in sorted(task_claims, key=lambda c: c["strategy"]):
        for channel in ("user", "document"):
            conditions.append((f"{channel}/{claim['strategy']}", {**claim, "channel": channel}))
    return conditions


def run_cell(task, condition: str, claim: dict | None, model: str, seed: int):
    """Dispatch one cell and return ``(sim, provenance)``."""
    if claim is None:
        run_task = task
        domain = register_domain(RETRIEVAL_VARIANT)
        provenance = {"channel": None, "strategy": None, "injected_text": None}
    else:
        text = claim["misinformation_text"]
        if claim["channel"] == "user":
            run_task = inject_user_claim(task, text)
            domain = register_domain(RETRIEVAL_VARIANT)
            target_doc = None
        else:
            run_task = task
            target_doc = claim["target_document"]
            domain = register_domain(RETRIEVAL_VARIANT, poisoned_doc=(target_doc, text))

        provenance = {
            "channel": claim["channel"],
            "strategy": claim["strategy"],
            "injected_text": text,
            "false_fact": claim["false_fact"],
            "target_document": target_doc,
            "generator_model": claim["generator_model"],
            "domain": domain,
        }

    config = TextRunConfig(
        domain=domain,
        llm_agent=model,
        llm_user=USER_MODEL,
        max_concurrency=1,
    )
    sim = run_single_task(config, run_task, seed=seed)
    return sim, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[AGENT_MODEL])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--limit", type=int, default=None, help="Cap cells, for smoke tests")
    args = parser.parse_args()

    pin_default_retrieval_variant()

    claims_by_task = load_claims()
    tasks = {t.id: t for t in get_tasks(DOMAIN) if t.id in claims_by_task}
    done = load_manifest()

    planned = [
        (model, task_id, condition, claim, seed)
        for model in args.models  # outermost: load each model once
        for task_id, task in tasks.items()
        for condition, claim in build_conditions(claims_by_task[task_id])
        for seed in args.seeds
    ]
    todo = [p for p in planned if cell_id(p[0], p[1], p[2], p[4]) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(planned)} cells planned, {len(planned) - len(todo)} already done, running {len(todo)}")

    for i, (model, task_id, condition, claim, seed) in enumerate(todo, 1):
        cell = cell_id(model, task_id, condition, seed)
        print(f"[{i}/{len(todo)}] {cell}")
        try:
            sim, provenance = run_cell(tasks[task_id], condition, claim, model, seed)
        except Exception as exc:
            # One bad cell must not abandon a sweep of thousands. It stays out
            # of the manifest, so a rerun retries exactly the failures.
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            continue

        save_run(
            sim,
            model=model,
            task_id=task_id,
            condition=condition,
            seed=seed,
            provenance=provenance,
        )
        record_done(cell)

    print("Done.")


if __name__ == "__main__":
    main()
