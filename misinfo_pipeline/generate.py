"""Step 0: generate the misinformation claim set.

Run once, review once, then freeze. Every later run reads this artifact, so the
misinformation is byte-identical across models, seeds and conditions -- which is
what makes the paired clean-vs-misinformed comparison valid.

    python -m misinfo_pipeline.generate

Produces ``artifacts/claims_raw.json``: one record per (task, strategy), each
carrying the stage-1 false fact, the stage-2 styled text, and the prompts that
produced them. Nothing here is trusted until ``verify.py`` has run and you have
signed off.
"""

import json

import litellm

from .config import (
    CLAIMS_RAW_PATH,
    GENERATOR_MODEL,
    MINT_MAX_TOKENS,
    MINT_REPETITION_PENALTY,
    MINT_TEMPERATURE,
    MINT_TOP_P,
)
from .mint import STRATEGY_DESCRIPTIONS, stage1_prompt, stage2_prompt
from .tasks import (
    load_documents,
    render_documents,
    render_gold_actions,
    render_scenario,
    required_document_ids,
    sample_tasks,
    save_task_sample,
)


def _complete(prompt: str) -> str:
    """One generation call using MINT's sampling parameters.

    ``repetition_penalty`` is passed through litellm's extra-params path since
    it is a native HF/vLLM sampler argument rather than an OpenAI one; providers
    that don't support it ignore it.
    """
    resp = litellm.completion(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=MINT_TEMPERATURE,
        top_p=MINT_TOP_P,
        max_tokens=MINT_MAX_TOKENS,
        repetition_penalty=MINT_REPETITION_PENALTY,
    )
    return resp.choices[0].message.content.strip().strip('"').strip()


def generate_for_task(task, documents: dict[str, dict]) -> list[dict]:
    """Generate one false fact for a task, then wrap it in all 8 strategies."""
    scenario = render_scenario(task)
    docs_rendered = render_documents(task, documents)
    gold_actions = render_gold_actions(task)

    # Stage 1: one false fact per task, shared by every strategy. Generating it
    # once (rather than per strategy) is what makes the 8 variants differ only
    # in rhetorical framing, so strategy is cleanly isolated in the sweep.
    p1 = stage1_prompt(scenario, docs_rendered, gold_actions)
    false_fact = _complete(p1)

    records = []
    for strategy in STRATEGY_DESCRIPTIONS:
        p2 = stage2_prompt(false_fact, scenario, gold_actions, strategy)
        text = _complete(p2)
        records.append(
            {
                "task_id": task.id,
                "domain": "banking_knowledge",
                "strategy": strategy,
                "false_fact": false_fact,
                "misinformation_text": text,
                "required_documents": required_document_ids(task),
                "generator_model": GENERATOR_MODEL,
                "stage1_prompt": p1,
                "stage2_prompt": p2,
                # Filled in by verify.py.
                "target_document": None,
                "verification": None,
            }
        )
    return records


def main() -> None:
    tasks = sample_tasks()
    save_task_sample(tasks)
    documents = load_documents()

    print(f"Generating claims for {len(tasks)} tasks using {GENERATOR_MODEL}")
    records = []
    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task.id}")
        records.extend(generate_for_task(task, documents))

    CLAIMS_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAIMS_RAW_PATH.write_text(json.dumps(records, indent=2))
    print(
        f"\nWrote {len(records)} claims ({len(tasks)} tasks x "
        f"{len(STRATEGY_DESCRIPTIONS)} strategies) to {CLAIMS_RAW_PATH}"
    )
    print("Next: python -m misinfo_pipeline.verify")


if __name__ == "__main__":
    main()
