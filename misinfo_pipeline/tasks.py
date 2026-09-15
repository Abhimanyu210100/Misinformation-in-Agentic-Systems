"""Task sampling and rendering.

Selection is a seeded random sample. All 97 banking_knowledge tasks already
satisfy the brief's stated criteria -- every one has gold `required_documents`
and at least one evaluation action -- so those criteria do not discriminate and
are not re-applied as filters here.

The pilot samples 20-30 tasks; the production run sets ``n=None`` to take
everything, across every domain.
"""

import json
import random

from tau2.data_model.tasks import Task
from tau2.run import get_tasks

from .config import (
    BANKING_DOCUMENTS_DIR,
    DOC_CHARS_IN_PROMPT,
    DOMAIN,
    MAX_DOCS_IN_PROMPT,
    N_TASKS,
    SCENARIO_CHARS_IN_PROMPT,
    TASK_SAMPLE_PATH,
    TASK_SAMPLE_SEED,
)


def sample_tasks(
    domain: str = DOMAIN,
    n: int | None = N_TASKS,
    seed: int = TASK_SAMPLE_SEED,
) -> list[Task]:
    """Return a seeded random sample of tasks, or all of them when ``n`` is None."""
    tasks = get_tasks(domain)
    if n is None or n >= len(tasks):
        return tasks
    rng = random.Random(seed)
    return rng.sample(tasks, n)


def save_task_sample(tasks: list[Task], path=TASK_SAMPLE_PATH) -> None:
    """Freeze which tasks were sampled, so a rerun cannot silently re-draw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "domain": DOMAIN,
                "seed": TASK_SAMPLE_SEED,
                "task_ids": [t.id for t in tasks],
            },
            indent=2,
        )
    )


def load_documents() -> dict[str, dict]:
    """Load every knowledge-base document, keyed by document id."""
    docs = {}
    for path in sorted(BANKING_DOCUMENTS_DIR.glob("*.json")):
        doc = json.loads(path.read_text())
        docs[doc["id"]] = doc
    return docs


def required_document_ids(task: Task) -> list[str]:
    """Gold document ids for a task, or [] if the domain has none."""
    return list(getattr(task, "required_documents", None) or [])


def render_scenario(task: Task) -> str:
    """Render the customer scenario as the generator prompt should see it.

    ``user_scenario.instructions`` is typed ``StructuredUserInstructions | str``
    in tau2: banking tasks use the plain string form, airline uses the
    structured object. Both are handled so this works when the sweep expands to
    other domains.
    """
    instructions = task.user_scenario.instructions
    text = instructions if isinstance(instructions, str) else str(instructions)
    return text[:SCENARIO_CHARS_IN_PROMPT]


def render_documents(task: Task, documents: dict[str, dict]) -> str:
    """Render a task's gold documents for the generator prompt, truncated.

    Tasks can require up to 30 documents and some run long, which would overrun
    the generator's context; both the count and each document are capped.
    """
    rendered = []
    for doc_id in required_document_ids(task)[:MAX_DOCS_IN_PROMPT]:
        doc = documents.get(doc_id)
        if doc is None:
            continue
        content = doc["content"][:DOC_CHARS_IN_PROMPT]
        rendered.append(f"--- {doc_id} | {doc['title']} ---\n{content}")
    return "\n\n".join(rendered) if rendered else "(no reference documents)"


def render_gold_actions(task: Task) -> str:
    """Render the task's evaluation actions as the correct tool calls."""
    criteria = task.evaluation_criteria
    actions = getattr(criteria, "actions", None) or [] if criteria else []
    lines = []
    for action in actions:
        name = getattr(action, "name", None)
        args = getattr(action, "arguments", None) or {}
        lines.append(f"- {name}({json.dumps(args)})")
    return "\n".join(lines) if lines else "(no recorded gold actions)"
