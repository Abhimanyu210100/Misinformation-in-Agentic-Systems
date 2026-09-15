"""The two injection channels.

tau2-bench exposes no override hook for either one, so both mechanisms were
derived by reading its source:

**Document channel.** ``banking_knowledge.get_environment()`` builds its
``KnowledgeBase`` internally and takes no parameter to supply one
(tau2-bench/src/tau2/domains/banking_knowledge/environment.py:36). But it
delegates to ``build_tools(variant, db, knowledge_base, ...)`` and
``build_policy(variant, knowledge_base, task)``, both of which accept a
``KnowledgeBase`` directly, and ``KnowledgeBase`` is a plain pydantic model
holding ``documents: Dict[str, Document]``. So this module reimplements that
function body with a deep-copied, poisoned knowledge base and registers the
result as a separate domain. ``TextRunConfig.domain`` then points at it.

**User channel.** ``run_single_task(config, task, ...)`` takes a live ``Task``
and never re-reads it from disk (tau2-bench/src/tau2/runner/batch.py:344), so
the claim is appended to a deep copy of the task's user-simulator instructions.

Note that the clean condition also needs a registered domain here: the stock
``banking_knowledge`` registration uses tau2's default retrieval variant
(``alltools``), and the sweep pins ``grep_only`` so every condition -- clean
included -- gives the agent the same tools.
"""

import hashlib
from copy import deepcopy

from tau2.domains.banking_knowledge.data_model import KnowledgeBase
from tau2.domains.banking_knowledge.environment import get_db, get_knowledge_base
from tau2.domains.banking_knowledge.retrieval import (
    build_policy,
    build_tools,
    resolve_variant,
)
from tau2.domains.banking_knowledge.tools import KnowledgeUserTools
from tau2.environment.environment import Environment
from tau2.knowledge.embeddings_cache import clear_cached_docs
from tau2.registry import registry

from .config import RETRIEVAL_VARIANT
from .mint import DOCUMENT_CHANNEL_TEMPLATE, USER_CHANNEL_TEMPLATE


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def pin_default_retrieval_variant(variant: str = RETRIEVAL_VARIANT) -> None:
    """Force tau2's own fallback retrieval variant for this domain.

    Grading does not go through the registered domain. ``calculate_reward``
    builds its own reference environment by calling banking_knowledge's
    ``get_environment()`` directly (tau2-bench/src/tau2/evaluator/evaluator_env.py:85),
    which falls back to ``DEFAULT_RETRIEVAL_VARIANT`` -- ``alltools`` -- and that
    variant builds a dense embedding index, requiring an OpenAI API key and
    embedding spend on every single graded run.

    Pinning the fallback to ``grep_only`` avoids that entirely. It cannot change
    any reward: banking tasks grade on database state, and poisoning only ever
    touches knowledge-base documents, never the transactional DB. It also makes
    the reference environment match the one the agent actually ran against,
    which is arguably more correct than the default.

    Patched on the ``environment`` module rather than ``retrieval`` because
    ``environment.py`` binds the constant by value at import time.
    """
    from tau2.domains.banking_knowledge import environment as bk_env

    bk_env.DEFAULT_RETRIEVAL_VARIANT = variant


def poison_document(kb: KnowledgeBase, doc_id: str, text: str) -> KnowledgeBase:
    """Return a deep copy of ``kb`` with ``text`` appended to one document.

    Appending (rather than rewriting the contradicted line) keeps the rest of
    the document byte-identical, so anything the agent reads other than the
    injected passage is exactly what the clean run would have shown it. The
    tradeoff is that the poisoned document now asserts both the true value and
    the false one -- see README §5.2, this is a known limitation of the pilot.
    """
    poisoned = kb.model_copy(deep=True)
    doc = poisoned.documents.get(doc_id)
    if doc is None:
        raise KeyError(f"Document {doc_id!r} not in knowledge base")
    doc.content = doc.content + DOCUMENT_CHANNEL_TEMPLATE.format(text=text)
    return poisoned


def register_domain(
    retrieval_variant: str = RETRIEVAL_VARIANT,
    poisoned_doc: tuple[str, str] | None = None,
) -> str:
    """Register a banking_knowledge domain and return its name.

    Args:
        retrieval_variant: tau2 retrieval variant to pin.
        poisoned_doc: ``(document_id, injected_text)`` to poison, or None for a
            clean knowledge base.

    The name is derived from the variant plus a hash of what was injected, so
    each distinct poisoning gets its own domain and re-registration is a no-op.
    tau2's ``registry.register_domain`` raises on a duplicate name, so the
    existence check is what makes this safe to call in a loop.
    """
    if poisoned_doc is None:
        name = f"bk_{retrieval_variant}_clean"
    else:
        doc_id, text = poisoned_doc
        name = f"bk_{retrieval_variant}_{_short_hash(doc_id + text)}"

    if name in registry.get_domains():
        return name

    def get_environment(**kwargs) -> Environment:
        # tau2 calls env constructors as `env_constructor(**env_kwargs)`
        # (runner/build.py:65), so accept and ignore whatever it passes.
        db = kwargs.get("db") or get_db()

        # CRITICAL: tau2 caches the indexed document list in a process-global
        # (retrieval.py::get_or_create_docs -> embeddings_cache). The cache is
        # keyed on nothing, so whichever knowledge base is built FIRST in a
        # process is silently reused by every environment built afterwards --
        # meaning a poisoned run would leak into the next clean run, or vice
        # versa, with no error. Clearing it per build costs a re-index of 698
        # documents and is the difference between a valid experiment and an
        # invalid one.
        clear_cached_docs()

        kb = get_knowledge_base()
        if poisoned_doc is not None:
            kb = poison_document(kb, poisoned_doc[0], poisoned_doc[1])

        variant = resolve_variant(retrieval_variant)
        return Environment(
            domain_name="banking_knowledge",
            policy=build_policy(variant, kb, kwargs.get("task")),
            tools=build_tools(variant, db, kb),
            user_tools=KnowledgeUserTools(db),
        )

    registry.register_domain(get_environment, name)
    return name


def inject_user_claim(task, text: str):
    """Return a deep copy of ``task`` with the claim appended to its script.

    ``user_scenario.instructions`` is typed ``StructuredUserInstructions | str``:
    banking tasks use the plain string, airline uses the structured object whose
    ``task_instructions`` field is what tau2 renders as "Task instructions:".
    Both are handled so this survives the expansion to other domains.

    The task is copied rather than mutated so one base task can produce all 8
    strategy variants without them contaminating each other.
    """
    injected = deepcopy(task)
    scenario = injected.user_scenario
    addition = USER_CHANNEL_TEMPLATE.format(text=text)

    if isinstance(scenario.instructions, str):
        scenario.instructions = scenario.instructions + addition
    else:
        scenario.instructions.task_instructions += addition
    return injected
