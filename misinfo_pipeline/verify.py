"""Step 0 gate: automated checks, then human sign-off.

MINT itself has no verification step -- it generates and ships. This gate is
added downstream, so MINT's generation logic stays untouched while the claims
that reach the sweep are ones you have actually looked at.

Two things are checked mechanically:

- **Attribution**: which gold document does the false fact concern? Needed both
  as a relevance signal and because the document channel has to know which
  document to poison.
- **Decision relevance**: does the claim touch an entity or value the gold
  action depends on? A claim about something the correct action doesn't hinge
  on is a null condition -- it would quietly dilute the effect you're measuring.

Be clear about what these checks are: lexical heuristics. They can tell you a
claim is *probably irrelevant*; they cannot tell you it is *false*. Establishing
that the claim actually contradicts the documentation is what your sign-off is
for. Review happens once per task (on the stage-1 false fact), not once per
strategy, since the 8 variants are restylings of the same underlying claim.

    python -m misinfo_pipeline.verify              # interactive review
    python -m misinfo_pipeline.verify --auto-approve   # smoke tests only
"""

import argparse
import json
import re

from .config import CLAIMS_RAW_PATH, CLAIMS_VERIFIED_PATH
from .tasks import load_documents, sample_tasks

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "that", "this", "these", "those", "you", "your", "it", "its", "as", "at",
    "by", "from", "can", "will", "would", "there", "their", "they", "not",
    "no", "any", "all", "may", "must", "should", "than", "then", "when",
}


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def attribute_document(false_fact: str, doc_ids: list[str], documents: dict) -> tuple[str | None, float]:
    """Pick the gold document the false fact most plausibly concerns.

    Scored by overlap of distinctive content words, normalized by the claim's
    own vocabulary so long documents don't win automatically.
    """
    claim_words = _content_words(false_fact)
    if not claim_words:
        return None, 0.0

    best_id, best_score = None, 0.0
    for doc_id in doc_ids:
        doc = documents.get(doc_id)
        if doc is None:
            continue
        doc_words = _content_words(doc["title"] + " " + doc["content"])
        score = len(claim_words & doc_words) / len(claim_words)
        if score > best_score:
            best_id, best_score = doc_id, score
    return best_id, round(best_score, 3)


def check_decision_relevance(false_fact: str, gold_actions_text: str) -> bool:
    """Does the claim mention any distinctive term from the gold action arguments?

    Catches the common null case where the generator invents a true-sounding
    claim about a product the correct action never touches.
    """
    action_words = _content_words(gold_actions_text)
    claim_words = _content_words(false_fact)
    return bool(action_words & claim_words)


def check_has_number(false_fact: str) -> bool:
    """Numeric claims are the most checkable kind, and the easiest to poison
    unambiguously (a fee, a rate, a threshold). Absence isn't disqualifying,
    only worth flagging during review."""
    return bool(_NUM_RE.search(false_fact))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip interactive review and approve everything. Smoke tests only.",
    )
    args = parser.parse_args()

    records = json.loads(CLAIMS_RAW_PATH.read_text())
    documents = load_documents()
    tasks_by_id = {t.id: t for t in sample_tasks()}

    from .tasks import render_gold_actions

    # Group by task: one false fact per task, 8 strategy restylings of it.
    by_task: dict[str, list[dict]] = {}
    for rec in records:
        by_task.setdefault(rec["task_id"], []).append(rec)

    approved, rejected = 0, 0
    for task_id, task_records in by_task.items():
        false_fact = task_records[0]["false_fact"]
        task = tasks_by_id.get(task_id)
        gold_actions_text = render_gold_actions(task) if task else ""

        target_doc, score = attribute_document(
            false_fact, task_records[0]["required_documents"], documents
        )
        relevant = check_decision_relevance(false_fact, gold_actions_text)
        has_number = check_has_number(false_fact)

        checks = {
            "target_document": target_doc,
            "attribution_score": score,
            "decision_relevant": relevant,
            "has_number": has_number,
        }

        if args.auto_approve:
            decision = target_doc is not None
        else:
            print("\n" + "=" * 72)
            print(f"TASK {task_id}")
            print(f"\nFALSE FACT:\n  {false_fact}")
            print(f"\nGOLD ACTIONS:\n{gold_actions_text}")
            print(f"\nCHECKS:")
            print(f"  target document   : {target_doc}  (overlap {score})")
            print(f"  decision relevant : {'YES' if relevant else 'NO  <-- likely null condition'}")
            print(f"  contains a number : {'yes' if has_number else 'no'}")
            if target_doc:
                doc = documents[target_doc]
                print(f"\nTARGET DOC ({doc['title']}):\n  {doc['content'][:600]}")
            answer = input("\nApprove this claim? [y/N/q] ").strip().lower()
            if answer == "q":
                print("Stopped. Nothing written.")
                return
            decision = answer == "y"

        for rec in task_records:
            rec["target_document"] = target_doc
            rec["verification"] = {**checks, "approved": decision}

        approved += decision
        rejected += not decision

    verified = [r for r in records if r["verification"]["approved"]]
    CLAIMS_VERIFIED_PATH.write_text(json.dumps(verified, indent=2))

    print(f"\nApproved {approved} tasks, rejected {rejected}.")
    print(f"Wrote {len(verified)} approved claims to {CLAIMS_VERIFIED_PATH}")
    print("Next: python -m misinfo_pipeline.sweep")


if __name__ == "__main__":
    main()
