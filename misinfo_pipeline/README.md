# `misinfo_pipeline/`

Runs tau2-bench tasks clean, then again with misinformation injected from the
user side and from the database side, and saves the full trace of every run.

This is **Experiment 1 infrastructure**. Experiment 2 (trace diffing) and
Experiment 3 (pre-write verification) are deliberately not built yet — they are
designed against traces we haven't looked at.

---

## 1. Design decisions

Every decision below was made explicitly during the design interview, including
the ones where the cheaper option was chosen over the stricter one. They're
recorded here so the reasoning survives.

| Decision | Choice | Why |
|---|---|---|
| Scope | Experiment 1 only; generation is a separate step 0 | Misinformation is generated once and frozen, so it's identical across every model, seed and condition |
| Domain | `banking_knowledge` first, domain-agnostic where cheap | The only tau2 domain with a real document store (698 docs), retrieval tools and gold `required_documents` |
| Document delivery | Poison at rest, agent finds it via `grep` | Genuine database poisoning — misinformation arrives as a tool observation, and exposure becomes measured rather than assumed |
| Agent loop | tau2's runner, plus a trace-extraction layer | tau2 already records actions, observations and reasoning; rebuilding it would mean reimplementing the DB-state evaluator your primary metric depends on |
| False fact | LLM-derived, MINT stage-1 style | Maximum fidelity to "the exact logic from MINT" |
| Verification | Automated checks + human sign-off | MINT has no verification step; the generator now owns the truth layer, so a gate was added downstream |
| Generator | `Llama-3.3-70B-Instruct`, overlap reported | MINT's default. It's also in the agent lineup — see §3.4 |
| Strategies | Generate all 8, run all 8 as a factor | Rhetorical susceptibility is a full experimental variable |
| Conditions | Clean vs. misinformation only | No matched-true control — see §6.1 for what this costs |
| Task selection | Seeded random sample | Pilot only; the real run is all tasks across all domains |
| Trace | Native trace + injection provenance + flat index | Native is lossless; provenance is the one thing unreconstructable later |

---

## 2. Pipeline

```
  step 0            GENERATE                    generate.py
  (once)            MINT stage 1: false fact per task
                    MINT stage 2: wrap in 8 strategies
                             |
                             v
                    VERIFY                      verify.py
                    automated checks + your sign-off
                             |
                             v
                    artifacts/claims_verified.json   <-- FROZEN
                             |
  every run                  v
                    SWEEP                       sweep.py
                    for each model (outermost)
                      for each task
                        for each of 17 conditions
                          for each seed
                             |
                             +-- clean      -> unmodified task + clean KB
                             +-- user/*     -> claim appended to the persona
                             +-- document/* -> claim appended to a KB document
                             |
                             v
                    tau2 run_single_task
                             |
                             v
                    SAVE                        trace.py
                    simulation.json + injection.json + index.jsonl
```

Commands:

```bash
cd /Users/Abhimanyu/Desktop/Misinformation-in-Agentic-Systems

python -m misinfo_pipeline.generate      # step 0, writes artifacts/claims_raw.json
python -m misinfo_pipeline.verify        # review gate, writes claims_verified.json
python -m misinfo_pipeline.sweep         # the runs
python -m misinfo_pipeline.sweep --limit 4   # smoke test first
```

Use the environment tau2-bench is installed into:
`tau2-bench/.venv/bin/python -m misinfo_pipeline.generate`.

---

## 3. Step 0 — generation

### 3.1 What was ported from MINT

MINT's real generation code lives in `download_datasets.py`, and it is a
two-stage pipeline. [mint.py](mint.py) ports it.

Preserved **verbatim**:

- [`STRATEGY_DESCRIPTIONS`](mint.py#L41) — copied character-for-character from
  MINT's `get_strategy_description()`. These are literature definitions (MINT
  cites [PubMed 36789378](https://pubmed.ncbi.nlm.nih.gov/36789378/)), so they
  are quoted rather than reworded.
- The two-stage architecture: one false fact per task, then 8 restylings of it.
- The instruction scaffolding — `"Consider this task:"`, `"False fact to use as
  base:"`, `"Do not explicitly give the answer. Do not output anything else."`
- MINT's sampling parameters: [`temperature=0.7, top_p=0.9,
  repetition_penalty=1.1, max_tokens=512`](config.py#L58).

### 3.2 What necessarily changed

MINT's prompts are written for multiple-choice QA. They end with *"mislead
someone to choose the wrong answer"* and render `Answer Choices: A) ... B) ...`.
A tau-bench task has no answer choices — it has a customer goal, reference
documents, and a gold tool call.

So [`STAGE1_TEMPLATE`](mint.py#L66) and [`STAGE2_TEMPLATE`](mint.py#L80) render
the documents and the gold action in place of answer choices, and "choose the
wrong answer" becomes "take the wrong action". MINT's original wording is kept
in [`MINT_ORIGINAL_STAGE1_INSTRUCTION`](mint.py#L53) and
[`MINT_ORIGINAL_STAGE2_INSTRUCTION`](mint.py#L59) so the diff is auditable
rather than silent.

One further adaptation: banking personas run to ~5,000 characters and are
mostly role-play stage directions. Left uncapped they dominate the prompt and
the model restates the scenario instead of making a claim about the
documentation — this was observed directly during testing. Hence
[`SCENARIO_CHARS_IN_PROMPT`](config.py#L79).

### 3.3 Generating

[`generate_for_task`](generate.py#L57) calls stage 1 **once** per task, then
reuses that single false fact across all 8 strategies. That is what makes the 8
variants differ only in rhetorical framing, so strategy is cleanly isolated as a
factor in the sweep.

### 3.4 The generator/subject overlap

MINT's default generator is `Llama-3.3-70B-Instruct`, which is also the first
model in the agent lineup. No model should be silently evaluated on text it
wrote itself, so this is a known limitation to report rather than hide.

The cheap empirical check: see whether Llama-3.3-70B's uptake rate is an outlier
against the other six. If it isn't, the concern is defused in a sentence; if it
is, that's a finding.

### 3.5 Verification

MINT generates and ships — it has no verification step anywhere. Since stage 1
is LLM-derived, the generator now owns the truth layer, so [verify.py](verify.py)
adds a gate *downstream* of generation, leaving MINT's logic untouched.

Two mechanical checks:

- [`attribute_document`](verify.py#L48) — which gold document does the claim
  concern? Needed as a relevance signal, and because the document channel has to
  know which document to poison.
- [`check_decision_relevance`](verify.py#L70) — does the claim touch anything
  the gold action depends on? A claim about a product the correct action never
  touches is a null condition that silently dilutes the effect.

**Be clear about what these are: lexical heuristics.** They can suggest a claim
is probably irrelevant. They cannot establish that it is *false*. That is what
your sign-off is for. Review is once per task (on the false fact), not once per
strategy — 25 decisions, not 200.

---

## 4. Conditions

17 per task, built by [`build_conditions`](sweep.py#L72):

| Condition | Task | Knowledge base |
|---|---|---|
| `clean` | unmodified | clean |
| `user/<strategy>` × 8 | claim appended to the persona | clean |
| `document/<strategy>` × 8 | unmodified | one document poisoned |

Pilot scale: 25 tasks × 17 conditions × 1 model × 1 seed ≈ **425 runs**.

---

## 5. Injection mechanisms

tau2 exposes no override hook for either channel. Both were derived by reading
its source.

### 5.1 User channel

`run_single_task(config, task, ...)`
([batch.py:344](../tau2-bench/src/tau2/runner/batch.py#L344)) takes a live
`Task` and never re-reads it from disk, so
[`inject_user_claim`](injection.py#L122) appends the claim to a deep copy of the
task's user-simulator instructions.

`UserInstructions` is typed `StructuredUserInstructions | str`
([tasks.py:49](../tau2-bench/src/tau2/data_model/tasks.py#L49)) — banking uses
the plain string, airline uses the structured object whose `task_instructions`
field tau2 renders as `"Task instructions:"`
([tasks.py:44](../tau2-bench/src/tau2/data_model/tasks.py#L44)). Both forms are
handled so this survives the move to other domains.

### 5.2 Document channel

`banking_knowledge.get_environment()` builds its `KnowledgeBase` internally and
accepts no parameter to supply one
([environment.py:36](../tau2-bench/src/tau2/domains/banking_knowledge/environment.py#L36)).
But it delegates to `build_tools(variant, db, knowledge_base, ...)` and
`build_policy(variant, knowledge_base, task)`, both of which take a
`KnowledgeBase` directly. So [`register_domain`](injection.py#L66) reimplements
that function body with a poisoned knowledge base and registers the result as a
separate tau2 domain; `TextRunConfig.domain` then points at it.

[`poison_document`](injection.py#L49) appends the claim to one document rather
than rewriting the contradicted line, so everything the agent reads other than
the injected passage is byte-identical to the clean run.

> **Known limitation.** Because the claim is appended, the poisoned document now
> asserts both the true value and the false one. A sufficiently careful agent
> could notice the contradiction — which is a different stimulus from a
> cleanly-false document. The alternative (locate and rewrite the contradicted
> line) is more faithful but brittle, since it depends on correctly identifying
> the line. Worth revisiting once the pilot shows how often agents notice.

Clean runs also register a domain here, because the stock `banking_knowledge`
registration uses tau2's default retrieval variant (`alltools`) and the sweep
pins [`grep_only`](config.py#L34) so every condition gives the agent the same
tools.

### 5.3 A cache bug that would have invalidated the experiment

tau2 caches its indexed document list in a **process-global**
(`retrieval.py::get_or_create_docs` → `embeddings_cache`). The cache is keyed on
nothing. So whichever knowledge base is built first in a process is silently
reused by every environment built afterwards.

Left alone, this means the poisoned KB leaks into subsequent clean runs — or the
clean KB masks the poison — with no error and no warning. It was caught during
testing: a clean environment was returning poisoned content.

[`clear_cached_docs()`](injection.py#L104) is called on every environment build.
It costs a re-index of 698 documents per build, and it is the difference between
a valid experiment and an invalid one. **Do not remove it as an optimization.**

---

## 6. What a skeptic will attack

### 6.1 There is no matched-true control

The sweep runs clean vs. misinformation only. So when a misinformed run fails,
the data cannot distinguish:

- the claim was **false**, and the agent believed it (the hypothesis), from
- an extra claim was **present** at all — more text, a more insistent customer,
  an extra retrieval hit — and it disrupted the agent regardless of truth value.

A matched-true condition (a true claim in the same slot) is the only thing that
separates these. It was considered and dropped in favor of a cheaper sweep. If a
reviewer raises it, the answer is either to run matched-true on a subset, or to
report the ambiguity honestly.

### 6.2 Exposure is not guaranteed in the document channel

The agent only sees the poisoned document if one of its own `grep` calls returns
it. Runs where it never did say nothing about credulity and must be analyzed
separately — [`detect_exposure`](trace.py#L41) records this per run, and the
user channel (where exposure is certain) is therefore not directly comparable to
the document channel without conditioning on it.

### 6.3 An 8B user simulator cannot run this domain at all

**Verified during testing, and it blocks local end-to-end validation.** With
`llama3.1:8b` as the user simulator, every `banking_knowledge` task terminates
after 2 messages: the simulator emits `###STOP###` instead of playing the
customer, so the agent never acts and every reward is 0.0.

This is not a bug in this pipeline. The same 8B simulator, same code path, holds
normal 10-22 message conversations on `airline`. The difference is the persona:
banking scenarios run to ~5,400 characters of role-play stage directions, and an
8B model cannot follow them while also managing tau2's control tokens.

Consequences:

- The 8B defaults are good for validating **plumbing**, not for producing
  results. A green smoke test here means the wiring works, not that the
  experiment ran.
- The pilot needs a stronger user-simulator model on the GPU. This is the same
  lesson the earlier airline pilot recorded, and it bites harder here.
- For a fast local sanity check of the runner itself, `airline` is the domain
  that actually completes conversations at 8B.

### 6.4 Generation quality at 8B is poor

Observed directly in testing: an 8B generator ignores MINT's "Do not output
anything else" and emits preamble, and its false facts ramble. The 70B
production generator should do considerably better, and the verify gate exists
to reject what doesn't. Don't judge the claim set by an 8B smoke test.

---

## 7. Models

Everything defaults to **8B** so the pipeline can be smoke-tested locally on
Ollama before GPU time is committed: [`AGENT_MODEL`](config.py#L43),
[`USER_MODEL`](config.py#L48), [`GENERATOR_MODEL`](config.py#L53).

The user simulator is deliberately not shrunk below 8B. A weak simulator emits
tau2's control tokens instead of playing the persona, which turns failures into
trivial 2-message "successes" and corrupts the eval itself rather than just the
agent side.

Production lineup (swap in via `--models`, or edit `config.py`):

| Model | Class |
|---|---|
| Llama-3.3-70B-Instruct | large, non-reasoning |
| Qwen2.5-72B-Instruct | large, non-reasoning |
| Qwen2.5-32B-Instruct | mid, non-reasoning |
| Mistral-Small3.2-24B | small, non-reasoning |
| Qwen3-32B | reasoning, toggleable |
| QwQ-32B | reasoning, always on |
| Magistral-24B | small reasoning |

Models are iterated **outermost** in [`main`](sweep.py#L118) so each is loaded
and served once rather than swapped per task.

---

## 8. Output

```
runs/<model>/<task_id>/<condition>/<seed>/
    simulation.json    tau2 SimulationRun, verbatim
    injection.json     what was injected, where, from which claim
runs/index.jsonl       one flat row per run
runs/manifest.jsonl    completed cells, for resume
```

`simulation.json` is lossless — it already holds every tool call, tool result,
reward, seed, and the full raw LLM response per assistant turn (`raw_data`, set
at [llm_utils.py:452](../tau2-bench/src/tau2/utils/llm_utils.py#L452)), which is
where reasoning content from vLLM-served reasoning models lands. That matters
for the Qwen3 reasoning-on/off comparison in Experiment 2.

`injection.json` is the one artifact that cannot be reconstructed afterwards:
once a run is over, nothing in the trace records what the document originally
said.

### Resume

tau2 checkpoints within a single run config, but this sweep spans conditions and
models, so [sweep.py](sweep.py) keeps its own manifest: one line per completed
cell, checked before dispatch ([`load_manifest`](sweep.py#L56)). A failed cell is
logged and skipped **without** being recorded, so a rerun retries exactly the
failures. Re-running the sweep is always safe.

---

## 9. Setup

`banking_knowledge` needs tau2's optional `knowledge` extra. `rank-bm25` was
missing and has been installed into `tau2-bench/.venv`. For a clean environment:

```bash
cd tau2-bench && uv sync --extra knowledge
```

Ollama must be running (`ollama serve`) with `llama3.1:8b` pulled for the
defaults to work.
