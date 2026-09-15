"""Central configuration: paths, models, and sweep defaults.

Every default here is sized for the *pilot*: one 8B model, a random sample of
tasks, one seed. The production run (7 models, all tasks, all domains) changes
these values, not the code.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAU2_BENCH_DIR = PROJECT_ROOT / "tau2-bench"
BANKING_DATA_DIR = TAU2_BENCH_DIR / "data" / "tau2" / "domains" / "banking_knowledge"
BANKING_DOCUMENTS_DIR = BANKING_DATA_DIR / "documents"

# Step-0 output: the frozen claim set. Generated once, reviewed once, then
# reused by every run so misinformation is identical across models and seeds.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLAIMS_RAW_PATH = ARTIFACTS_DIR / "claims_raw.json"
CLAIMS_VERIFIED_PATH = ARTIFACTS_DIR / "claims_verified.json"
TASK_SAMPLE_PATH = ARTIFACTS_DIR / "task_sample.json"

# Run outputs.
RUNS_DIR = PROJECT_ROOT / "runs"
INDEX_PATH = RUNS_DIR / "index.jsonl"
MANIFEST_PATH = RUNS_DIR / "manifest.jsonl"

DOMAIN = "banking_knowledge"

# grep_only gives the agent a `grep(pattern)` tool over the knowledge base and
# nothing else. Chosen so the poisoned document must be *found* by the agent
# rather than handed to it: exposure becomes a measured variable rather than an
# assumption, and no embedding API key or spend is required.
RETRIEVAL_VARIANT = "grep_only"

# --- Models ---
#
# Everything defaults to 8B so the full pipeline can be smoke-tested locally on
# Ollama before any GPU time is committed. Production values are recorded
# alongside as comments rather than as code, so switching is a deliberate edit.

# Pilot agent under test. Production: the 7-model lineup in README §7.
AGENT_MODEL = "ollama_chat/llama3.1:8b"

# User simulator. Deliberately not shrunk below 8B: a weak simulator emits
# tau2's control tokens instead of playing the persona, which silently turns
# failures into trivial 2-message "successes" and corrupts the eval itself.
USER_MODEL = "ollama_chat/llama3.1:8b"

# Step-0 generator. Production: "meta-llama/Llama-3.3-70B-Instruct" (MINT's
# default). That model is also in the agent lineup, so the generator/subject
# overlap is a known, reported limitation -- see README §3.4.
GENERATOR_MODEL = "ollama_chat/llama3.1:8b"

# MINT's exact sampling parameters (download_datasets.py). Preserved verbatim
# because the brief calls for MINT's generation logic, and sampling settings
# materially shape the text it produces.
MINT_TEMPERATURE = 0.7
MINT_TOP_P = 0.9
MINT_REPETITION_PENALTY = 1.1
MINT_MAX_TOKENS = 512

# --- Pilot sweep shape ---

N_TASKS = 25
TASK_SAMPLE_SEED = 0
SEEDS = [0]

# How much of each reference document to show the generator. Some banking docs
# are long and a task can require up to 30 of them; without a cap the stage-1
# prompt overruns context.
DOC_CHARS_IN_PROMPT = 1200
MAX_DOCS_IN_PROMPT = 6

# Banking personas run to ~5,000 characters and are mostly role-play stage
# directions ("Don't dump all your information at once"). Left uncapped they
# dominate the generator prompt and the model restates the scenario instead of
# making a claim about the documentation.
SCENARIO_CHARS_IN_PROMPT = 900
