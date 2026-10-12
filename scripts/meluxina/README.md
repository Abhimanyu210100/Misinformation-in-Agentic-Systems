# MeluXina cheat sheet

Running tau2-bench with Ollama on MeluXina, project `p201155`, user `u104856`.

- **Model:** `qwen2.5:72b`, which is Qwen2.5-72B-Instruct, 4-bit Q4_K_M.
- **Seed:** 42 for every run.
- **Limit:** at most 2 domains per job, because each domain uses 2 of the node's 4 A100s.

Paths on MeluXina:

| What | Where |
|---|---|
| Repo | `/project/home/p201155/u104856/Misinformation-in-Agentic-Systems` |
| Ollama binary | `/project/home/p201155/u104856/ollama-0.33.1/` |
| Model weights | `/project/home/p201155/u104856/ollama-models/` |

## Laptop: sync code and results

Run from the repo root on the laptop.

```bash
scripts/meluxina/sync.sh push   # send code changes to MeluXina (do this after every edit)
scripts/meluxina/sync.sh pull   # bring results/ back from MeluXina
ssh meluxina                    # log in
```

## MeluXina: submit jobs

Run from the repo folder:

```bash
cd /project/home/p201155/u104856/Misinformation-in-Agentic-Systems
```

**Quick test: 2 tasks, about 5 to 10 minutes**
```bash
TAU2_TASK_IDS="0 1" sbatch --qos=test --time=00:30:00 scripts/meluxina/tau2_ollama.sbatch airline
```

**One domain**
```bash
sbatch scripts/meluxina/tau2_ollama.sbatch airline
```

**All four domains, as two jobs that run in parallel**
```bash
sbatch --time=48:00:00 scripts/meluxina/tau2_ollama.sbatch telecom airline
sbatch --time=48:00:00 scripts/meluxina/tau2_ollama.sbatch retail banking_knowledge
```

**Extra tau2 flags.** Put them after `--`. They apply to every domain in the job.
```bash
sbatch scripts/meluxina/tau2_ollama.sbatch airline -- --num-trials 2
```

Things to know before submitting:
- **Time limits.** The default is 24 hours. The `default` QoS allows up to 48 hours. The `test` QoS allows up to 30 minutes.
- **No resume.** A job that hits its time limit does not resume. Resubmitting starts the domain over.
- **Use both GPU pairs.** The whole node is billed even if you run one domain. Put 2 domains in a job when you can.
- **`TAU2_TASK_IDS`.** Only set it at the start of the `sbatch` line, as in the quick test above. If you `export` it, every later job will also run only those tasks. Clear it with `unset TAU2_TASK_IDS`.

## Check on jobs

```bash
squeue -u $USER                     # your queued/running jobs (PD = waiting, R = running; empty = none left)
sacct -u $USER -S today -X --format=JobID,JobName,State,Elapsed,ExitCode,Start,End
                                    # history: COMPLETED / FAILED / TIMEOUT / CANCELLED
sacct -j <JOBID> -X --format=JobID,State,Elapsed,ExitCode      # one job
scancel <JOBID>                     # cancel a job
```

You also get an email at abhiswaroop210100@gmail.com when a job starts, ends, or fails. Check spam the first time.

## Read logs

```bash
cat  logs/meluxina/tau2-ollama.<JOBID>.out           # summary: "<domain>: finished OK" or "FAILED"
tail -f logs/meluxina/<JOBID>/tau2_<domain>.log      # live tau2 progress (Ctrl+C to stop watching)
tail -50 logs/meluxina/<JOBID>/tau2_<domain>.log     # end of run: pass rate, errors
less logs/meluxina/<JOBID>/ollama_server0.log        # Ollama server for the 1st domain (GPUs 0,1)
less logs/meluxina/<JOBID>/ollama_server1.log        # Ollama server for the 2nd domain (GPUs 2,3)
```

What a healthy run looks like:
- The Ollama log shows `offloaded 81/81 layers to GPU`.
- The tau2 log shows lines like `Status: 3/50 complete. Avg reward: ...`.

## Results

```bash
ls results/                          # one folder per finished run: <domain>_qwen2.5_72b_<date>_<time>/
```

- **Contents.** Each folder holds `results.json` and one `<domain>_<task_id>_clean.json` per task.
- **Copying back.** Use `scripts/meluxina/sync.sh pull` on the laptop.
- **Unfinished runs.** If a run did not finish, its partial output is only in `third_party/tau2-bench/data/simulations/<run-name>/`.

## Budget and storage

```bash
myquota                              # GPU node-hours used / allowed this month (67), storage use
```

## One-time setup

This is already done. Re-run it only if the Ollama install, the Python environment, or the model is missing:

```bash
bash scripts/meluxina/setup.sh       # on a login node; safe to re-run, skips finished steps
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Permission denied (publickey)` on `ssh meluxina` | Key passphrase not in the agent: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519_mlux` |
| Job vanished from `squeue` quickly | It finished or failed: check `sacct -j <JOBID>` and the `.out` file |
| `ModuleNotFoundError` in the tau2 log | Python env incomplete: re-run `setup.sh` on a login node |
| `Ollama ... did not start` | Check `ollama_server<N>.log`; often a GPU or memory problem |
| Your edits are not picked up | You forgot `scripts/meluxina/sync.sh push` on the laptop |
