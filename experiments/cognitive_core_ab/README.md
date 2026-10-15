# Cognitive Core A/B/C/D paraphrasing experiment

This lab-only experiment compares four prompt/contract conditions on six
frozen consultation scenarios with three independent samples each.

- **A** — current production prompt + current `SemanticTurn` schema.
- **B** — reduced consultant prompt + the same `SemanticTurn` schema.
- **C** — minimal conversational prompt, plain-text response, no `SemanticTurn`.
- **D** — condition A plus the frozen conditional-paraphrasing policy.

Condition D differs from A only by the appended policy. Production files are
read-only inputs. Human Experience remains enabled and uses the same bounded
retrieval for every condition.

## Commands

```powershell
.\venv\Scripts\python.exe experiments\cognitive_core_ab\run_experiment.py --dry-run
.\venv\Scripts\python.exe experiments\cognitive_core_ab\run_experiment.py --pilot
.\venv\Scripts\python.exe experiments\cognitive_core_ab\run_experiment.py --full
```

`--pilot` performs the mandatory eight-call S1/S3 gate. `--full` refuses to
start until that exact pilot has passed, then completes the same frozen
72-call schedule without repeating valid pilot samples. Completed and attempted
call keys are tracked in the append-only hash-chained checkpoint; an attempted
key is never retried. There are no retries or evaluator calls. The hard budget
is 72 model calls and USD 1.00 for the corrected run.

Raw requests, responses, adapted structured results, token usage, prompts,
hashes and the frozen schedule are written under `results/<run_id>/`.
Behavioral scoring is intentionally human/read-only; the runner does not use
lexical overlap as a quality proxy.
