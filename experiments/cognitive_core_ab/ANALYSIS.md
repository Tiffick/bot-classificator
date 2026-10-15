# Cognitive Core A/B/C/D analysis

## Live run status

- Run: `cc-abcd-20261011T015406Z-ea892bd5`
- Model requested: `gpt-5-mini`
- Model returned: `gpt-5-mini-2025-08-07`
- API calls: **72** (`18` per condition), with no retries or evaluator calls.
- Checkpoint: **144** events (`72` requests + `72` responses), valid hash chain,
  no `API_FAILURE` events, and a valid `RUN_COMPLETED` marker.
- Token usage: 393,834 input, 330,880 cached input, 56,456 output,
  55,296 reasoning, 450,290 total.
- Actual cost: **USD 0.1369225**.

## Measurement validity

**INVALID / INCONCLUSIVE.** The frozen maximum completion budget was 800
tokens. In 66 of 72 calls the model used the entire budget for reasoning,
returned `finish_reason=length`, and emitted no visible answer. The remaining
six calls stopped normally and produced plain text only in condition C for
S2/S3. Consequently, the four conditions cannot be compared on the frozen
behavioral rubric.

| Condition | Calls | Non-empty outputs | Contract-valid outputs | `length` |
| --- | ---: | ---: | ---: | ---: |
| A | 18 | 0 | 0 | 18 |
| B | 18 | 0 | 0 | 18 |
| C | 18 | 6 | 6 plain-text | 12 |
| D | 18 | 0 | 0 | 18 |

The six observable C replies are `C.S2.1`, `C.S3.1`, `C.S2.2`, `C.S3.2`,
`C.S2.3`, and `C.S3.3`. Each acknowledges or repeats the correction and then
asks multiple questions and/or offers solution design. They are preserved as
raw observations, but they do not supply a matched A/B/C/D comparison and are
not used to infer the effect of conditional paraphrasing.

No score is assigned for grounding, added understanding, progress,
naturalness, correction handling, unwarranted inference, or paraphrasing
because missing model-visible answers are a measurement failure rather than a
behavioral outcome. In particular, the result does **not** support or falsify
condition D, prompt complexity effects, or SemanticTurn-induced paraphrasing.

The run consumed its complete authorized 72-call budget. It must not be
resumed or repeated without a new explicit authorization and a separately
reviewed output-budget correction.

## Corrected experiment preparation

The corrected v2 run is separate from the invalid directory and uses:

- exact snapshot `gpt-5-mini-2025-08-07`;
- `reasoning_effort=minimal` for all four conditions;
- `max_completion_tokens=4000` for all four conditions;
- deterministic frozen fixture IDs/timestamps;
- an eight-call A/B/C/D pilot on S1 and S3;
- pilot calls counted toward the frozen 72-call total only after all pilot
  criteria pass;
- no retries, fallback or evaluator calls;
- a USD 1.00 hard cost gate, checked before every call.

The conservative full-run estimate is 1,088,589 input tokens plus at most
288,000 output tokens, or **USD 0.848147** at the recorded pricing.

The mandatory pilot then attempted five calls and stopped before the full run.
Four S1 responses were recorded without output exhaustion; A/B/C validated,
while D produced an invalid local-vs-existing item reference. The A.S3.1 API
response was received but the lab harness failed before durable recording when
`deepcopy()` encountered the immutable ACS `FrozenDict`. No retry occurred.
The pilot therefore failed and the remaining schedule was not executed.
Cumulative history and accounting are maintained in `FINAL_REPORT.md`.

## Offline harness repair after failed pilot

The lab-only `deepcopy(previous_acs)` path was replaced with deterministic
Pydantic reconstruction: `model_dump(mode="python")` followed by
`type(model).model_validate(...)`. Nine offline regression tests pass. They
cover the original immutable-`FrozenDict` failure, all S1–S6 fixture
round-trips/adaptations, deterministic payload construction and corrected
request settings. No API call or new pilot was made, and both earlier
checkpoints retain their original SHA-256 hashes.

## Fresh isolated v3 pilot

Run `cc-abcd-v3-20261011T023640Z-d13f5f46` completed exactly the frozen
eight-call A/B/C/D pilot on S1 and S3 and then stopped. Every raw API response,
usage record, latency and cost was durably appended and fsynced before a
separate semantic-validation event. The 24-event checkpoint hash chain is
valid. There were no retries, API failures, evaluator calls or full-run calls.

All eight responses were visible and ended with `finish_reason=stop`; none
used reasoning tokens or exhausted the 4000-token ceiling. The pilot still
failed its strict validity gate. A/B/D on S1 referred to newly declared local
items through `existing_id` instead of the local `handle`. A.S3.1 used the
persistent ACS ID in `reply_segments.realizes` instead of a declared content
handle. B.S3.1 and D.S3.1 adapted successfully. Therefore structured validity
was A 0/2, B 1/2, D 1/2 (2/6 combined); both C outputs were valid plain text.

The eight calls used 43,990 input tokens, 36,352 cached input tokens, 4,430
output tokens, 0 reasoning tokens and 48,420 total tokens, costing USD
0.01167830. The pilot verdict is **FAIL**, and the full 72-call experiment is
not ready to run. The next action is a read-only root-cause audit of the
model's local-vs-existing reference failures, not another live pilot and not a
validation relaxation.

For each saved response, assess:

1. grounding in user evidence;
2. added understanding;
3. consultation progress;
4. naturalness;
5. correction handling;
6. unwarranted inference;
7. SemanticTurn validity for A/B/D;
8. paraphrasing functionality: `NONE`, `FUNCTIONAL`, `REDUNDANT`, or both
   `FUNCTIONAL` and `REDUNDANT` when distinct fragments warrant both labels.

Absence of repetition is not automatically good, and functional repetition is
not penalized. Evaluator votes must not be confused with independent model
samples; this experiment has no evaluator API calls.
