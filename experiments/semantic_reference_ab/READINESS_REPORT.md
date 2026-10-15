# SemanticTurn Reference A/B — Readiness Report

## Status

READY FOR SEPARATE LIVE AUTHORIZATION. No API calls were made during preparation.

## Experimental question

Does adding a compact reference map plus two valid examples and one invalid
example reduce complete SemanticTurn validation failures, compared with the
unchanged assembled production Cognitive Core instructions?

This experiment tests reference-contract operationalization. It does not test
a new architecture, schema, model, HumanModel, ACS, adapter, or consultation
policy.

## Conditions

- **A — Control:** the current assembled production system instructions,
  including the current Human Experience system instruction, byte-identical to
  production assembly.
- **B — Treatment:** A followed only by `treatment_appendix.txt`.

The treatment appendix contains:

1. a compact `handle` / `existing_id` / `realizes` map;
2. one complete valid example for a current-turn item handle and ActionContent
   `realizes` handle;
3. one complete valid example for an actual item already present in that
   example's HumanModel;
4. one invalid local-handle-as-`existing_id` example and the expected reason
   for rejection.

## Frozen design

- Model: `gpt-5-mini-2025-08-07`.
- Reasoning effort: `minimal`.
- Maximum completion tokens: `4000`.
- Retries: `0`.
- Scenarios: exact frozen S1 and S3 fixtures from the prior A/B/C/D pilot.
- Conditions: A and B.
- Independent repetitions: 3 per condition/scenario cell.
- Fixed, counterbalanced schedule: 12 calls total.
- Same scenario-specific strict JSON Schema for A and B.
- Same frozen input for A and B within each scenario.
- No evaluator calls and no adaptive follow-up.

The authoritative schedule and hashes are in `manifest.json`. The runner fails
closed if the current dependencies, prompts, fixtures, schemas, schedule, or
examples differ from the manifest.

## Metrics

The primary unit is one independent production response. A response passes only
if it:

1. parses as JSON;
2. passes the current scenario-specific JSON Schema;
3. passes `SemanticTurn.model_validate`;
4. passes `adapt_semantic_turn`;
5. produces a valid `CognitiveTurnResult`.

Report valid samples out of 6 per condition and also by scenario. Do not treat
multiple validation layers as independent observations.

Consultation quality is secondary and must be reviewed after the run from the
saved replies, preferably blind to condition labels. No quality evaluator API
is included in the budget.

## Offline validation

- Correct example `correct_local_handle_and_realizes`:
  JSON Schema PASS; SemanticTurn PASS; adapter PASS; internal contract PASS.
- Correct example `correct_existing_id_and_realizes`:
  JSON Schema PASS; SemanticTurn PASS; adapter PASS; internal contract PASS.
- Incorrect example `incorrect_local_handle_as_existing_id`:
  JSON Schema PASS; expected SemanticTurn failure confirmed:
  `Persistent ID must start with 'mi_'`.
- S1 and S3 input hashes match the prior pilot manifest.
- S1 and S3 schema hashes match the prior pilot manifest.
- Condition A hash matches current production assembly.
- Condition B is mechanically verified as A plus only the frozen appendix.
- All 12 request payloads were prepared offline without an API client call.
- `py_compile` PASS.
- Runner dry-run PASS.

The lab JSON Schema checker implements every keyword present in the current
generated schema and fails closed if a new unsupported keyword appears. It does
not replace production Pydantic or adapter validation.

## Execution boundary

The default command is offline-only:

```powershell
.\venv\Scripts\python.exe experiments\semantic_reference_ab\run_experiment.py --dry-run
```

A future live run additionally requires both `--execute` and the explicit
authorization token embedded in the lab runner. Without both, it cannot call
the API. Live execution was not authorized or performed in this preparation.

## Files

- `manifest.json` — frozen design, metrics, schedule, dependency hashes.
- `treatment_appendix.txt` — the only A-to-B instruction difference.
- `examples.json` — full validation fixtures for the examples.
- `run_experiment.py` — offline validator and separately gated live runner.
- `READINESS_REPORT.md` — this report.
