# SemanticTurn Reference A/B Pilot — Final Report

## Run

- Run ID: `semantic-ref-ab-20261012T042853Z-9c26c00d`
- Model requested and returned: `gpt-5-mini-2025-08-07`
- Reasoning effort: `minimal`
- Maximum completion tokens: `4000`
- Retries: `0`
- Evaluator calls: `0`
- API calls: `12`
- Finish reason: `stop` for all 12 responses
- Checkpoint hash chain: PASS
- Checkpoint SHA-256:
  `FFD9BA4D7E02EA14B57069142CE91702C92A7050D71FA749FB1DEF48C4D0DDE4`

The preflight passed immediately before execution: package hashes, frozen S1/S3
input hashes, scenario-specific schema hashes, control prompt identity, and the
assertion that treatment equals control plus only the frozen appendix all
matched `manifest.json`.

Raw API responses were fsync-persisted to the run checkpoint before semantic
validation. No response was repaired, retried, or replaced.

## Conditions

- **A — Control:** current assembled production Cognitive Core system
  instructions, unchanged.
- **B — Treatment:** A plus only `treatment_appendix.txt`, containing the
  compact reference map, two valid examples, and one invalid example.

Both conditions used the same frozen input and same strict JSON Schema within
each scenario. Call order was the frozen counterbalanced order from the
manifest.

## Per-sample validation

| Sample | Generation | JSON Schema | SemanticTurn | Adapter | CognitiveTurnResult | Complete |
| --- | --- | --- | --- | --- | --- | --- |
| A.S1.1 | PASS | PASS | PASS | PASS | PASS | PASS |
| B.S1.1 | PASS | PASS | PASS | PASS | PASS | PASS |
| B.S3.1 | PASS | PASS | PASS | PASS | PASS | PASS |
| A.S3.1 | PASS | PASS | PASS | FAIL | not reached | FAIL |
| B.S1.2 | PASS | PASS | PASS | PASS | PASS | PASS |
| A.S1.2 | PASS | PASS | PASS | PASS | PASS | PASS |
| A.S3.2 | PASS | PASS | PASS | FAIL at internal construction | FAIL | FAIL |
| B.S3.2 | PASS | PASS | PASS | PASS | PASS | PASS |
| A.S1.3 | PASS | PASS | PASS | PASS | PASS | PASS |
| B.S1.3 | PASS | PASS | PASS | PASS | PASS | PASS |
| B.S3.3 | PASS | PASS | PASS | PASS | PASS | PASS |
| A.S3.3 | PASS | PASS | PASS | PASS | PASS | PASS |

Exact failures:

1. **A.S3.1** — `SemanticWireError: Wrong-type semantic handle:
   itm_followup_eating_response; expected content`. The response declared this
   name as an item handle, then used it where a current ActionContent handle was
   required by target ownership/subject and `reply_segments.realizes`.
2. **A.S3.2** — adapter reference mapping completed, but construction of the
   internal result failed: `CONSENT target subject must be SYSTEM_PROPOSAL or
   SYSTEM_TRANSITION`. The selected content was `SYSTEM_REFLECTION`.

The second failure is reported as an adapter failure because
`adapt_semantic_turn` cannot return a result; its recorded underlying boundary
is CognitiveTurnResult validation.

## Primary metric

| Slice | Complete valid | Rate |
| --- | ---: | ---: |
| A overall | 4/6 | 66.7% |
| B overall | 6/6 | 100% |
| A.S1 | 3/3 | 100% |
| B.S1 | 3/3 | 100% |
| A.S3 | 1/3 | 33.3% |
| B.S3 | 3/3 | 100% |
| S1 overall | 6/6 | 100% |
| S3 overall | 4/6 | 66.7% |

Layer totals:

- API generation: 12/12.
- JSON parsing: 12/12.
- Current JSON Schema: 12/12.
- `SemanticTurn.model_validate`: 12/12.
- Semantic reference mapping: 11/12; A.S3.1 failed.
- Complete `adapt_semantic_turn` return: 10/12.
- Valid CognitiveTurnResult: 10/12; A.S3.2 failed and A.S3.1 never reached it.

## Token accounting

| Condition | Input | Cached input | Output | Reasoning | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 46,353 | 37,376 | 3,848 | 0 | 50,201 |
| B | 51,153 | 44,288 | 2,730 | 0 | 53,883 |
| Total | 97,506 | 81,664 | 6,578 | 0 | 104,084 |

Treatment increased the prompt/input size as expected. Its lower output-token
count in this run is descriptive only and is not a quality conclusion.

## Human quality review preparation

`BLINDED_RESPONSES.json` contains all 12 exact user-facing replies in a
deterministically shuffled order without condition labels or technical validity.
`BLINDING_KEY.json` contains the separate mapping back to sample keys. No LLM
evaluator was used.

The saved set supports subsequent human review of groundedness, useful
progress, repetition, question burden, and Product Boundary. This report does
not declare B a consultation-quality winner based only on technical validity.

## CONFIRMED

1. The frozen run executed exactly 12 API calls with the requested model and
   parameters, without retries, fallback, evaluator, or adaptive calls.
2. All raw responses were non-empty, had `finish_reason=stop`, and are preserved
   in the hash-chained checkpoint.
3. In this sample, B achieved 6/6 complete contract validity and A achieved 4/6.
4. Both observed technical failures occurred in A.S3; neither occurred in the
   six treatment responses.
5. JSON Schema and initial SemanticTurn structural validation passed all 12;
   the two failures required Python cross-reference/internal-contract checks.
6. Production code, production prompt, SemanticTurn, schema, and validators
   were not changed during execution or analysis.

## OBSERVED

1. The treatment responses used the demonstrated separation among local item
   handles, current ActionContent handles, and `realizes` consistently in all
   six samples.
2. Control S1 happened to be valid 3/3; the technical difference appeared only
   in the more reference-intensive S3 context.
3. Replies in both conditions still vary in length, repetition, hypothesis
   strength, and whether they ask another question. Those properties require
   the prepared human review and are not settled by contract validity.

## UNKNOWN AND LIMITATIONS

1. Three repetitions per cell are a small descriptive sample. They do not
   establish statistical significance or expected production failure rates.
2. Only S1 and S3 were tested. Generalization to other ACS interactions,
   relations, proposal materialization, multiple previous targets, or other
   models is unknown.
3. The experiment changes prompt length as well as example content; it cannot
   isolate which specific table/example element caused any difference.
4. B's 6/6 result does not prove superior consultation quality. Human scoring
   has not yet been performed.
5. A.S3.2 is an interaction/content compatibility failure rather than the exact
   local-vs-existing reference error targeted by the treatment hypothesis.
6. No conclusion is made about hidden reasoning or why the model emitted a
   particular structure.

## Artifact inventory

- `results/semantic-ref-ab-20261012T042853Z-9c26c00d/checkpoint.jsonl` — raw
  responses, usage, validation events, and hash chain.
- `results/semantic-ref-ab-20261012T042853Z-9c26c00d/BLINDED_RESPONSES.json` —
  condition-blind human-review set.
- `results/semantic-ref-ab-20261012T042853Z-9c26c00d/BLINDING_KEY.json` — private
  response-to-sample mapping.
- `manifest.json`, `treatment_appendix.txt`, `examples.json`,
  `package_hashes.json`, and `run_experiment.py` — frozen preparation package.
