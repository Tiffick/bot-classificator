# Semantic Delta A/B — Technical Report

## Run identity

- Run ID: `semantic-delta-ab-20261013T010202Z-8b1ed0be`
- Model requested and observed: `gpt-5-mini-2025-08-07`
- Reasoning effort: `minimal`
- Maximum completion tokens: `4000`
- Retries: `0`
- Conditions: `A`, `B`
- Scenarios: `S1`, `S3`, `S4`, `S5`, `S6`
- Repetitions: `3` per condition/scenario combination
- API calls: `30` (exactly the frozen schedule)

## Preflight

All frozen preflight checks passed before the first API request:

- production assembly matched the frozen hashes;
- condition A was byte-identical to the production assembly;
- condition B was exactly A plus the frozen treatment appendix;
- the JSON Schema hash matched the frozen schema;
- all five scenario fixtures matched their frozen hashes;
- manifest and package hashes validated;
- the schedule contained exactly 30 unique calls.

## Execution integrity

- API requests: `30`
- API responses: `30`
- Validation records: `30`
- API failures: `0`
- Finish reason `stop`: `30/30`
- Retries, evaluator calls, adaptive calls, and extra calls: `0`
- Checkpoint hash chain: valid (`90` events)
- Final event SHA-256: `AB0E8159226237DE7620996282E8345DF80426E2F3724D1AE9DBC0239B04A4CF`
- Checkpoint file SHA-256: `5DEFB4E1CEAA571A467B5231165D646AE1F61A704C6D8C41DFAD00FB6DA33EF9`

Raw API responses were appended and flushed to `checkpoint.jsonl` before semantic validation. Invalid outputs were retained unchanged.

## Technical validation results

### By condition

| Condition | JSON parse | JSON Schema | SemanticTurn | Adapter | CognitiveTurnResult / complete |
|---|---:|---:|---:|---:|---:|
| A | 15/15 | 15/15 | 13/15 | 10/15 | 10/15 |
| B | 15/15 | 15/15 | 11/15 | 10/15 | 10/15 |

### Complete validity by scenario

| Scenario | A | B |
|---|---:|---:|
| S1 | 2/3 | 0/3 |
| S3 | 3/3 | 3/3 |
| S4 | 3/3 | 3/3 |
| S5 | 1/3 | 2/3 |
| S6 | 1/3 | 2/3 |
| **Total** | **10/15** | **10/15** |

### Invalid samples and first failure

| Sample | First failing layer | Exact failure class |
|---|---|---|
| A.S1.1 | SemanticTurn | Local item handles (`it_problem_pattern`, `it_previous_attempts`, `it_emotional_response`) were supplied as `existing_id`; persistent item IDs must start with `mi_`. |
| A.S5.1 | Adapter | `ordinary_addressed` was applied to a previous target whose interaction was neither `OPEN_RESPONSE` nor `CLARIFICATION`. |
| A.S5.2 | Adapter | `ordinary_addressed` was applied to a previous target whose interaction was neither `OPEN_RESPONSE` nor `CLARIFICATION`. |
| A.S6.2 | SemanticTurn | Local item handle `itm_self_recognition_loss_1` was supplied as `existing_id`; persistent item IDs must start with `mi_`. |
| A.S6.3 | Adapter | Reply realization referenced unknown semantic handle `reply_content_1`. |
| B.S1.1 | SemanticTurn | Local item handles were supplied as `existing_id`; persistent item IDs must start with `mi_`. |
| B.S1.2 | SemanticTurn | Local item handles were supplied as `existing_id`; persistent item IDs must start with `mi_`. |
| B.S1.3 | SemanticTurn | Local item handles were supplied as `existing_id`; persistent item IDs must start with `mi_`. |
| B.S5.1 | Adapter | `ordinary_addressed` was applied to a previous target whose interaction was neither `OPEN_RESPONSE` nor `CLARIFICATION`. |
| B.S6.1 | SemanticTurn | Local item handle `it_seen_self_changed` was supplied as `existing_id`; persistent item IDs must start with `mi_`. |

All 30 outputs passed JSON parsing and JSON Schema validation. The stricter Pydantic SemanticTurn validation caught six invalid reference-addressing outputs. The deterministic adapter caught four additional cross-object or interaction failures. No invalid response was repaired or resubmitted.

## Token accounting

| Condition | Input | Cached input | Output | Reasoning | Total |
|---|---:|---:|---:|---:|---:|
| A | 116,472 | 104,192 | 7,920 | 0 | 124,392 |
| B | 121,152 | 110,336 | 9,092 | 0 | 130,244 |
| **Total** | **237,624** | **214,528** | **17,012** | **0** | **254,636** |

Under the pricing assumptions frozen in the manifest, the measured run cost is approximately `$0.0451612` (`$0.0215148` for A and `$0.0236464` for B).

## Blind evaluation materials

- `BLINDED_RESPONSES.json`: user/context material and assistant replies, without condition labels or validation results.
- `BLINDED_ACTION_CONTENT.json`: selected ActionContent and corresponding reply segments, without condition labels or validation results.
- `BLINDING_KEY.json`: separate mapping from blind IDs to condition/sample identities; it must not be supplied with either evaluation set.

SHA-256:

- `BLINDED_RESPONSES.json`: `E0DD7B7BDD9B5D2E13981C48FDF44B735D9DD0AC28335B8C6098AE55FB2E99B9`
- `BLINDED_ACTION_CONTENT.json`: `1FD76E2CFAA8C0D7D43DA39A03C60EFAB22995E98DD8A6F242A500FFF384A03D`
- `BLINDING_KEY.json`: `B0DB0B7A7EDDF4EE285BA7685681FE1CD87878E2EA765046D15CF6ABF405C1B9`

## Interpretation boundary

### CONFIRMED

- Both conditions achieved complete technical validity for `10/15` samples.
- Their scenario distributions differed: B was lower on S1, equal on S3/S4, and higher on S5/S6.
- The treatment did not improve aggregate technical validity in this 30-call run.

### OBSERVED

- The most frequent invalidity was confusing a new local handle with an existing persistent item ID.
- JSON Schema acceptance alone did not establish full SemanticTurn validity; Pydantic and adapter checks caught all recorded technical failures.

### UNKNOWN

- Consultation quality has not been judged.
- The experiment does not establish a behavioral winner.
- With three repetitions per cell, scenario-level differences may be sampling variation and require independent human evaluation and, if justified later, additional preregistered evidence.
