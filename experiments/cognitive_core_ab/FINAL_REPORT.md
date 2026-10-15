# Cognitive Core A/B/C/D experiment — cumulative report

Status: **fresh eight-call pilot completed and failed; full run not authorized by gate**

## 1. EXPERIMENT OBJECTIVE

The experiment tests four competing explanations for questionnaire-like or
paraphrasing-heavy Cognitive Core behavior:

- the production prompt may be overloaded;
- the `SemanticTurn` output contract may constrain conversational quality;
- a simpler conversational instruction may improve the response;
- an explicit conditional-paraphrasing policy may improve responses without
  removing useful repetition.

The purpose is comparative evidence for a later engineering decision, not to
justify a preselected prompt or architecture change.

## 2. EXPERIMENT CONFIGURATION

### Conditions

- **A:** current production prompt plus strict `SemanticTurn`.
- **B:** reduced consultant prompt plus the same strict `SemanticTurn`.
- **C:** minimal conversational prompt and plain-text output.
- **D:** A plus the frozen conditional-paraphrasing policy.

Production source files, schemas, validators, model family, Human Experience
selection and scenario inputs are read-only inputs to the lab experiment.

### Scenarios and sampling

- S1: recurring difficulty;
- S2: rejection of a hypothesis;
- S3: resolution of contradiction;
- S4: insufficient evidence;
- S5: ambiguous short answer;
- S6: meaningful mechanism.

The full target is three independent selections per condition and scenario:
`4 × 6 × 3 = 72` calls. Evaluation is human/read-only using grounding, added
understanding, progress, naturalness, correction handling, unwarranted
inference, contract validity and repetition classification (`NONE`,
`FUNCTIONAL`, `REDUNDANT`). There are no evaluator API calls.

### Corrected configuration (v2)

- requested snapshot: `gpt-5-mini-2025-08-07`;
- reasoning effort: `minimal`, consistently for A/B/C/D;
- maximum completion tokens: `4000`, consistently for A/B/C/D;
- strict JSON Schema remains enabled for A/B/D; C remains plain text;
- OpenAI client retries: `0`;
- new-run cost gate: USD 1.00;
- mandatory pilot: A/B/C/D on S1 and S3, exactly eight calls.

The exact snapshot is the snapshot returned by the previous alias-based run.
Official OpenAI documentation lists a 128,000-token maximum output and current
prices of USD 0.25/1M input, USD 0.025/1M cached input and USD 2.00/1M output:
https://developers.openai.com/api/docs/models/gpt-5-mini

## 3. EXECUTION HISTORY

### Stage 1 — offline preparation

The first runner froze A/B/C/D, S1–S6, three samples per cell, an interleaved
call order, prompt/input/schema hashes, append-only checkpointing, no retries
and a 72-call hard limit. Its dry-run and offline project tests passed.

### Stage 2 — first full live run (invalid)

After explicit authorization, run `cc-abcd-20261011T015406Z-ea892bd5`
executed all 72 calls. The transport run completed without API failures and
with a valid checkpoint hash chain, but its 800-token completion ceiling was
too small for reasoning plus visible output. Sixty-six calls exhausted the
ceiling; A/B/D produced no visible output in any sample. The result was
therefore declared **INVALID / INCONCLUSIVE**, not a behavioral failure of
those conditions.

### Stage 3 — corrected-run preparation

The previous result directory was left unchanged. The lab runner was revised
to use the exact snapshot, `minimal` reasoning, a 4000-token ceiling, latency
telemetry, a USD 1.00 new-run gate and a fail-closed eight-call pilot before
the rest of the schedule. Valid pilot calls count toward the 72-call target.
Resume accounting now treats every API request as an attempt and refuses to
retry a key whose attempt lacks a response.

### Stage 4 — mandatory corrected pilot (failed and stopped)

Run `cc-abcd-v2-20261011T021544Z-9f1be97e` attempted five calls and then
stopped. Four S1 responses were durably saved. All four had non-empty visible
output, `finish_reason=stop`, zero reasoning tokens and no token exhaustion.
A.S1.1, B.S1.1 and C.S1.1 passed their applicable validation. D.S1.1 returned
JSON but failed `SemanticTurn` validation because it referred to newly added
items through `existing_id` values such as `i1_skipped_meal_then_binge`
instead of local `handle` references.

The fifth request, A.S3.1, reached the API and returned to the process, but the
lab harness raised `TypeError` while preparing validation because Python
`deepcopy()` attempted to mutate the immutable `FrozenDict` inside the frozen
previous ACS. The raw response and usage were therefore not durably appended.
No retry was made. The pilot could not reach eight calls, and the full run was
not started, exactly as required by the fail-closed pilot rule.

### Stage 5 — offline harness repair

No API calls were made. The crash was localized to generic Python
`deepcopy(previous_acs)`: Pydantic's copy machinery attempted to populate the
contract's immutable `FrozenDict` mappings in place, which correctly raised
`TypeError`. The harness now reconstructs each model deterministically using
`type(value).model_validate(value.model_dump(mode="python"))`. This preserves
strict validation, produces independent model instances and does not mutate or
reinterpret ACS semantics.

All S1–S6 fixtures were audited offline. Their current message, Human Model,
previous ACS (where present) and DialogueHistory round-trip to byte-equivalent
JSON representations, and a minimal valid `SemanticTurn` adapts successfully
for every scenario. Deterministic IDs, timestamps, payload hashes and request
settings were checked across repeated fixture construction.

Ten lab-only regression tests passed, including an exact reproduction of the
old `deepcopy(FrozenDict)` exception and six parametrized S1–S6 adaptation
checks. The failed pilot directory is also explicitly excluded from resume
discovery, so A.S3.1 cannot be attempted again through the normal runner.
`py_compile`, experiment dry-run and `git diff --check` also passed.
No production file, prompt, schema or previous result artifact was changed.

### Stage 6 — fresh isolated pilot after harness repair

Run `cc-abcd-v3-20261011T023640Z-d13f5f46` executed exactly eight calls and
stopped. Raw response, usage, latency and cost were fsync-persisted before a
separate `SEMANTIC_VALIDATION` event for every call. The checkpoint contains
8 requests, 8 raw responses and 8 validation events in a valid hash chain.
There were no retries, API failures, evaluator calls or full-run calls.

All eight responses were non-empty, returned `finish_reason=stop`, used zero
reasoning tokens and avoided output exhaustion. The pilot nevertheless failed:
only 2/6 A/B/D structured samples passed strict `SemanticTurn` adaptation.
A/B/D on S1 used local item handles as `existing_id`; A.S3.1 used a persistent
ACS ID in `reply_segments.realizes` instead of a declared content handle.
B.S3.1 and D.S3.1 were valid. Both C samples were non-empty plain text.

## 4. PILOT RESULTS

### Failed v2 pilot

**FAIL.** The output-budget repair worked for all four recorded calls, but the
pilot failed two mandatory criteria:

1. D.S1.1 was not a valid `SemanticTurn`.
2. The harness crashed during A.S3.1 validation, so all eight calls did not
   complete and persist.

Recorded calls:

| Key | Visible | Finish | Reasoning | Validation |
| --- | --- | --- | ---: | --- |
| A.S1.1 | yes | stop | 0 | valid SemanticTurn |
| B.S1.1 | yes | stop | 0 | valid SemanticTurn |
| C.S1.1 | yes | stop | 0 | valid plain text |
| D.S1.1 | yes | stop | 0 | invalid SemanticTurn |

Known telemetry for those four responses: 21,747 input tokens, 0 cached input,
2,194 output tokens, 0 reasoning tokens, 23,941 total tokens and USD
0.00982475. A.S3.1 is a fifth API call with an unrecorded response; its exact
usage and billed cost cannot be recovered from the checkpoint. Its frozen
conservative maximum was USD 0.01346375.

### Fresh v3 pilot

**FAIL.** All eight calls completed and persisted, but strict structured-output
validity was only 2/6 for A/B/D.

| Key | Visible | Finish | Input | Cached | Output | Reasoning | Valid | Cost USD |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| A.S1.1 | yes | stop | 7,608 | 7,552 | 1,075 | 0 | no | 0.00235280 |
| B.S1.1 | yes | stop | 5,780 | 5,632 | 671 | 0 | no | 0.00151980 |
| C.S1.1 | yes | stop | 435 | 0 | 710 | 0 | yes (plain text) | 0.00152875 |
| D.S1.1 | yes | stop | 7,924 | 7,808 | 779 | 0 | no | 0.00178220 |
| A.S3.1 | yes | stop | 7,732 | 7,680 | 210 | 0 | no | 0.00062500 |
| B.S3.1 | yes | stop | 5,904 | 4,864 | 513 | 0 | yes | 0.00140760 |
| C.S3.1 | yes | stop | 559 | 0 | 183 | 0 | yes (plain text) | 0.00050575 |
| D.S3.1 | yes | stop | 8,048 | 2,816 | 289 | 0 | yes | 0.00195640 |

Totals: 43,990 input, 36,352 cached input, 4,430 output, 0 reasoning,
48,420 total tokens, USD 0.01167830. Non-empty rate: 8/8. Token-exhaustion
rate: 0/8. Structured validity: A 0/2, B 1/2, D 1/2, combined 2/6.

## 5. FULL EXPERIMENT RESULTS

No valid full comparative result exists. The fresh pilot failed its mandatory
contract-validity criterion, so the corrected full schedule remains blocked.

## 6. COMPARATIVE ANALYSIS

The fresh pilot supplies matched visible observations but is not sufficient to
rank conditions: it contains one sample per condition for only S1 and S3, and
four structured outputs are invalid. Observable pilot tendencies are:

- A vs D: both made local/existing-reference errors on S1; D was valid on S3
  while A was not. Both still paraphrased user material.
- A vs B: neither was valid on S1; B was valid on S3. B's replies remained
  questionnaire-like and asked for detail beyond the correction.
- B vs C: C avoided contract failures because it had no contract, but its S1
  answer crossed Product Boundary into an extensive solution menu and its S3
  answer asked three questions. Contract removal therefore did not by itself
  produce better consultation.
- B vs D: both were valid on S3; D used Recognition while B asked another
  clarification. One sample cannot establish superiority.

The pilot does not support proceeding to 72 calls and does not establish any
prompt or architecture winner.

## 7. REPRESENTATIVE RAW RESPONSES

The first run preserved every raw API response in its checkpoint. A typical
A/B/D failure is an empty assistant content with `finish_reason=length`, 800
completion tokens and 800 reasoning tokens. Six observable C replies existed
only for S2/S3; they acknowledged or repeated the correction and then asked
multiple questions and/or offered solution design. They are observations from
an incomplete comparison and are not treated as evidence of C superiority or
inferiority.

Complete raw data:
`results/cc-abcd-20261011T015406Z-ea892bd5/checkpoint.jsonl`.

All eight fresh-pilot assistant contents are preserved verbatim in:
`results/cc-abcd-v3-20261011T023640Z-d13f5f46/checkpoint.jsonl`. Their keys are
`A.S1.1`, `B.S1.1`, `C.S1.1`, `D.S1.1`, `A.S3.1`, `B.S3.1`, `C.S3.1`, and
`D.S3.1`; no response was truncated or overwritten.

## 8. CONFIRMED FINDINGS

- The original 800-token ceiling was inadequate for this reasoning-model
  configuration.
- The first transport execution made exactly 72 calls with no retries and no
  evaluator calls, and its checkpoint hash chain is valid.
- No comparative behavioral conclusion can be drawn from that run.
- The corrected minimal-reasoning/4000-token configuration reliably produced
  visible output in the fresh pilot: 8/8 non-empty, 8/8 `stop`, 0 reasoning
  tokens and 0 token-exhaustion events.
- Strict contract reliability failed the pilot: 2/6 valid A/B/D samples.
- Removing `SemanticTurn` did not automatically improve consultation: both C
  replies showed substantive behavioral defects despite being deliverable.

## 9. UNCERTAINTIES AND LIMITATIONS

- The corrected 4000-token/minimal-reasoning configuration eliminated output
  exhaustion but did not pass the strict contract-validity pilot gate.
- D.S1.1 demonstrates that strict schema enforcement can still reject a
  visible answer; one sample does not establish a condition-level rate.
- The v2 pilot harness had an ACS-copying defect for scenarios with a previous
  ACS. This defect is now repaired offline through
  deterministic Pydantic reconstruction, without weakening the production
  schema.
- Remaining blocker: D.S1.1 is still a genuine recorded contract-invalid model
  output. It remains part of the failed pilot and was not altered or regraded.
- Both failed pilots are terminal artifacts and must never be resumed.
- Three samples per scenario describe variability but cannot establish
  statistical superiority.
- Human rubric judgments remain subjective and must not be presented as
  objective proof.
- A/B/D include strict structured output while C does not; that is the
  intended contract comparison but also a known bundled difference.

## 10. RECOMMENDED NEXT ACTION

Do not start the full experiment. First perform a read-only root-cause audit of
why the model repeatedly encodes newly declared local items as `existing_id`
under the strict schema and why A.S3.1 used an ACS ID in `realizes`. Determine
whether this is prompt/schema operationalization or model execution before any
new pilot. Do not weaken validation, and do not infer conversational quality
from contract-free C alone.

## 11. TOTAL EXPERIMENT ACCOUNTING

### Previous invalid run

- API calls: 72 (A/B/C/D: 18 each)
- input tokens: 393,834
- cached input tokens: 330,880
- output tokens: 56,456
- reasoning tokens: 55,296
- total tokens: 450,290
- actual cost: USD 0.1369225
- non-empty visible outputs: 6/72
- valid A/B/D `SemanticTurn`: 0/54

### Corrected run

- API calls: 5 attempted, no retries
- durably recorded responses: 4
- known input tokens: 21,747
- known cached input tokens: 0
- known output tokens: 2,194
- known reasoning tokens: 0
- known total tokens: 23,941
- known actual cost: USD 0.00982475
- fifth-call actual usage/cost: unavailable because the harness failed before
  durable response recording; conservative maximum USD 0.01346375

### Combined historical total

- API calls: 85
- known input tokens: 459,571
- known cached input tokens: 367,232
- known output tokens: 63,080
- known reasoning tokens: 55,296
- known total tokens: 522,651
- known actual cost: USD 0.15842555
- combined actual cost including A.S3.1 is not exactly recoverable from local
  artifacts; conservative upper bound: USD 0.17188930

## 12. ARTIFACT INVENTORY

### Experiment package

- `README.md`
- `run_experiment.py`
- `test_run_experiment.py`
- `ANALYSIS.md`
- `FINAL_REPORT.md` (this authoritative cumulative report)

Current lab code hashes after the offline repair and durable-response update:

- `run_experiment.py` SHA-256:
  `9D105C40FE1A647C44507B6216148A90E397F9014D1CFE8570BD86E904D68DA5`
- `test_run_experiment.py` SHA-256:
  `227EAD87DFA86B3CC9D3F721271A82DADB7D1FCB7B13B71C3E637937AAC79B6D`

### Previous invalid run

- directory: `results/cc-abcd-20261011T015406Z-ea892bd5/`
- `checkpoint.jsonl` SHA-256:
  `973BD3B226B87EB66F31162C4A0D1349C9DF4F5C14EB718364D42ED0E3204FCB`
- `manifest.json` SHA-256:
  `4E728BC0F79D00D2838253C4EF0C52468195600DA64FCFEA98FD2ADD4A140DED`
- `prompts.json` SHA-256:
  `5B27678B009509283991DBCEED37803D850DECFC602C37F008303B7A94C93B26`
- `RUN_COMPLETED`

No corrected run overwrote these files.

### Failed corrected pilot

- directory: `results/cc-abcd-v2-20261011T021544Z-9f1be97e/`
- `checkpoint.jsonl` SHA-256:
  `6C92E3E94EEC5541A2833D970CF5375279A410350C0EF2467635E7BCB98F73A8`
- `manifest.json` SHA-256:
  `69467650FFF67F340FB892199D86B617A3785F137C04A2C65F59A46AB498DB9E`
- `prompts.json` SHA-256:
  `5B27678B009509283991DBCEED37803D850DECFC602C37F008303B7A94C93B26`
- checkpoint chain: valid through the final A.S3.1 `API_REQUEST` event
- no `PILOT_PASSED` and no `RUN_COMPLETED` marker

### Fresh isolated v3 pilot

- directory: `results/cc-abcd-v3-20261011T023640Z-d13f5f46/`
- `checkpoint.jsonl` SHA-256:
  `79CE42045C7E7FB3F989E52D949D64227EC53407747A2131F21BEFF7E74352AE`
- `manifest.json` SHA-256:
  `89DE80C3E6F19331D0538BA3E93169BFF727457907171DE6D900C049693E8321`
- `prompts.json` SHA-256:
  `5B27678B009509283991DBCEED37803D850DECFC602C37F008303B7A94C93B26`
- `PILOT_FAILED` SHA-256:
  `76373998D1271B4D07A9205BB96D20B42E2D7BF233E6A9E84F80BD061FDE9560`
- final checkpoint event SHA-256:
  `EDB6ADBAEE3489460701EDF98FFC679D758183A80683FA3C61C7A9A64B3C4FE5`
- checkpoint chain: valid through 24 events (8 request, 8 response,
  8 validation)
- terminal marker: `PILOT_FAILED`; no `PILOT_PASSED` and no
  `RUN_COMPLETED` marker
