# Semantic Delta A/B — Readiness Report

## Question

Does a compact clarification of semantic delta make Cognitive Core more often
produce useful understanding rather than a renamed paraphrase, without
increasing unsupported hypotheses or unnecessary questions?

The design implements Candidate 1 from
`experiments/COGNITIVE_CORE_ACTION_SELECTION_AUDIT.md`. It is a lab prompt
intervention, not a production change.

## Conditions

- **A — Control:** exact current assembled production Cognitive Core prompt,
  including the current Human Experience system instruction.
- **B — Treatment:** A plus only `treatment_appendix.txt`.

The technical `handle` / `existing_id` / `realizes` appendix from the prior
Semantic Reference experiment is not included. A and B use the same strict
scenario-specific JSON Schema, inputs, model and generation parameters.

## Scenarios

- **S1:** exact frozen recurring-mechanism fixture from the prior experiment;
  a grounded P001 professional version is available.
- **S3:** exact frozen correction/contradiction fixture from the prior
  experiment.
- **S4:** insufficient information for a mechanism; a useful focal question is
  allowed, invented causality is not.
- **S5:** simple correction of a previous statement with no material new
  ambiguity; correction should not become a new questionnaire.
- **S6:** a user expresses a meaningful experience where a short grounded
  reflection can be useful and artificial mechanism-building is unnecessary.

The expected principles are evaluation metadata and are never sent to the
model. S4-S6 have deterministic HumanModel, previous ACS and DialogueHistory.
Human Experience is retrieved through the same current conservative production
selector; abstention is preserved when no card is grounded.

## Repetitions and future budget

The frozen design uses three independent selections per condition/scenario:

`2 conditions × 5 scenarios × 3 repetitions = 30 API calls maximum`.

Three repetitions per cell are the smallest useful continuation of the prior
pilot that exposes obvious variability while keeping cost and review burden
bounded. It is a descriptive falsification/repeatability check, not a powered
statistical trial. The deterministic order is counterbalanced: A appears first
in 8 matched pairs and B in 7.

Configuration:

- model: `gpt-5-mini-2025-08-07`;
- reasoning effort: `minimal`;
- max completion tokens: `4000`;
- retries: `0`;
- evaluator calls: `0`;
- future hard API budget: `30`;
- conservative maximum cost estimate: `$0.407683` under the prices recorded in
  `manifest.json` (actual cost will depend on tokens/cache).

## Human evaluation

Quality is evaluated condition-blind under `BLIND_REVIEW_PROTOCOL.md`. The
primary dimension is useful semantic progress. Grounding, unsupported claims,
question value, naturalness and stage fit are kept visible as separate
dimensions. S5/S6 are explicitly protected against a rubric that rewards
hypothesis production for its own sake.

Technical SemanticTurn validity and completeness of ActionContent transfer to
the reply are reported separately. Neither is treated as proof of consultation
quality. No LLM evaluator is used.

## Reproducibility and safety

The manifest freezes:

- current Git HEAD and relevant file hashes;
- exact A/B prompt hashes;
- treatment and scenario-catalog hashes;
- scenario input and JSON Schema hashes;
- model/configuration;
- 30-call order.

`run_experiment.py` defaults to dry-run. Live execution additionally requires
`--execute` and the exact authorization token. It writes an append-only
hash-chained checkpoint, durably stores each raw API response before validation,
does not repair invalid outputs, has no retry/fallback/adaptive/evaluator call,
and produces a deterministic blind set plus a separate key after the frozen
schedule. It also writes a condition-blind ActionContent/reply view for the
separate transfer-completeness pass, so that this check does not require opening
the condition key.

## Known limitations

1. Three samples per cell cannot establish statistical effectiveness.
2. The treatment adds prompt length as well as semantic guidance; those effects
   cannot be separated in this experiment.
3. S1/S3 are reused fixtures but are now tested against the current production
   assembly; historical outputs are comparators, not samples in this run.
4. S4-S6 are synthetic controls. They are realistic but do not represent the
   full variety of production conversations.
5. Human evaluation may remain subjective; reviewer disagreements must be
   preserved rather than collapsed silently.
6. A valid Structured Output does not prove that the chosen action was the most
   useful one.

## Status

Offline validation must pass against `package_hashes.json` and the frozen
dependency block before any future API call. Live execution requires separate
human authorization.

`PRODUCTION CHANGES = 0`
`API CALLS = 0`
`EXPERIMENT STATUS = READY`
