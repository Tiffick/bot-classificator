# Semantic Delta A/B — Blind Human Review Protocol

## Unit of review

One independently generated assistant reply is one observation. Technical
validator results and the A/B condition are hidden during consultation-quality
review. Evaluator LLMs are prohibited.

Two human reviewers should score the shuffled response set independently. If
only one reviewer is available, the result remains descriptive and the missing
inter-reviewer check must be reported.

## Evidence shown to a reviewer

For each blinded response:

- scenario title and exact conversational input (Human Model, previous ACS,
  relevant history and current USER message) rendered without condition labels;
- exact user-facing reply;
- no sample key, condition, repetition, technical validity or token usage.

Scenario evaluation-focus notes in `scenarios.json` are instructions for the
analysis owner, not content shown beside individual responses.

## Required ratings

Use the following ordinal scale for each dimension:

- `0` — clear failure or harmful handling;
- `1` — limited/mixed;
- `2` — clearly adequate.

1. **Meaningful semantic progress** — compared with what the USER had already
   said, does the response create useful understanding or purposeful progress?
2. **Grounding** — are connections and conclusions supported by the supplied
   user material?
3. **No invented cause/mechanism** — does the response avoid presenting an
   unsupported explanation as belonging to this person?
4. **Question value** — if a question is asked, could plausible answers
   materially change understanding or the next useful move? If no question is
   needed and none is asked, score `2`.
5. **Naturalness and appropriateness** — does it read like a natural consultant
   move rather than a questionnaire or internal process?
6. **Stage fit** — is the move appropriate for the current Discovery state and
   Product Boundary?

For S5 and S6, absence of a new hypothesis must not reduce any score by itself.

## Required categorical annotations

Record independently from the ordinal ratings:

- `unnecessary_question`: `YES / NO / UNCERTAIN`;
- `unsupported_user_claim`: `YES / NO / UNCERTAIN`;
- `mechanical_paraphrase`: `YES / NO / UNCERTAIN`;
- `reply_conveys_declared_action_meaning`: `YES / PARTIAL / NO / NOT_ASSESSABLE`.

The last annotation is performed in a second, still condition-blind pass where
the reviewer may see the selected ActionContent and reply mapping. It must not
be inferred from the reply-only quality pass.

## Technical analysis kept separate

For every sample, report separately:

- generation completed;
- JSON parsing;
- strict JSON Schema validation;
- `SemanticTurn.model_validate`;
- deterministic adapter;
- `CognitiveTurnResult` validation;
- selected intent, ActionContent and ResponseTargets.

Technical validity is not a consultation-quality score and cannot by itself
make either condition the winner.

## Comparison

Primary comparison: paired A versus B distribution for meaningful semantic
progress, reported overall and by scenario. Also report every other dimension,
categorical annotation, and raw reviewer disagreements. Do not create a single
weighted total unless a weighting rule is frozen before responses are viewed.

With three repetitions per condition/scenario, results are descriptive. Report
counts and paired differences; do not claim statistical effectiveness or broad
production generalization.
