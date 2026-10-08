# FSB-D D-only matched model-swap measurement plan

## Question

Does changing only the production model from `gpt-5-mini` to
`gpt-6-astra` change the observable FSB-D selection from local Reflection to
the already-qualified direct experiential discrimination?

This is not a general model-quality comparison. It tests only the localized
`MODEL EXECUTION — FORMULATION→DECISION` hypothesis for one frozen state.

## Matched input

All three samples reuse the exact FSB-D input, prompt, wire schema,
`CognitiveTurnResult`, selected-only boundary, request parameters, and one-call
production route from the revised-policy run. The historical mini request must
reconstruct to:

`C477473587CB6E1228C41A34ADBBD56A27F8B17C4881C3107D14E9743B0AED18`

The Astra request may differ only at the root `model` key. Removing that key
from both requests must yield:

`69255152863A307218646267AB5B9BD74B572AAB97327F41C124C906039772AF`

No model-specific temperature, reasoning, token-limit, tool, seed, response
format, or prompt setting is added. The client transport settings remain
`max_retries=0` and `timeout=120.0`. Provider-side implementation differences
belong to the intended model variable.

## Frozen order and budget

1. `FSB-D-A1`
2. `FSB-D-A2`
3. `FSB-D-A3`

Each slot is one independent `CognitiveCore.propose()` call. There are no
evaluator, smoke, confirmation, retry, or adaptive calls. Maximum: exactly
three production API calls.

## Deterministic observable classification

Classification uses only the selected action projection: `decision_intent`,
all ActionContent kinds and semantic content, ResponseTarget interactions, and
the actual reply. It does not infer hidden candidates or reasoning.

`DIRECT_DISCRIMINATION` requires all of:

- `MECHANISM_DISCOVERY`;
- `SYSTEM_QUESTION`;
- `OPEN_RESPONSE` or `CLARIFICATION`;
- observable language asking for the strong desire/hunger/eating impulse;
- both temporal sides: before the difficult conversation and after it;
- a question/discrimination cue rather than confirmation of the user's stated
  uncertainty.

The frozen classifier normalizes case/`ё`, whitespace and punctuation and
recognizes fixed Russian morphological stems for desire/hunger/eating,
before/after, difficult conversation, and question/discrimination. This is a
closed classifier for FSB-D, not a general NLP policy.

`LOCAL_REFLECTION` requires `REFLECTION` + `SYSTEM_REFLECTION`, failure to meet
the Direct rule, and a recap of at least two already-given D elements (missed
lunch, difficult conversation, overeating, or the already-stated timing
uncertainty), without a cautious hunger/tension Working Version.

Everything else is `OTHER`, including Working Version, another mechanism
question, Transition, or semantic/intent mismatch.

## Frozen verdict

- Direct 3/3 → `STRONG_MODEL_SENSITIVITY_SUPPORT`
- Direct 2/3 → `MODEL_SENSITIVITY_SUPPORT`
- Local Reflection 3/3 → `MINI_SPECIFICITY_FALSIFIED`
- every other distribution → `MIXED_MODEL_EXECUTION`

Evaluator votes are absent; the independent unit is each new production
selection.

## Safety and resume

The runner is fail-closed. Every successful API response is recorded between
`CALL_PREPARED` and `CALL_COMPLETED` in an append-only hash chain. A dangling
prepared/response call, STOP, hash mismatch, non-prefix order, request drift,
or completed run cannot be resumed. Live execution requires explicit
`--new-run` or `--resume`; no mode performs nothing.
