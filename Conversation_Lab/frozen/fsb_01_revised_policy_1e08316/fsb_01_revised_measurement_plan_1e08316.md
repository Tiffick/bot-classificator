# FSB-01 revised-policy behavioral regression — measurement plan

## Question

Does the selected-only Cognitive Core at HEAD `1e08316` apply the revised
value policy without either a global direct-discrimination bias or a global
testable-working-version bias, while preserving H/R/S/T/A behavior?

Reduction of system uncertainty is not standalone user value. Neither action
type has default priority. Exact DecisionIntent is metadata, not a gold label.

## Phase 0: independent FSB-D qualification gate

Before any production call, compare the two already-frozen FSB-D actions with
the existing symmetric product-policy rubric. Use six blind judgments with
three orientations each way. Proceed only if DIRECT_DISCRIMINATION is higher in
at least four of six judgments, neither orientation accounts for all directional
support, and no structural or protocol failure occurs. A failed or indeterminate
gate stops the run with zero production calls. The gate never changes the case,
actions, prompt, schedule, or later evaluation.

Qualification is an independently authorized execution phase. Its live mode is
hard-capped at six API calls and always stops after writing the qualification
aggregate, including when the gate passes. A passed gate records a review pause;
production requires a later, explicit `--resume-production` authorization for
that exact hash-chained run. Ordinary resume cannot cross the review pause.

The working version is grounded, cautious, correctable, and in-bounds. Direct
discrimination must win, if it does, because its available timing observation
has greater expected useful delta—not because the alternative violates a guard.

## Production phase

After a passed qualification gate, run exactly one `gpt-5-mini` production call
for FSB-H, FSB-M, FSB-R, FSB-S, FSB-T, FSB-A, and FSB-D in that order. Retry is
zero. Complete all seven production calls before post-hoc evaluation. This phase
cannot begin in qualification-only mode.

## Post-hoc evaluation

- H/R/S/T/A: use the historical FSB-01 comparator and frozen challengers,
  preserving their historical meaning.
- M: use the symmetric product-policy rubric against the independently frozen
  FSB-M testable working version. The old direct challenger remains historical
  evidence, not the sole revised-policy gold.
- D: use the same symmetric product-policy rubric against the qualified frozen
  direct-discrimination action.
- Every production comparison receives exactly three blind judgments in frozen
  orientations and deterministic aggregation.
- M/D expose only an expanded observable action projection: intent, content
  kind, semantic content, target interaction, and reply text. No state mutation,
  candidate list, rationale, score, or hidden reasoning is exposed.

## Interpretation

PASS requires valid outputs, no critical ABSTAIN, no challenger-higher aggregate,
no Product Boundary or semantic/context defect, and no interaction mismatch.
A single challenger-higher result is a regression signal, not proof of a
reproducible policy failure; confirmation requires a separate future freeze.
Technical failure, failed qualification, critical ABSTAIN, or protocol drift is
INCONCLUSIVE. No confirmations, retries, adaptive samples, Astra production, or
model routing occur in this run.
