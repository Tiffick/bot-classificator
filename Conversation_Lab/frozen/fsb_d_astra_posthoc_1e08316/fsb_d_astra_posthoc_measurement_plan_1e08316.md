# FSB-D Astra saved-action post-hoc comparison

## Comparison question

For each of the three already-saved Astra FSB-D actions, compare its expected
product value against the same qualified frozen Direct Discrimination action.
No new production selection is made.

The analysis unit is one saved Astra action. Three evaluator judgments are
repeat judgments for that pair, not independent production observations.

## Rubric

The existing symmetric product-policy rubric is reused byte-for-byte. It is
fully applicable because it evaluates arbitrary observable next consultation
actions, not intent-family conformity. It compares Expected Human Model delta,
User Burden, Grounding, Correctability, Epistemic Safety, Forward Value,
Natural Consultation Quality and Product Boundary. It explicitly gives no
default advantage to a question, Reflection, uncertainty reduction, or either
left/right position.

Direct receives no bonus for being Direct. Previous Attempts receives no
penalty for having frozen class OTHER. The evaluator judges observable expected
useful value in the same frozen Discovery context.

## Frozen pairs and order

For A1, then A2, then A3:

1. Astra left / Direct right
2. Direct left / Astra right
3. Astra left / Direct right

All nine calls are fixed before live execution. There is no result-dependent
branching.

## Per-sample aggregation

Blind LEFT/RIGHT verdicts are normalized to `ASTRA_HIGHER`, `DIRECT_HIGHER`,
`EQUIVALENT_OR_BOTH_ACCEPTABLE`, or `ABSTAIN`.

The existing three-repeat selected-action policy is retained:

- three identical verdicts produce that verdict;
- two directional Astra votes produce `ASTRA_HIGHER`;
- two directional Direct votes produce `DIRECT_HIGHER`;
- other materially split patterns produce `ABSTAIN`;
- `EQUIVALENT_OR_BOTH_ACCEPTABLE` requires unanimous equivalence.

## Overall result

The three per-sample aggregates are the independent units:

- Direct 3/3: `DIRECT_REPRODUCIBLY_HIGHER`
- Astra 3/3: `ASTRA_REPRODUCIBLY_HIGHER`
- Direct at least 2/3 and Astra 0: `DIRECT_MAJORITY_HIGHER`
- Astra at least 2/3 and Direct 0: `ASTRA_MAJORITY_HIGHER`
- Equivalent 3/3: `EQUIVALENT_PATTERN`
- Any per-sample ABSTAIN: `INCONCLUSIVE`
- Otherwise: `MIXED_COMPARATIVE_VALUE`

These labels describe only the three frozen FSB-D actions and do not rank the
models generally.

## Budget and authorization

Production calls: zero. Evaluator calls after separate human authorization:
exactly nine, model `gpt-6-astra`, retry zero. `--dry-run` is offline. Live
execution requires explicit `--new-run` or safe `--resume`.
