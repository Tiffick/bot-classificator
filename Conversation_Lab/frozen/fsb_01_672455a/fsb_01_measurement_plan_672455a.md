# FSB-01 frozen measurement plan

## Question

Does the current selected-only Cognitive Core reproducibly select an observable action that is lower-value than an independently frozen eligible challenger?

The observable defect is **MISSED HIGH-VALUE ALTERNATIVE**. No inference about latent consideration, hidden cognition, candidate ranking, or chain-of-thought is permitted.

## Execution

1. Verify HEAD and every frozen SHA. A mismatch stops the run with zero API calls.
2. Run one production `CognitiveCore.propose()` call for each case in frozen baseline order, using `gpt-5-mini`, retry 0. Finish all six production calls before evaluator execution.
3. Compare every completed selected action post-hoc with its frozen challenger using three blind `gpt-6-astra` judgments and the frozen orientations.
4. If and only if a baseline aggregate is `CHALLENGER_HIGHER`, run one additional production call on the exact same frozen state, then three comparator judgments with the frozen confirmation orientations.
5. Never pass evaluator output back to production. Never modify a challenger after freeze.

## Aggregation

Use `Conversation_Lab.selected_action_evaluator.aggregate_judgments` unchanged: unanimous verdict is stable; 2/3 matching directional judgments form a directional majority; substantial disagreement or uncertainty aggregates to ABSTAIN.

## Frozen interpretation

- **CURRENT POLICY ACCEPTABLE**: all six baseline production results are valid, no aggregate is CHALLENGER_HIGHER, no critical comparison is ABSTAIN, and declared intent is semantically aligned with the selected action.
- **POLICY CHANGE JUSTIFIED**: at least one case is CHALLENGER_HIGHER both at baseline and its single conditional confirmation, with an eligible grounded challenger and no technical or measurement failure.
- **INCONCLUSIVE**: any isolated non-reproduced miss, critical ABSTAIN, invalid production/evaluator output, invalid challenger, insufficient context, or result not attributable to one concrete Decision-policy boundary.

Exact DecisionIntent family is coverage metadata, not the primary gold criterion. Pairwise semantic value against the independently frozen challenger is primary.
