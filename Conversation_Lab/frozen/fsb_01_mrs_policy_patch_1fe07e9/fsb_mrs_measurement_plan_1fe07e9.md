# FSB-M/R/S focused policy-patch regression

## Question

Does the Decision-policy patch at HEAD 1fe07e9 remove the three previously
reproducible MISSED HIGH-VALUE ALTERNATIVE results without technical,
intent-semantic, or Product Boundary regression?

No inference about latent consideration, hidden cognition, candidate generation,
ranking, rationale, or chain-of-thought is permitted.

## Frozen execution

1. Verify HEAD, every package SHA, and the offline validator before any API call.
2. Run exactly one selected-only `CognitiveCore.propose()` call for FSB-M, FSB-R,
   and FSB-S in that order with `gpt-5-mini`, retry 0.
3. Only after all three production outputs are structurally valid, run exactly
   three blind `gpt-6-astra` judgments per case in frozen case/orientation order.
4. Use the existing `selected_action_evaluator` normalization and aggregation.
5. No adaptive confirmation, smoke, challenger generation, retry, or extra call.

Exact future budget: 3 production + 9 evaluator = 12 API calls.

## Frozen interpretation

- **PATCH REGRESSION PASS**: no case aggregates to `CHALLENGER_HIGHER`; every
  production output is valid; there is no semantic intent mismatch or new
  Product Boundary violation.
- **PATCH REGRESSION FAIL**: at least one valid comparison aggregates to
  `CHALLENGER_HIGHER`.
- **INCONCLUSIVE**: technical/structural failure, critical `ABSTAIN`, state or
  challenger drift, or evaluator protocol violation.

DecisionIntent family is not gold. Comparative quality against the independently
frozen challenger is primary.
