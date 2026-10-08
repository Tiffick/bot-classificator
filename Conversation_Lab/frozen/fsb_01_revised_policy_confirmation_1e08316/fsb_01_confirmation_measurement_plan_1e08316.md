# FSB-01 revised-policy non-adaptive confirmation — measurement plan

## Confirmation question

Do the five `CHALLENGER_HIGHER` signals observed for FSB-M/R/S/T/D in run
`fsbrev-20260921T043353Z-975a17f7` recur under byte-identical states, prompt,
models, benchmarks, projections and evaluator semantics?

The unit of repeatability is an independent production selection. Evaluator
votes measure one selection and are never counted as independent production
replications.

## Fixed repetitions and order

Run two new independent production selections per case. Together with the
already frozen original selection this yields three observable production
selections per case. Two new selections are the minimum that permits a
predeclared distinction between both confirmations, neither confirmation and a
split result.

Complete all ten production calls before any evaluator call. Production order
is round-major: M/R/S/T/D repetition 1, then M/R/S/T/D repetition 2. Evaluate
each sample with exactly three blind judgments in the inherited frozen
orientation order. Retry, confirmation-on-result, adaptation and extra calls
are forbidden.

## Comparators

- FSB-M: unchanged symmetric product-policy rubric and frozen Testable Working
  Version B; expanded observable projection.
- FSB-D: unchanged symmetric product-policy rubric and qualified frozen Direct
  Discrimination benchmark; expanded observable projection.
- FSB-R/S/T: unchanged historical FSB-01 rubric and challenger projection.

## Per-case classification

- `REPRODUCED`: both new independent production samples aggregate to
  `CHALLENGER_HIGHER`.
- `NOT_REPRODUCED`: neither new sample aggregates to `CHALLENGER_HIGHER`, and
  both comparisons are decisive and protocol-valid.
- `MIXED`: exactly one of the two new samples aggregates to
  `CHALLENGER_HIGHER`, while the other is decisive and protocol-valid.
- `INCONCLUSIVE`: any critical `ABSTAIN`, structural/transport failure,
  state/benchmark drift, mapping ambiguity or protocol violation.

The original sample remains historical evidence and is not recounted as a new
call or evaluator vote.

## Overall verdict

- `FAILURE_PATTERN_REPRODUCED`: all five cases are `REPRODUCED`.
- `FAILURE_PATTERN_PARTIALLY_REPRODUCED`: no case is inconclusive, not all five
  reproduce, and at least one case is `REPRODUCED` or `MIXED`.
- `FAILURE_PATTERN_NOT_REPRODUCED`: all five cases are `NOT_REPRODUCED`.
- `INCONCLUSIVE`: any case is `INCONCLUSIVE` or the frozen protocol fails.

The observable preliminary/local-Reflection pattern is audited from the ten
selected actions after the fixed run. It is not inferred from evaluator votes,
DecisionIntent alone or hidden reasoning.

