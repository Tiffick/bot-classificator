# FSB-01 revised-policy confirmation freeze — HEAD 1e08316

- This is a fixed confirmation of the existing M/R/S/T/D candidate misses, not
  a new policy experiment.
- States, prompt, models, comparators, evaluator contracts, orientations and
  observable projections are referenced from immutable artifacts by SHA.
- The completed source run and its five original `CHALLENGER_HIGHER` aggregates
  are bound by run/checkpoint SHA and validated offline.
- Two new production selections per case are frozen before live execution.
  Evaluator votes are measurement repeats, not production pseudoreplications.
- All ten production calls precede all thirty evaluator calls.
- Retry, smoke, adaptive sampling, result-conditioned calls and protocol edits
  are disabled. Hard maximum: 10 production + 30 evaluator = 40 API calls.
- Live execution requires an explicit command after human review; preparation
  and dry-run perform zero API calls.

