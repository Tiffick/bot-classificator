# Frozen matched FSB-M prompt clarification A/B

- HEAD: `1550ec2`.
- Three independent production samples per arm are frozen before live execution.
- Balanced production order: A1, B1, B2, A2, A3, B3.
- Control uses the exact prompt blob from `d34fdca`; treatment uses the exact
  prompt blob from `1550ec2`.
- Both arms use `gpt-6-astra`, the same FSB-M state, challenger, production
  contract/wire schema/projection, selected-only boundary and evaluator.
- All six production calls complete before the eighteen evaluator calls.
- Each sample receives three blind judgments in the unchanged FSB-M orientation
  `(challenger-left, selected-left, challenger-left)`.
- No retries, confirmations, smoke calls, adaptive sampling or prompt tuning.
- Exact future API budget: 6 production + 18 evaluator = 24.
