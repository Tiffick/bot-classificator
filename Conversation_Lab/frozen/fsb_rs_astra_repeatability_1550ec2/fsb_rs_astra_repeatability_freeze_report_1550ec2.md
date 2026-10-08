# FSB-R/S Astra Repeatability Freeze — 1550ec2

- Purpose: measure repeatability of previously observed Astra behavior on FSB-R and FSB-S.
- This is not a prompt, architecture, or candidate-materialization experiment.
- Source states and challengers are byte-bound to `fsb_01_672455a`.
- Production and evaluator models are both `gpt-6-astra`.
- Frozen sample order: R1, R2, R3, S1, S2, S3.
- All six production calls precede all evaluator calls.
- Each sample receives exactly three frozen-orientation blind judgments.
- Retry, confirmation, adaptive sampling, smoke calls, and challenger generation are disabled.
- Exact future budget: 6 production + 18 evaluator = 24 calls.
- Semantic analysis must inspect content, not DecisionIntent alone.
