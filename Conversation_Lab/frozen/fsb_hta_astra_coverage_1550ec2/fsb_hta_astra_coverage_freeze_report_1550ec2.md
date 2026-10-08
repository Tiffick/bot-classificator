# FSB-H/T/A Astra Coverage Freeze — 1550ec2

- Purpose: fill the missing observable Astra coverage cells for FSB-H, FSB-T, and FSB-A.
- This is not a prompt, architecture, or model-ranking experiment.
- Source states and challengers are byte-bound to `fsb_01_672455a`.
- Production and evaluator models are both `gpt-6-astra`.
- Frozen order: H, T, A; all production calls precede evaluator calls.
- Each case receives one production sample and three frozen-orientation blind judgments.
- Retry, confirmation, adaptive sampling, smoke calls, and challenger generation are disabled.
- Exact future budget: 3 production + 9 evaluator = 12 calls.
- Semantic analysis must inspect selected content and target semantics, not DecisionIntent alone.
