# FSB-01 full selected-only production regression — HEAD 1550ec2

This freeze references the original FSB-01 H/M/R/S/T/A cases, independently
frozen challengers, evaluator prompt/schema and baseline orientations byte for
byte from `Conversation_Lab/frozen/fsb_01_672455a`.

The production boundary is the current selected-only Cognitive Core at HEAD
`1550ec2`, prompt SHA
`4FEB93BE179B7F7976F099EF828A8B578EBAB6CA0C06FC0C09CD147EF2C0AA14`,
with `gpt-5-mini`. All six production calls complete before the eighteen blind
`gpt-6-astra` evaluator calls. There are no retries, confirmations, adaptive
samples, prompt overlays, state edits, candidate materialization or challenger
generation. The exact future budget is 24 API calls.

FSB-M analysis must inspect whether a Mechanism Discovery question truly
discriminates the working explanations; intent family alone is not success.
