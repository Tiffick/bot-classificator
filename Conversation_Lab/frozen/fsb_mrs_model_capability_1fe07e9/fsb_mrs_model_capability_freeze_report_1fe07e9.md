# FSB-M/R/S matched model-capability isolation freeze

- Production HEAD: `1fe07e9`.
- Source freeze: `Conversation_Lab/frozen/fsb_01_mrs_policy_patch_1fe07e9`.
- Cases and order: `FSB-M`, `FSB-R`, `FSB-S`.
- States, challengers, production prompt, wire schema, projection, evaluator prompt,
  evaluator schema, orientations, aggregation, retry policy and selected-only boundary
  are bound byte-for-byte to the source freeze.
- The only experimental variable is the production CognitiveCore model:
  `gpt-5-mini` becomes `gpt-6-astra`.
- Evaluator model remains `gpt-6-astra`.
- Live budget is exactly three production calls followed by nine blind evaluator
  calls. Smoke calls, retries, confirmations and adaptive calls are forbidden.
- Verdict interpretation is frozen in the manifest before live execution.
