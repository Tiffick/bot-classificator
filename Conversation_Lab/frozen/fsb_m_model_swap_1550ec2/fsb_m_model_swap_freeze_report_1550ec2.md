# FSB-M model-swap falsification freeze — 1550ec2

This package freezes a three-sample `gpt-5-mini` production arm against the
already completed Astra treatment samples B1/B2/B3 from run
`fsbmab-20260915T020555Z-7c327de8`.

The only intended request difference is `model`: `gpt-6-astra` becomes
`gpt-5-mini`. The FSB-M state, serialized input, prompt, strict wire schema,
internal `CognitiveTurnResult`, selected-only boundary, API family and request
parameters remain fixed. There is no evaluator, retry, confirmation, smoke call,
prompt overlay, candidate materialization or adaptive sampling.

The observable outcome is classified into exactly one of:

- `DIRECT_EXPERIENTIAL_DISCRIMINATION`;
- `REFLECTION_EVALUATION`;
- `OTHER`.

The runner records the complete validated `CognitiveTurnResult`; final semantic
classification remains an offline analysis of those observable outputs. The
future live budget is exactly three production calls.
