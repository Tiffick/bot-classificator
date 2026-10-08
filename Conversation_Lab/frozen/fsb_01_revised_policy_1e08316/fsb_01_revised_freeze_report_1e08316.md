# FSB-01 revised-policy freeze — HEAD 1e08316

- Original H/M/R/S/T/A contexts and historical challengers are referenced by
  immutable SHA from `fsb_01_672455a`; they are not copied or modified.
- The historical comparator remains the H/R/S/T/A instrument.
- The frozen FSB-M product rubric, schema, guards, and working-version action are
  referenced by SHA from `fsb_m_product_policy_ab_1550ec2`.
- FSB-D is a new realistic control. Both actions are grounded, safe, and useful;
  the qualification gate tests whether the available timing observation has
  greater expected useful delta.
- Phase 0 uses six balanced blind judgments. Only a passed frozen gate permits
  the seven production calls. Qualification-only execution is capped at six
  calls and pauses after its aggregate even on PASS; production requires a
  separate explicit resume-production authorization. Later evaluation is
  post-hoc and non-interventional.
- Production is `gpt-5-mini`; evaluator is `gpt-6-astra`; retry/smoke/
  confirmations/adaptation/Astra-production are disabled.
- Hard maximum: 7 production + 27 evaluator = 34 API calls.
