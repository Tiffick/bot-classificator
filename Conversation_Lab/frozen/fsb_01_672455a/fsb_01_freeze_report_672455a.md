# FSB-01 offline refreeze report

Result: **READY FOR LIVE RUN**

All six decision states are exact deterministic replays through the production Discovery contracts and `apply_cognitive_turn`. No LLM or API was used. Behavioral cases, challengers, evaluator contract and measurement policy are unchanged from the 0b8366a freeze.

## Reachability

- FSB-H: 3 applied turn(s); state SHA-256 `B66AE6732920603550C7D04398D772D7735DF7C9E69231680EFD713638003A1D`.
- FSB-M: 2 applied turn(s); state SHA-256 `F69D71205B0C25EFB8815839D1882A2C1837C483BFA2E78F2AF07D3141BAE337`.
- FSB-R: 2 applied turn(s); state SHA-256 `053DF0D956958973626CF0D04BA5B74363DF9D18397A2835B31F90CB3AE1597F`.
- FSB-S: 2 applied turn(s); state SHA-256 `6E6A592CDAF9F3E61D9E27A7DFDCDCA92D69361782993810E4A4D882F0A25C66`.
- FSB-T: 2 applied turn(s); state SHA-256 `95B041D24525D78913A18444C61C68D0FBCC32D6AE489B35ABAE0D9C59FF2E13`.
- FSB-A: 1 applied turn(s); state SHA-256 `348FE8AF420B259E9EF4781ECAC7CEB318CA1A7049E2428F7A03863052AEAEBB`.

FSB-A replay produces two distinct persistent ACS response targets owned by two distinct active contents. The current `Да.` can therefore be reconciled as AMBIGUOUS without inventing an unsupported ACS shape.

## Frozen safeguards

- Challengers contain only decision_intent, semantic_focus and basis_evidence_refs.
- Every evidence reference resolves in the corresponding frozen input.
- Challengers were generated before any future production winner and are hashed.
- No T6/T12/T14 state hash is reused.
- No candidate/consideration/evaluator data enters production state.
- Baseline is 24 calls; maximum is 48; retry is zero.
- Production HEAD: `672455a`.
- CognitiveTurnResult schema SHA-256: `CD1C41BB15F3EEB9D1B6D5FDC8FD1EADAE498D69DE6950A65A8AA00797775D39`.
- CognitiveCore wire schema SHA-256: `F37521D307E20C84A7FC9709791EBFBDD2653726F2B506D5D139BD7485C826BF`.
- The wire-only selected_action representation is absent from HumanModel, ACS and DialogueHistory snapshots.

## Artifact SHA-256

- `fsb_01_cases_672455a.json`: `E9CB63B9D54F19EBCEF06A592C83DF70C3E75561C934C0D453BD9D40E914CEA1`
- `fsb_01_challengers_672455a.json`: `23F38F097ED4B2A30902D22EEE128CB88EAE9B446EDA5E8C9912F1752960AF51`
- `fsb_01_evaluator_prompt_672455a.txt`: `C5ACFB34717B6F548CE7F0F65349E7F0C56DA3203580242E1B40A9BD3DB1FCF6`
- `fsb_01_evaluator_schema_672455a.json`: `6D9B6D568D3F58C1D622F18B93FB3D428CA9631AB0B3C8029C134E9270FCB54B`
- `fsb_01_measurement_plan_672455a.md`: `2A83EAB01B8F5D2E4E2BC08B82063453FD9CF00F452273206913CD24CDD425E0`
- `fsb_01_offline_validator_672455a.py`: `B8F71D2DC24631D363E61FF0AABB41BC2806E499EF3979CC066B7FBC4EB837F1`
