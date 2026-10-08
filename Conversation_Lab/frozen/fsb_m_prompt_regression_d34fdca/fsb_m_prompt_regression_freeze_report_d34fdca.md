# Focused FSB-M prompt regression freeze

- Production HEAD: `d34fdca`.
- Source state/challenger: exact FSB-M bytes and semantic hashes from
  `fsb_01_mrs_policy_patch_1fe07e9`.
- Experimental change: production prompt SHA changes from
  `EFADD7944F729F4FB76BC5184AD009A81C173224D2B9AA4938A1C84355410920` to
  `38FB4AF63196302FE4BFF5CFA4A202E5548DC9A0F93385DA07B1B1CB23597475`.
- Arm A: one `gpt-6-astra` production call.
- Arm B: one `gpt-5-mini` production call.
- Both production calls finish before six post-hoc blind evaluator calls.
- The frozen challenger, evaluator model/prompt/schema/aggregation/orientation,
  wire schema, projection, selected-only boundary and retry policy are unchanged.
- Confirmations, retries, smoke calls and adaptive changes are forbidden.
- Exact live budget: two production plus six evaluator calls, total eight.
