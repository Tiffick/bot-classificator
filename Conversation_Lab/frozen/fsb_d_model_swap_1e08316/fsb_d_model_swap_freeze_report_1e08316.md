# FSB-D D-only model-swap freeze report

- Production HEAD: `1e08316`
- Historical baseline: three independent `gpt-5-mini` FSB-D selections, all
  local Reflection and all losing to the frozen Direct comparator 3/3.
- New production model: `gpt-6-astra`
- Repetitions: three, fixed before live execution.
- Evaluator calls: zero.
- Only intended request change: root `model` identifier.
- FSB-D state, Direct action, prompt, schema, contract, projection boundary and
  parameters are referenced by immutable SHA and are not copied or modified.
- The historical mini request is reconstructable byte/canonically to the
  recorded SHA; the Astra request and model-excluded request SHA are frozen in
  the manifest.
- Observable classification and final interpretation are fixed before live
  execution.
- No result from this experiment supports a claim about hidden reasoning or
  general model superiority.
