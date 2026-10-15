# Cognitive Core — Prompt Engineering Compliance Audit

Дата: 2026-10-12
Режим: STRICT READ-ONLY для production; создан только этот отчёт.
Стандарт: `experiments/Prompt_Engineering_Rules_Research_2026-10-11.md`
(R01–R20).
Предыдущая фактическая реконструкция:
`COGNITIVE_CORE_PROMPT_ASSEMBLY_AUDIT.md`.

Шкала уверенности: **HIGH / MEDIUM / LOW**.
Шкала серьёзности: **CRITICAL / HIGH / MEDIUM / LOW / NONE**.
Серьёзность означает потенциальное влияние, а не доказанную частоту дефекта.

## 1. Executive Summary

Текущая production-инструкция — не бессодержательный «будь хорошим
консультантом» prompt. Она явно задаёт consultation value, ограничивает
уточнения, различает Reflection/Recognition, запрещает ritual echo и solution
design, описывает uncertainty и использует native strict Structured Output.
Это подтверждённые сильные стороны.

Одновременно аудит выявил три наиболее значимых зоны риска.

1. **CRITICAL — operationalization of semantic references.** Prompt объясняет
   `handle` и `existing_id`, но без компактной таблицы и положительных/ошибочных
   полных примеров. Сгенерированная JSON Schema представляет `existing_id`
   просто как `string`; префикс и существование проверяются только Python после
   ответа. В fresh pilot 4 из 6 A/B/D outputs не прошли адаптацию: A/B/D на S1
   использовали local handles как `existing_id`, A.S3 использовал persistent
   ACS ID в `realizes` (`experiments/cognitive_core_ab/FINAL_REPORT.md:128-141,
   172-187`). Это подтверждает существование failure class, но **не доказывает**,
   что причиной является только формулировка prompt.

2. **HIGH — incomplete action-selection ordering.** Prompt задаёт важный
   приоритет: clarification только при material delta; когда механизм уже
   grounded — Reflection/Recognition/Stop/other progression. Но между
   Reflection, Recognition, новой областью, STOP и answer нет полного
   operational tie-breaker. `expected consultation value`, `meaningful` и
   `достаточно grounded` остаются оценочными. Это может оставлять модели
   привычный безопасный шаблон «пересказ + вопрос», но причинная связь пока
   только гипотеза.

3. **HIGH — task/context load and trust separation.** Один вызов одновременно
   интерпретирует пользователя, обновляет Human Model, reconciles ACS, выбирает
   action, формирует natural reply и согласует semantic references. Base prompt
   содержит 61 bullet; с HE instruction — 17,547 characters, отдельно schema —
   около 18.2k characters. DialogueHistory находится внутри JSON data, но у
   history entries отсутствует поле `role` (`ai/cognitive_core.py:376-381,
   389-391`). Также нет прямой инструкции игнорировать embedded commands в
   пользовательском/history/card тексте. Это подтверждённые свойства входа;
   их реальное влияние требует отдельной проверки.

Общий ответ на главный вопрос: текущая организация **может** способствовать
механическому пересказу, потому что структурно проверяемые обязанности
(grounding, state update, ActionContent, reply realization) можно формально
выполнить, повторив текущий материал, тогда как «добавить meaningful insight»
оценивается только prose policy и не имеет schema-level критерия. Она **может**
недостаточно продвигать консультацию из-за неполного tie-breaker между
допустимыми action regimes. Ошибки SemanticTurn имеют более прямое объяснение:
сложный reference lifecycle, отсутствие few-shot examples и невозможность
schema проверить существование/тип конкретных ссылок. Ни один из этих
механизмов пока не доказан как единственная причина.

## 2. R01–R20 Compliance Matrix

### Summary matrix

| Rule | Краткое название | Применимость | Verdict | Severity | Confidence |
|---|---|---:|---|---|---|
| R01 | Наблюдаемая цель | да | PARTIAL | MEDIUM | HIGH |
| R02 | Одна главная задача | да | PARTIAL | HIGH | HIGH |
| R03 | Разделить инструкции и данные | да | PARTIAL | HIGH | HIGH |
| R04 | Конкретные исполнимые инструкции | да | PARTIAL | MEDIUM | HIGH |
| R05 | Нет скрытой конкуренции | да | PARTIAL | HIGH | MEDIUM |
| R06 | Контекст по необходимости | да | PARTIAL | MEDIUM | HIGH |
| R07 | Разнообразные few-shot examples | да | FAIL | HIGH | HIGH |
| R08 | Осторожные контрпримеры | да | PARTIAL | LOW | HIGH |
| R09 | Правило выбора next action | да | PARTIAL | HIGH | HIGH |
| R10 | Разделить стиль и содержание | да | PASS | NONE | HIGH |
| R11 | Работа с uncertainty | да | PASS | NONE | HIGH |
| R12 | Corrections и short answers | да | PARTIAL | MEDIUM | HIGH |
| R13 | Contract не заменяется examples | да | PASS | NONE | HIGH |
| R14 | Namespace и lifecycle IDs | да | FAIL | CRITICAL | HIGH |
| R15 | Schema не проверяет все связи | да | PASS | NONE | HIGH |
| R16 | Декомпозиция только по измерениям | да | PARTIAL | MEDIUM | MEDIUM |
| R17 | Model/version awareness | да | FAIL | MEDIUM | HIGH |
| R18 | Не перегружать ритуальными rules | да | PARTIAL | MEDIUM | HIGH |
| R19 | Trust boundary внешнего текста | да | PARTIAL | HIGH | HIGH |
| R20 | Failure/recovery design | да | PARTIAL | HIGH | HIGH |

### Detailed findings

#### R01. Сначала определить наблюдаемую цель — PARTIAL

**Поддержка.** Prompt называет цель «полезное понимание, а не максимальная
полнота данных» (`ai/prompts/cognitive_core_system_prompt.txt:16-41`), задаёт
наблюдаемые запреты на questionnaire/echo (`84-103`) и формальную цель strict
output (`1-11,133-143`). Product Boundary также имеет различимые allowed и
forbidden outcomes (`70-78,112-127`).

**Проблема.** Критерии `meaningful`, `grounded`, `полезное понимание` и
`expected consultation value` не превращены в единый per-turn observable
acceptance rule. Contract validity наблюдаема, а quality delta — нет.

**Оценка:** PARTIAL; severity MEDIUM; confidence HIGH.

#### R02. Указать одну главную задачу каждого вызова — PARTIAL

**Поддержка.** Lines 1-11 явно называют один full Cognitive Discovery turn и
порядок «понять → StatePatch → выбрать action → reply». Lines 145-147 требуют
Decision before Response.

**Проблема.** Один call выполняет минимум шесть самостоятельных обязанностей:
interpretation, state patch, reconciliation, decision selection, natural
response, reference bookkeeping/grounding. Нет общей hierarchy вида
contract validity vs semantic usefulness vs style. «LLM решает WHAT» задаёт
границу, но не разрешает конфликт между natural reply и сложным wire.

**Оценка:** PARTIAL; severity HIGH; confidence HIGH.

#### R03. Развести роль, инструкции, данные, примеры и формат — PARTIAL

**Поддержка.** System instructions и user JSON разделены; schema передаётся
отдельно через `response_format` (`ai/cognitive_core.py:285-322,186-197`).
Prompt имеет sections A/B/C. Human Experience обозначен как hypotheses, not
facts (`ai/human_experience.py:39-56`).

**Проблема.** DialogueHistory entries передают только `{id, sequence, text}` —
без `role` (`ai/cognitive_core.py:376-381,389-391`). Поэтому внутри dynamic
data происхождение отдельных historical utterances не маркировано явно.
Few-shot examples отсутствуют, поэтому смешения examples/data нет, но и
обучающей функции examples тоже нет.

**Оценка:** PARTIAL; severity HIGH; confidence HIGH.

#### R04. Писать конкретные, исполнимые инструкции — PARTIAL

**Поддержка.** Anti-echo имеет positive alternative: вступление допустимо,
если добавляет connection/consequence/pattern/insight (`prompt:86-88`).
Clarification gate задан через material change (`27-41`). Correction behavior
описан (`52-58`).

**Проблема.** Значимые decision terms (`meaningful`, `context-close`,
`достаточно grounded`, `natural progression`) остаются без operational
boundary examples. Для сложных ссылок prose конкретен, но не демонстрирует
полный correct object.

**Оценка:** PARTIAL; severity MEDIUM; confidence HIGH.

#### R05. Не допускать противоречий и скрытой конкуренции — PARTIAL

**Поддержка.** Direct question vs Product Boundary явно разрешён: answer if it
fits understanding/explanation, Transition if solution design is required
(`prompt:70-78,112-127`). Clarification не запрещена полностью (`129-130`).

**Проблема.** Нет явной глобальной priority hierarchy между hard contract,
grounding, action utility и naturalness. Appended HE says «Prefer one grounded,
easily correctable observation» and «Do not demand confirmation»
(`ai/human_experience.py:42-56`), тогда как base prompt одновременно допускает
Evaluation для consequential hypothesis (`prompt:42-51,169-191`). Это не
формальное противоречие, но создаёт дополнительный conditional layer.

**Оценка:** PARTIAL; severity HIGH; confidence MEDIUM.

#### R06. Давать контекст по принципу необходимости — PARTIAL

**Поддержка.** DialogueHistory ограничена closure: last 8 плюс source messages,
нужные HumanModel/ACS (`ai/cognitive_core.py:428-453`). HE ограничен P001/P006/
P007, максимум 3 cards и 6000 chars (`ai/human_experience.py:23-25,240-260`).

**Проблема.** HumanModel items/relations передаются полностью, без relevance
filter/size limit (`ai/cognitive_core.py:344-374`). History closure может
расширяться всеми source refs. Роли history удалены. Нельзя считать это
избыточностью без реальных размеров production state, но boundedness
асимметрична.

**Оценка:** PARTIAL; severity MEDIUM; confidence HIGH.

#### R07. Снабжать сложные правила разнообразными few-shot examples — FAIL

Prompt не содержит полных `input → correct SemanticTurn` examples для ADD vs
existing references, correction, short reply, rejection, ambiguity или
reply realization. Есть prose examples/phrases, но не few-shot demonstrations.
При сложности 37 schema definitions это правило применимо.

**Оценка:** FAIL; severity HIGH; confidence HIGH.

#### R08. Показывать контрпримеры осторожно — PARTIAL

Prompt использует несколько forbidden phrase examples («Правильно ли я
понял?», «понял(а)», meta-questions) и сразу объясняет условие допустимости
(`prompt:86-103`). Однако bad structured-reference examples не сопоставлены с
correct alternatives; большая часть section B — negative rules без paired
output.

**Оценка:** PARTIAL; severity LOW; confidence HIGH.

#### R09. Определять правило выбора следующего действия — PARTIAL

**Поддержка.** Question допустим при material delta; grounded recurring
mechanism должен вести к Reflection/Recognition/Stop/other progression
(`prompt:27-47`). Intent roles перечислены (`144-167`). Correction и boundary
также имеют отдельные routes.

**Проблема.** Нет полного tie-breaker между Reflection, Recognition,
STOP_EXPLORATION, другой областью и bounded answer. «Выбери по expected
consultation value» задаёт цель, но не разрешает близкие cases. Поэтому
questionnaire или local Reflection остаются формально возможными прочтениями.

**Оценка:** PARTIAL; severity HIGH; confidence HIGH.

#### R10. Отдельно управлять стилем и содержанием — PASS

Natural/warm style задан отдельно от substantive rules (`prompt:16-22`), а
anti-echo привязан к added meaning, не к запрещённым словам (`86-103`). Output
text отделён от `semantic_content`; literal equality не требуется
(`255-257`).

**Оценка:** PASS; severity NONE; confidence HIGH.

#### R11. Явно задавать работу с неопределённостью — PASS

Prompt чётко различает USER material, USER interpretation, other-person view,
system hypothesis и HE (`prompt:48-69,104-111`). Rejected/uncertain behavior
задано. HE cards названы optional hypotheses, not evidence
(`ai/human_experience.py:42-56`).

**Оценка:** PASS; severity NONE; confidence HIGH.

#### R12. Прописывать поведение при коррекциях и коротких ответах — PARTIAL

Correction flow задан (`prompt:50-58`), «не знаю» считается valid answer
(`79-81`), bare yes не подтверждает несколько targets (`110-111`), а
reconciliation имеет AMBIGUOUS/outcome semantics (`169-199`).

Недостаёт full examples, показывающих, как short answer связывается с previous
target и как correction одновременно обновляет state и выбирает новый action.
Отсутствие role в history может дополнительно затруднять retrospective context.

**Оценка:** PARTIAL; severity MEDIUM; confidence HIGH.

#### R13. Не подменять примерами спецификацию контракта — PASS

Contract задан native strict JSON Schema, wire models и application validator,
а не примерами (`ai/cognitive_core.py:186-197`;
`ai/cognitive_semantic_wire.py:44-319,322-618`). Примеры отсутствуют, но
формальная спецификация присутствует.

**Оценка:** PASS; severity NONE; confidence HIGH.

#### R14. Для identifiers описывать namespace и lifecycle — FAIL

**Поддержка.** Prompt правильно говорит: existing objects use exact persistent
IDs; new objects/current content declare handles; handles are turn-local and
share one namespace; previous targets use special input handles
(`prompt:201-224`).

**Проблема.** Нет требуемой compact table и positive/negative structured
examples. JSON Schema для `_ExistingItem.existing_id` и
`_ReinforceItem.existing_id` содержит только `type:string`: Python
`AfterValidator` prefix check не экспортируется в schema
(`ai/cognitive_semantic_wire.py:58-70,93-110`). Existence and correct object
kind проверяются adapter (`372-391`). `realizes` schema знает только handle
pattern, не declaration existence (`280-289,592-599`).

Fresh pilot наблюдал именно эти failure classes (`FINAL_REPORT.md:128-141,
172-187`).

**Оценка:** FAIL; severity CRITICAL; confidence HIGH.

#### R15. Не рассчитывать, что strict schema проверит links — PASS

Проект не полагается только на schema: adapter fail-closed проверяет existing
membership, local declarations/types, previous targets, compatibility,
materialization и realizes references (`ai/cognitive_semantic_wire.py:
337-618`). `CognitiveTurnResult` добавляет cross-field invariants
(`ai/cognitive_turn.py:374-412,438-553`). Невалидные pilot outputs записаны как
failures, а не silently accepted.

**Оценка:** PASS; severity NONE; confidence HIGH.

#### R16. Разделять задачи, когда сложность превышает reliability — PARTIAL

Техническое ID assignment уже вынесено в deterministic Python, сохраняя one
LLM call (`prompt:6-11`; adapter `ai/cognitive_semantic_wire.py:322-618`). Это
соответствует правилу не добавлять стадии без evidence. Но pilot показывает
2/6 strict validity; полноценный controlled comparison не завершён
(`FINAL_REPORT.md:172-192`). Поэтому необходимость дальнейшей декомпозиции
остаётся UNKNOWN, а текущая достаточность one-call contract не подтверждена.

**Оценка:** PARTIAL; severity MEDIUM; confidence MEDIUM.

#### R17. Учитывать особенности модели и версии — FAIL

Production requests alias `gpt-5-mini` и не задаёт reasoning effort или output
budget (`ai/cognitive_core.py:153-159,186-197`). Lab experiment отдельно
фиксировал snapshot and parameters (`FINAL_REPORT.md:48-50`), но это не меняет
production request. Следовательно, exact production behavior не полностью
version-pinned/reproducible.

**Оценка:** FAIL; severity MEDIUM; confidence HIGH.

#### R18. Не перегружать prompt ритуальными instructions — PARTIAL

Base prompt: 257 lines, 16,322 chars, 2,101 whitespace-delimited words,
61 bullets. Enabled HE appends 1,224 chars and 6 bullets; combined system
content is 17,547 chars. Hypothesis safety, anti-paraphrase, correction and
question optionality повторяются между base and HE. Повторы перечислены в
предыдущем audit (`COGNITIVE_CORE_PROMPT_ASSEMBLY_AUDIT.md:355-376`).

Размер сам по себе не является defect. Проблема — повторная salience одних
норм при отсутствии examples для более сложного reference lifecycle.

**Оценка:** PARTIAL; severity MEDIUM; confidence HIGH.

#### R19. Применять trust boundary к внешнему тексту — PARTIAL

System instructions отделены от user JSON; HE явно обозначен как optional
knowledge, not facts. Однако prompt не говорит прямо, что embedded
instructions внутри current user text, history или card fields являются
untrusted data и не могут менять system policy. DialogueHistory role labels
также исключены из projection.

**Оценка:** PARTIAL; severity HIGH (potential security/integrity impact);
confidence HIGH для отсутствия rule, UNKNOWN для exploitability.

#### R20. Проектировать failure and recovery — PARTIAL

Сильная сторона: `max_retries=0`, ошибки parsing/schema/adaptation разделены и
fail-closed (`ai/dialog_engine.py:252-260`; `ai/cognitive_core.py:198-248`). Нет
silent semantic repair; только narrowly bounded quote-case canonicalization
(`ai/cognitive_core.py:77-147`).

Недостаток: production path не содержит ограниченного recovery strategy для
invalid/empty SemanticTurn; exception завершает turn. Pilot подтвердил
несколько invalid outputs. Это может быть осознанным fail-closed решением, но
полный user-facing recovery contract не определён.

**Оценка:** PARTIAL; severity HIGH; confidence HIGH.

## 3. Instruction Competition Analysis

### 3.1 Existing partial priority order

Из prompt можно восстановить неполный порядок:

1. Уважить correction/refusal/pause (`prompt:52-58,79-81`).
2. Direct user question has priority, но остаётся внутри Product Boundary
   (`70-78`).
3. Clarification допустим только при material uncertainty/delta (`27-35`).
4. Если recurring mechanism достаточно grounded, не продолжать вопросы ради
   полноты; выбрать Reflection, Recognition, Stop, другую область или natural
   progression (`36-41`).
5. Если нужна personal solution design — Transition + consent (`70-78,
   112-127`).

Это реальная policy, а не отсутствие guidance.

### 3.2 What remains unordered

Prompt не даёт однозначного tie-breaker:

- Reflection vs Recognition, когда обе grounded;
- Reflection vs bounded answer;
- STOP_EXPLORATION vs перейти в другую meaningful area;
- professional HE hypothesis vs synthesis only;
- optional acknowledgement vs direct substantive start.

Определения intent есть, но они описывают функцию выбранного action, а не
полный selection algorithm (`prompt:144-167`).

### 3.3 Why mechanical paraphrase can be the easiest formal completion

**Observed facts:** model must emit state operations with current source quotes,
ActionContent, reply segments and references. Reflection/Recognition must be
grounded. Schema validates structure, not whether reply adds new understanding.
Anti-echo exists only as prose. In fresh S1 samples A/D repeated the user’s
chain and asked evaluation; those outputs were additionally contract-invalid.

**Hypothesized mechanism:** restating the current USER chain simultaneously
satisfies visible grounding, fills `semantic_content`, makes source references
easy and gives `reply_segments` something clearly related to selected action.
Producing a genuinely new but safe observation requires finer balancing of
uncertainty, HE guards and Product Boundary. Therefore echo may be a low-risk
local solution for the model even though explicitly prohibited.

**Not proven:** pilot has one S1 sample per condition and several invalid
outputs. It does not estimate frequency or establish that schema causes echo.
Condition C showed that removing contract does not automatically produce good
consultation (`FINAL_REPORT.md:194-210`).

## 4. SemanticTurn Instruction Analysis

### New vs existing objects

- Text instruction: correct distinction at `prompt:201-214`.
- Schema: new references use `{handle}`, existing use `{existing_id}`.
- Gap: schema does not expose the `mi_`/`rel_` prefix validators for
  `existing_id`; membership/type are post-response checks.
- Operational ambiguity: no complete correct sample shows ADD followed by
  ActionContent reference to the new handle, contrasted with an existing ID.

### References between objects

- Prompt states shared handle namespace and owner/subject distinction
  (`201-224`).
- Schema constrains shapes and enums.
- Adapter alone proves declaration existence and reference type
  (`ai/cognitive_semantic_wire.py:337-391,554-599`).

### Previous ACS

- Input assigns `previous_target_N` handles
  (`ai/cognitive_core.py:46-55,398-426`).
- Prompt explicitly forbids copying `rt_`/`aci_` for reconciliation
  (`220-224`).
- Adapter maps handles to persistent targets and validates outcome/interaction
  compatibility (`ai/cognitive_semantic_wire.py:443-518`).
- This area is comparatively explicit, but has no multi-target example.

### Reply segments and selected action

- Prompt says `ReplySegment.realizes` names current ActionContent and targeted
  content must be realized (`255-257`).
- Schema enforces only handle syntax/length and non-empty segment text.
- Adapter/Internal contract enforce that the handle was declared and every
  targetable content is realized (`ai/cognitive_semantic_wire.py:592-617`;
  `ai/cognitive_turn.py:506-518`).
- A.S3.1’s persistent `aci_...` in `realizes` is direct observed evidence that
  prose + schema did not prevent this failure in that pilot configuration.

### Prevention boundary

| Error class | Schema prevents | Text warns | Python catches after output |
|---|---:|---:|---:|
| Extra fields / missing declared keys | yes | yes | yes |
| Invalid enum/branch shape | yes | yes | yes |
| Malformed handle syntax/length | yes | indirectly | yes |
| `existing_id` has correct persistent prefix | no | yes | yes |
| Existing object really exists | no | yes | yes |
| Local handle was declared and has correct type | no | yes | yes |
| Previous target handle exists | no | yes | yes |
| Outcome compatible with interaction | not fully | yes | yes |
| `realizes` names declared current content | no | yes | yes |
| `source_quote` exact/unique/current | only non-empty | yes | yes |
| Reply adds understanding rather than echo | no | yes | human evaluation only |

## 5. Prompt Complexity Analysis

### Measured surface

- Base production prompt: 257 lines, 16,322 chars, ~2,101 whitespace words,
  61 bullet instructions.
- Enabled HE instruction: 20 lines, 1,224 chars, 6 bullets.
- Combined system content: 17,547 chars.
- Semantic schema: ~18.2k compact JSON chars, 37 definitions.
- Representative dynamic user JSON in previous audit: 2,642 chars, but real
  HumanModel/history can be larger.
- Few-shot input/output examples: 0.

### Independent tasks in one call

1. Interpret current user meaning.
2. Reconcile previous ACS.
3. Produce evidence-grounded HumanModel changes.
4. Select one next action/intent.
5. Compose natural user-facing reply.
6. Construct a cross-referenced SemanticTurn satisfying schema and hidden
   application-level invariants.
7. Optionally integrate selected HE while preserving epistemic guards.

### Distribution of information

- Behavioral policy: base system prompt.
- Additional HE policy: appended system block.
- User/state/knowledge data: one JSON user message.
- Structural types: separate JSON Schema.
- Referential/business integrity: mostly Python after output, partially
  restated in prose.

This distribution is architecturally coherent, but model-visible information
about one operation is spread across prose, schema and dynamic IDs. Large size
alone does not prove overload; the stronger concern is unequal support:
repetitive behavioral guards, but no demonstrations of the most failure-prone
reference operations.

## 6. Root-Cause Hypotheses

### H1 — Semantic-reference operationalization is insufficient

1. **Observed:** no full namespace examples; schema `existing_id` is plain
   string; 4/6 structured pilot outputs invalid, with local/existing and
   realizes mistakes.
2. **Mechanism hypothesis:** model learns object shapes but not reliably the
   lifecycle distinction across a long instruction set.
3. **Could explain:** SemanticTurn validation failures.
4. **Not proven:** a compact table/example would fix reliability; model or API
   structured-output behavior may remain limiting.
5. **Minimal test:** frozen lab-only matched pilot on S1/S3, current prompt vs
   identical prompt with only one compact namespace table and two positive +
   one explicitly marked invalid reference examples; fixed snapshot/settings,
   strict schema, 3 independent selections per cell, no evaluator needed for
   validity.

### H2 — Action-selection policy lacks a decisive tie-breaker

1. **Observed:** partial priority exists, but Reflection/Recognition/Stop/new
   area are not totally ordered.
2. **Mechanism hypothesis:** model defaults to familiar local reflection plus
   another question when several actions remain allowed.
3. **Could explain:** questionnaire continuation and insufficient progression.
4. **Not proven:** current prompt causes this pattern systematically; full A/B
   experiment is incomplete.
5. **Minimal test:** frozen selected-action semantic scenarios with no style
   grading: grounded mechanism, material ambiguity, exhausted axis, correction.
   Compare current policy against one-factor explicit tie-breaker; 3 production
   selections/scenario, classify observable action semantics before live.

### H3 — Structural grounding makes repetition a locally easy solution

1. **Observed:** output must state patch, content and reply; meaningful delta
   has no structural validator; S1 A/D repeated the user chain.
2. **Mechanism hypothesis:** reuse of USER wording minimizes unsupported
   inference and reference risk.
3. **Could explain:** mechanical paraphrase despite explicit prohibition.
4. **Not proven:** correlation is based on a tiny, failed pilot; C also produced
   poor behavior without schema.
5. **Minimal test:** matched S1/S6 run with identical schema and policy, changing
   only an operational output criterion that permits direct synthesis without
   re-encoding unchanged user facts; predefine functional/redundant rubric.

### H4 — Missing history roles weakens provenance resolution

1. **Observed:** projected history lacks role; only sequence/text are sent.
2. **Mechanism hypothesis:** model may confuse prior system proposals with user
   statements, particularly during correction/reconciliation.
3. **Could explain:** unsupported reinforcement, recycled hypotheses or wrong
   correction targets.
4. **Not proven:** alternating dialogue may be inferred; no trace establishes
   an actual role-confusion failure.
5. **Minimal test:** matched frozen correction/multi-target scenarios with
   identical content, only adding explicit history roles in lab projection;
   compare reconciliation validity and provenance errors.

### H5 — Repeated HE/base guards alter salience

1. **Observed:** hypothesis safety, anti-paraphrase and correction rules occur
   in both base and appended HE instructions.
2. **Mechanism hypothesis:** repeated caution could favor conservative echo or
   question over a novel observation.
3. **Could explain:** weak Recognition/development of understanding.
4. **Not proven:** repetition may improve safety; current pilot cannot isolate HE
   wording.
5. **Minimal test:** HE-triggering frozen scenario, current appended instruction
   vs semantically identical deduplicated block, fixed selected card and schema.

### H6 — Production model/version parameters are insufficiently pinned

1. **Observed:** production uses alias and omits reasoning/output budget.
2. **Mechanism hypothesis:** server snapshot/default changes can alter contract
   reliability and behavior.
3. **Could explain:** non-reproducible failures across dates, not a specific
   echo pattern.
4. **Not proven:** no evidence current alias changed during observed runs.
5. **Minimal test:** first record actual production snapshot/usage in one
   authorized smoke; then matched alias vs fixed snapshot only if they differ.

### H7 — Fail-closed has no user-facing recovery

1. **Observed:** invalid outputs raise; retry/fallback is zero; pilot invalids
   are preserved.
2. **Mechanism hypothesis:** a recoverable model reference mistake terminates
   the conversation turn.
3. **Could explain:** runtime hard stops, not paraphrase.
4. **Not proven:** outer Telegram/global error handling was not audited here.
5. **Minimal test:** offline integration test injecting each recorded invalid
   payload and observing exact handler-visible outcome; no API required.

## 7. Prioritized Findings

### CRITICAL

1. **R14 namespace lifecycle is not operationally demonstrated, and schema
   cannot enforce core existing/local distinctions.** Actual pilot failures
   match this class. Test H1 first.

### HIGH

2. **R02/R09 incomplete prioritization among valid next actions.** Plausible
   contributor to local Reflection/questionnaire behavior; test H2.
3. **R07 no diverse full examples for a complex semantic contract.** Closely
   coupled to H1; test together rather than as an independent broad rewrite.
4. **R03/R19 history/trust labeling gap.** Roles are omitted and embedded text
   is not explicitly declared untrusted; test H4 and separately audit prompt
   injection offline.
5. **R20 no defined production recovery after valid API transport but invalid
   SemanticTurn.** Test H7 offline before proposing runtime change.

### MEDIUM

6. **R18 duplicated HE/base guards and uneven instruction emphasis.** Test H5
   only after contract validity is understood.
7. **R06 unbounded full HumanModel projection.** Measure real context sizes
   before treating this as a defect.
8. **R17 alias/default generation parameters.** Reproducibility risk; not a
   demonstrated cause of current consultation symptoms.
9. **R12 corrections/short answers lack demonstrations.** Can be covered in H1/
   H4 scenario set.

### LOW

10. **R08 negative instructions are not consistently paired with positive
    structured examples.** Address only as part of a focused example design,
    not by adding more prohibitions.

No confirmed finding warrants an architecture rewrite on present evidence.

## 8. Minimal Validation Experiments

Recommended order, without executing any now:

1. **Offline failure replay and schema exposure check (0 API).** Feed all four
   recorded invalid v3 payloads through current schema/adapter; preserve exact
   first failing boundary and verify which invalid values schema itself permits.
2. **Namespace example falsification pilot.** One-factor prompt supplement,
   S1/S3, current vs supplement, fixed snapshot/settings, 3 selections/cell.
   Primary metric: strict adapter validity; secondary quality remains blinded.
3. **Action-selection tie-breaker pilot.** Only after contract condition is
   stable. Freeze four semantic boundary scenarios and classify observable
   selected action, not wording.
4. **History-role matched test.** Correction/multi-target cases with and without
   explicit role in dynamic projection; same prompt/schema/model.
5. **HE duplication test.** Only HE-triggered cases with frozen card selection;
   current vs deduplicated HE instruction.
6. **Offline recovery-path test.** Inject invalid model content into production
   orchestration and record user-visible failure behavior without API.

Do not proceed to the full 72-call A/B/C/D run now. The cumulative report itself
states that strict pilot validity failed and full run remains blocked
(`FINAL_REPORT.md:189-210`).

## 9. Evidence and Limitations

### Primary evidence

- Rules R01–R20:
  `experiments/Prompt_Engineering_Rules_Research_2026-10-11.md:11-79`.
- Production prompt:
  `ai/prompts/cognitive_core_system_prompt.txt:1-257`.
- HE instruction and retrieval:
  `ai/human_experience.py:37-57,166-201,218-340`.
- Actual message/context assembly:
  `ai/cognitive_core.py:285-453`.
- API call and strict schema:
  `ai/cognitive_core.py:150-248`.
- Semantic schema/adapter:
  `ai/cognitive_semantic_wire.py:44-618`.
- Internal cross-field validator:
  `ai/cognitive_turn.py:374-412,438-553`.
- Actual assembly reconstruction and measured sizes:
  `COGNITIVE_CORE_PROMPT_ASSEMBLY_AUDIT.md:1-590`.
- Incomplete A/B/C/D evidence and accounting:
  `experiments/cognitive_core_ab/FINAL_REPORT.md:128-210,234-267`.

### Limitations

1. No new API call or LLM experiment was performed.
2. The A/B/C/D full experiment is incomplete; no condition is ranked.
3. Fresh pilot contains one sample per condition for S1/S3 and only 2/6 valid
   structured outputs. It proves failure existence, not frequency or cause.
4. Prompt compliance is not behavioral compliance. PASS means the rule is
   materially represented in current design, not that every model response
   will obey it.
5. FAIL/PARTIAL does not automatically imply the research document’s example
   implementation must be copied. Ratings reflect the rule’s intent and actual
   project evidence.
6. Current working tree was already dirty before this audit. Conclusions apply
   to the actual files read, not necessarily HEAD alone.

## Conclusion

The strongest established issue is not merely prompt length: it is the
operational gap between a complex semantic reference lifecycle and what strict
schema can enforce before the output reaches Python. The strongest behavioral
hypothesis is incomplete action-selection prioritization combined with a
structural incentive to reuse grounded user material. Both are testable; neither
is yet proven as the sole cause of mechanical paraphrase or weak consultation
progression.
