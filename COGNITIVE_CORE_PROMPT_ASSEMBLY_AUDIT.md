# Cognitive Core — actual prompt assembly audit

Дата аудита: 2026-10-12
Режим: STRICT READ-ONLY для production; создан только этот отчёт.
Проверяемое состояние: текущий working tree поверх HEAD
`0d693909da4a2a85eff22a8e65aa87aa76ed8309`.

Статусы утверждений:

- **CONFIRMED** — прямо подтверждено текущим кодом или офлайн-воспроизведением.
- **LIKELY** — следует из косвенных свидетельств, но не наблюдалось на сетевом уровне.
- **UNKNOWN** — без live-трассировки или внешней среды установить нельзя.

## 1. Executive Summary

**CONFIRMED.** Обычный Telegram text turn проходит по цепочке:

`main.py` → `bot.handlers.message_handler()` → `run_dialog_engine()` →
`CognitiveCore.propose()` → `OpenAI.chat.completions.create()`.

На один обычный Cognitive Discovery turn выполняется **один** LLM-вызов.
Исключение — уже активный Telegram handoff: тогда `run_dialog_engine()` уходит в
детерминированный handoff path до создания `CognitiveCore` и делает ноль
Cognitive LLM-вызовов.

Модель получает ровно два chat message:

1. `system`: содержимое
   `ai/prompts/cognitive_core_system_prompt.txt`; когда Human Experience включён,
   к нему дописывается `HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION`.
2. `user`: один JSON-объект с текущей репликой, проекциями Human Model и
   previous ACS, отфильтрованной DialogueHistory и выбранными Human Experience
   cards либо явным abstention.

Отдельной роли `developer` нет. DialogueHistory не передаётся как серия
`user`/`assistant` сообщений OpenAI — она находится массивом внутри второго,
единственного `user` message. Source-of-truth Research документы напрямую не
читаются runtime и в запрос не включаются.

`SemanticTurn` задаётся двумя каналами одновременно:

- текстовые семантические инструкции находятся в system prompt;
- structural contract передаётся отдельно как strict JSON Schema через
  `response_format.type=json_schema`.

После ответа дополнительные cross-field и reference проверки выполняются
Python adapter/validators. Часть этих проверок описана в prompt, но не вся
выражается самой JSON Schema.

## 2. Production Call Path

### 2.1 Startup and Telegram entry

**CONFIRMED.** `main.py:7-17` загружает `.env`, создаёт Aiogram `Bot` и
`Dispatcher`, подключает router и запускает polling. `bot/handlers.py:49-82`
получает сообщение, загружает профиль и вызывает `run_dialog_engine(user_text,
profile, user_id, ...)`.

### 2.2 Discovery memory and current USER message

**CONFIRMED.** `ai/dialog_engine.py:53-57` загружает
`shadow_discovery_memory`, а при отсутствии создаёт пустые HumanModel, ACS=None
и DialogueHistory. В `ai/dialog_engine.py:240-250` память валидируется, после
чего текущая Telegram-реплика превращается в новый `DialogueMessage` роли
`USER`. Эта новая реплика ещё не добавлена в `dialogue_history`; она передаётся
в Cognitive Core отдельно.

### 2.3 One production Cognitive call

**CONFIRMED.** `ai/dialog_engine.py:252-260`:

- повторно вызывает `load_dotenv()`;
- создаёт `OpenAI(max_retries=0)`;
- создаёт `CognitiveCore` с его default model;
- один раз вызывает `core.propose(...)` через `asyncio.to_thread`.

`ai/cognitive_core.py:153-159` задаёт default model `gpt-5-mini`.
`ai/cognitive_core.py:178-197` собирает сообщения и вызывает:

```python
self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_turn",
            "strict": True,
            "schema": semantic_turn_json_schema(previous_acs),
        },
    },
)
```

Таким образом:

- API surface: OpenAI Chat Completions;
- method: `chat.completions.create`;
- requested model: `gpt-5-mini`;
- client retries: `0`;
- structured output: strict JSON Schema named `semantic_turn`;
- call-level `temperature`, `top_p`, `reasoning_effort`,
  `max_completion_tokens`, seed и tools **не передаются**;
- применяются SDK/API defaults для неуказанных параметров.

**UNKNOWN.** Фактический snapshot за alias `gpt-5-mini`, server-side defaults и
финальное HTTP-представление SDK нельзя подтвердить без запрещённого live API
или сетевой трассировки.

### 2.4 After the model

**CONFIRMED.** `ai/cognitive_core.py:205-248` извлекает content, парсит JSON и
передаёт payload в `adapt_semantic_turn`. После адаптации проверяются source
spans (`ai/cognitive_core.py:250-267`). DialogEngine затем соединяет
`reply_segments`, применяет `apply_cognitive_turn` и сохраняет новое Discovery
state (`ai/dialog_engine.py:266-289`). Эти post-response шаги не добавляют
новых LLM-вызовов.

## 3. Actual Request Structure

Финальный call-level объект имеет следующую подтверждённую структуру:

```json
{
  "model": "gpt-5-mini",
  "messages": [
    {"role": "system", "content": "<base prompt [+ HE instruction]>"},
    {"role": "user", "content": "<JSON string containing input view>"}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "semantic_turn",
      "strict": true,
      "schema": "<runtime SemanticTurn JSON Schema>"
    }
  }
}
```

Порядок `messages` фиксирован в `ai/cognitive_core.py:307-322`: сначала
`system`, затем `user`. Других сообщений нет.

System content собирается в `ai/cognitive_core.py:293-306`:

```text
base = exact UTF-8 contents of ai/prompts/cognitive_core_system_prompt.txt
if Human Experience enabled:
    system = base.rstrip() + "\n" + HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION + "\n"
else:
    system = base
```

User content — `json.dumps(_input_view(...), ensure_ascii=False)`, то есть JSON
строка без ASCII escaping русских символов (`ai/cognitive_core.py:309-320`).

## 4. Prompt Components Map

| Компонент | Источник | Статичность | Условие | Место/роль | Назначение |
|---|---|---|---|---|---|
| Base Cognitive Core instructions | `ai/prompts/cognitive_core_system_prompt.txt:1-257`, загрузка `ai/cognitive_core.py:43,324-331` | статический файл, но текущий working-tree content | всегда для Cognitive call | первое message, `system` | консультационная policy, prohibitions, semantic output rules |
| Human Experience system instruction | `ai/human_experience.py:37-57`, append `ai/cognitive_core.py:293-306` | статический текст | `human_experience_enabled()==True` | конец того же `system` message | статус cards как гипотез, guards и способ использования |
| Current USER message | `ai/cognitive_core.py:342-343,389-391` | динамический | всегда | JSON внутри второго `user` message | id, sequence, exact text |
| Human Model projection | `ai/cognitive_core.py:344-374` | динамический | всегда, возможно пустой | JSON `human_model` | items/relations и их epistemic metadata |
| Previous ACS projection | `ai/cognitive_core.py:375,398-426` | динамический | `null` либо предыдущее ACS | JSON `previous_active_conversation_state` | предыдущие active contents и unresolved targets |
| DialogueHistory projection | `ai/cognitive_core.py:376-381,428-453` | динамический и фильтрованный | всегда, возможно пустой | JSON `dialogue_history` | literal history closure, не отдельные chat messages |
| Human Experience selection | `ai/human_experience.py:166-181,218-260,323-340` | динамический | default enabled; emergency flag may disable | JSON `human_experience` | до 3 отобранных professional hypotheses или abstention |
| SemanticTurn JSON Schema | `ai/cognitive_semantic_wire.py:48-319` | schema from current Python models; conditional narrowing | всегда; при ACS=None две arrays получают maxItems=0 | отдельный `response_format` | structural output constraint |
| Model and generation parameters | `ai/cognitive_core.py:153-159,186-197`; client `ai/dialog_engine.py:252-254` | статические defaults текущего runtime | Cognitive call | call parameters | model alias, messages, strict schema; retry policy on client |

### Components that are not present

**CONFIRMED.** В production request отсутствуют:

- отдельное `developer` message;
- few-shot chat examples;
- отдельные historical chat messages OpenAI;
- `12_Cognitive_Cycle.txt` или другие Research documents как runtime text;
- legacy Engine outputs;
- hidden candidate list;
- второй LLM judge/evaluator.

Поиск production imports/references показывает, что Research Cognitive Cycle
не загружается runtime; активный static prompt path задаётся только
`ai/cognitive_core.py:43`.

## 5. Static vs Dynamic Content

### Static

1. **Base prompt** — 257 строк текущего
   `ai/prompts/cognitive_core_system_prompt.txt`:
   - lines 1-11: one-call role and LLM-WHAT/Python-HOW boundary;
   - lines 14-81: `A. HOW TO CONDUCT THE CONSULTATION`;
   - lines 84-130: `B. STRICT BEHAVIORAL PROHIBITIONS`;
   - lines 133-257: `C. SEMANTIC OUTPUT CONTRACT`.
2. **HE appended instruction** — точный block
   `ai/human_experience.py:39-56`.
3. **SemanticTurn model definitions** и enums, из которых Pydantic строит
   schema (`ai/cognitive_semantic_wire.py:48-289`).

### Dynamic

1. exact current USER text/id/sequence;
2. current HumanModel projection;
3. previous ACS projection or `null`;
4. filtered history closure;
5. HE selected cards/abstention;
6. schema narrowing based on presence of previous ACS.

Текущая `.env` не содержит `HUMAN_EXPERIENCE_ENABLED`, поэтому при данном
локальном startup применяется default enabled (`ai/human_experience.py:184-201`).
Значения `false/no/off/0` аварийно отключают слой; неизвестное значение
логируется и оставляет его включённым.

## 6. SemanticTurn Contract Exposure

### A. What is exposed in text instructions

**CONFIRMED.** Prompt section C (`ai/prompts/cognitive_core_system_prompt.txt:
133-257`) сообщает модели:

- четыре top-level поля;
- значения и смысл DecisionIntent;
- interaction/reconciliation semantics;
- distinction between persistent `existing_id` and current-turn `handle`;
- rules for ADD/REINFORCE/CORRECT;
- previous-target input handles;
- source grounding;
- state patch, materialization and reply realization invariants.

### B. What is exposed in JSON Schema

**CONFIRMED.** Schema строится из strict/frozen Pydantic models. `_Wire` имеет
`extra="forbid"` (`ai/cognitive_semantic_wire.py:48-49`), а
`_normalize_strict_schema` дополнительно делает все declared properties
required (`ai/cognitive_core.py:58-74`). Optional values представлены через
nullable branches, но ключи остаются обязательными.

Schema непосредственно задаёт:

- top-level object and required `state_patch`, `reconciliation`,
  `selected_action`, `reply_segments`;
- `additionalProperties=false` для wire objects;
- handle regex `^[A-Za-z][A-Za-z0-9_-]*$`, max 32;
- persistent item/relation ID prefixes and UUID validation;
- all enum values (`DecisionIntent`, content kinds, interactions, outcomes,
  operations, origins);
- discriminated object alternatives for item/relation changes,
  reconciliation, materialization and selected-action intents;
- required/min/max array constraints expressible by Pydantic;
- REFLECTION/RECOGNITION/TRANSITION primary-content types;
- minimum targets for Discovery intents and zero targets for STOP_EXPLORATION;
- non-empty reply segments.

When previous ACS is `None`, `semantic_turn_json_schema()` sets
`maxItems=0` for reconciliation and proposal materializations
(`ai/cognitive_semantic_wire.py:300-319`).

### C. What is only enforced after the response

**CONFIRMED.** Следующие правила нельзя считать известными модели только из
schema; они enforced adapter/internal validators после ответа (хотя многие
также описаны prose prompt):

- referenced `existing_id` actually exists in the supplied HumanModel
  (`ai/cognitive_semantic_wire.py:372-376`);
- handles are unique across the shared item/relation/content declaration
  namespace and resolve to the expected type (`337-370`);
- previous target handle exists (`443-449`);
- outcome is compatible with actual previous-target interaction and one target
  is not resolved twice (`451-518`);
- proposal materialization refers to a resolved, persistable target and valid
  previous active content (`521-552`);
- owner/subject/realizes handles resolve to declared current content
  (`554-599`);
- source quote occurs exactly once in current USER text, with only a narrow,
  unique case-insensitive fallback, and becomes coordinates (`ai/cognitive_core.py:
  77-147`);
- internal `CognitiveTurnResult` cross-field invariants, including target/content
  realization and intent/content constraints (`ai/cognitive_turn.py:374-412,
  438-553`).

Это не означает, что модель совсем не получает эти правила: значительная их
часть продублирована prose в section C. Но JSON Schema сама не кодирует все
referential/cross-object facts конкретного input.

## 7. Context Injection Map

### Current message

Передаётся полностью как `{id, sequence, text}`. Timestamp и role исключены;
роль следует из поля `current_user_message` и внешней chat role `user`.

### Human Model

Передаются все current `items.values()` и `relations.values()` без отдельного
лимита количества (`ai/cognitive_core.py:344-374`).

Item includes: `id`, `kind`, `content`, `provenance`, `status`, superseded IDs,
deduplicated source message IDs. Relation includes analogous metadata plus
source/target item IDs. Исключены timestamps, created/updated turn numbers и
source character offsets.

### Previous ACS

Если ACS отсутствует — `null`. Иначе передаются все active contents и response
targets (`ai/cognitive_core.py:398-426`). Persistent ACS ID, target `rt_` IDs,
created turn и source-message IDs не передаются. Каждому target детерминированно
назначается input-scoped handle `previous_target_1`, `previous_target_2`, ... по
iteration order mapping (`ai/cognitive_core.py:46-55`). Persistent active
content IDs и subject IDs остаются видимыми как контекст; для reconciliation
модель обязана использовать semantic target handle.

### DialogueHistory

Передаётся не полная история безусловно, а closure
(`ai/cognitive_core.py:428-453`):

1. последние 8 сообщений;
2. все source message IDs, на которые ссылаются HumanModel items/relations;
3. previous ACS source system message;
4. source message каждого previous active content.

Closure дедуплицируется по message ID и сортируется по sequence. Если required
message отсутствует, call fail-closed до API. Каждое history message содержит
только `{id, sequence, text}`; role и timestamp в projection отсутствуют.

### Human Experience

Default enabled. Retriever получает current text и **все** previous USER texts
из DialogueHistory (`ai/cognitive_core.py:295-303`), но внутри использует не
более последних 12 previous USER texts плюс current (`ai/human_experience.py:
240-241`).

В текущем pilot operationally участвуют только P001, P006, P007 через явные
predicates (`ai/human_experience.py:243-250,271-320`). Максимум 3 cards и 6000
JSON characters (`ai/human_experience.py:23-25,253-257,333-340`). Unknown or
malformed catalog causes safe abstention, не crash (`230-238`).

В модель для каждой selected card передаются только:

- `id`;
- `professional_hypothesis`;
- `alternative_explanations`;
- `use_when`;
- `avoid`.

Title, lens, recognition signals, ready-made recognition text, discriminating
question и inspiration sources исключены (`ai/human_experience.py:323-330`).

Даже при abstention system HE instruction всё равно appended, если layer
enabled; dynamic JSON сообщает `abstained=true`, reason и пустой cards list.

## 8. Instruction Duplication Inventory

Ниже только инвентаризация повторений, без оценки как defect.

| Тема | Base prompt | Appended HE instruction / schema |
|---|---|---|
| Hypothesis is not fact/evidence | prompt 48-69, 104-111 | HE 42-56 |
| Do not invent hidden motives/causes | prompt 59-69, 104-109 | HE 49-51 |
| Ground versions in USER material | prompt 48-51, 65-69 | HE 42-45 |
| Do not mechanically paraphrase | prompt 42-47, 86-103 | HE 45-54 |
| Correction/rejection remains authoritative | prompt 50-58, 79-81, 110-111 | HE 55-56 |
| Reflection/Recognition may not need question | prompt 36-51, 100-103 | HE 52-54 |
| Internal concepts must not be exposed | prompt 20-22, 93-95 | HE 45-50 |
| Update/state/reference rules | prompt section C 201-257 | schema structure plus post-response adapter checks |
| Choose one useful next action | prompt 27-41, 138-167 | selected-action union permits exactly one branch |
| Advice/Product Boundary | prompt 70-78, 112-127 | no equivalent HE text; schema does not semantically detect advice |

Base prompt также сам содержит positive rule и corresponding prohibition для
нескольких тем: consultation-value clarification (27-41 vs 89-103), internal
meta-language (20-22 vs 93-95), Product Boundary (70-78 vs 112-127). Это
зафиксированное смысловое повторение, не вывод об ошибочности.

## 9. Reconstructed Request

Офлайн-воспроизведение выполнено без API через реальные
`CognitiveCore._messages()` и `semantic_turn_json_schema()` на существующей
fixture `tests/test_cognitive_core.py:79-139`. Никакой самостоятельно
составленный substitute prompt не использовался. Environment flag отсутствовал,
поэтому HE был включён и безопасно abstained.

### Reproduction facts

- message roles: `system`, `user`;
- system content: 17,547 characters;
- system SHA-256:
  `30ED3AC456D77EAB8C2D100D961C57C02B6E3EF2109A97A2575F1D7740FBDBD0`;
- user JSON content: 2,642 characters;
- user content SHA-256:
  `C55D384E56B99A25335928A5645DB84660EAFF7AF2F0CE422A084B5D780BE44F`;
- compact schema: 18,231 characters;
- canonical schema SHA-256:
  `67561F316597BC45594F5EF9EE4136087ADFFE7F99553394448B876C950C5DCC`;
- canonical reconstructed call object SHA-256:
  `85AAE4BAEAC87437F705F872FB61246C6823F170F649A8B2D627DF477AEDC633`.

Synthetic UUIDs below were generated by the existing fixture; they are not
real user data.

### Exact dynamic USER payload used in the reconstruction

```json
{
  "current_user_message": {
    "id": "msg_ef9da50116fa4bc4948245bccb3a93b2",
    "sequence": 13,
    "text": "текущая реплика"
  },
  "human_model": {
    "items": [
      {
        "id": "mi_0628487eaf814703ad2b360fbeedf178",
        "kind": "experience",
        "content": "старый значимый опыт",
        "provenance": "user_provided",
        "status": "rejected",
        "supersedes_item_ids": [],
        "source_message_ids": ["msg_ec401514a59148d1a02bb016ea725ca2"]
      },
      {
        "id": "mi_b3a6397a78864057b3d87b4119576aae",
        "kind": "experience",
        "content": "скорректированный опыт",
        "provenance": "user_provided",
        "status": "corrected",
        "supersedes_item_ids": [],
        "source_message_ids": ["msg_ec401514a59148d1a02bb016ea725ca2"]
      }
    ],
    "relations": [
      {
        "id": "rel_398c3260bdb94d91855c34819a78883b",
        "meaning": "пользователь видит связь",
        "source_item_ids": ["mi_0628487eaf814703ad2b360fbeedf178"],
        "target_item_ids": ["mi_b3a6397a78864057b3d87b4119576aae"],
        "provenance": "user_interpretation",
        "status": "uncertain",
        "supersedes_relation_ids": [],
        "source_message_ids": ["msg_ec401514a59148d1a02bb016ea725ca2"]
      }
    ]
  },
  "previous_active_conversation_state": {
    "active_content": [
      {
        "id": "aci_cba94d8563374d3b90eba6cddd728c49",
        "kind": "system_proposal",
        "content": "предыдущая системная версия",
        "model_item_ids": ["mi_0628487eaf814703ad2b360fbeedf178"],
        "relation_ids": []
      }
    ],
    "response_targets": [
      {
        "handle": "previous_target_1",
        "active_content_id": "aci_cba94d8563374d3b90eba6cddd728c49",
        "subject": {
          "kind": "model_item",
          "id": "mi_0628487eaf814703ad2b360fbeedf178"
        },
        "interaction": "evaluation"
      }
    ]
  },
  "dialogue_history": [
    {"id":"msg_ec401514a59148d1a02bb016ea725ca2","sequence":1,"text":"старое user 1"},
    {"id":"msg_7d161e54a59c4424b5b5fe86e9c3d0a5","sequence":2,"text":"старое system 1"},
    {"id":"msg_f538abd59fa04a4f88c307104d574da7","sequence":5,"text":"старое user 3"},
    {"id":"msg_88e2afb49549405aac184e51fa72b1fb","sequence":6,"text":"старое system 3"},
    {"id":"msg_eea534fdff404f698f33329b423c08bf","sequence":7,"text":"старое user 4"},
    {"id":"msg_e2aecff84fd4493d945ce204ebbc12f2","sequence":8,"text":"старое system 4"},
    {"id":"msg_092d0e87dd5a49388b26a7fb7a54b4ba","sequence":9,"text":"старое user 5"},
    {"id":"msg_e50c7b58bd264f6b9e037183b29650d1","sequence":10,"text":"старое system 5"},
    {"id":"msg_9265823828ed497e85b216ba1d859877","sequence":11,"text":"старое user 6"},
    {"id":"msg_ebf355be33664e14be8e5c2ca4d4a909","sequence":12,"text":"старое system 6"}
  ],
  "human_experience": {
    "enabled": true,
    "status": "experimental_hypotheses_not_evidence",
    "retrieval_method": "conservative_explicit_scenarios_v1",
    "abstained": true,
    "abstain_reason": "no_reliable_match",
    "selected_cards": []
  }
}
```

### Schema actually supplied

The schema is the exact return value of
`semantic_turn_json_schema(previous_acs)` and is passed under
`response_format.json_schema.schema`; it is not inserted into either message.
For this fixture previous ACS exists, so the non-narrowed schema is used.

Top-level contract:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "state_patch",
    "reconciliation",
    "selected_action",
    "reply_segments"
  ],
  "properties": {
    "state_patch": {"$ref": "#/$defs/_StatePatch"},
    "reconciliation": {
      "type": "array",
      "items": {
        "anyOf": [
          {"$ref": "#/$defs/_OrdinaryAddressedResolution"},
          {"$ref": "#/$defs/_OutcomeResolution"}
        ]
      }
    },
    "selected_action": {
      "anyOf": [
        {"$ref":"#/$defs/_HumanDiscovery"},
        {"$ref":"#/$defs/_MechanismDiscovery"},
        {"$ref":"#/$defs/_Reflection"},
        {"$ref":"#/$defs/_Recognition"},
        {"$ref":"#/$defs/_Transition"},
        {"$ref":"#/$defs/_RespectPause"},
        {"$ref":"#/$defs/_StopExploration"},
        {"$ref":"#/$defs/_AnswerQuestion"}
      ]
    },
    "reply_segments": {
      "type":"array",
      "minItems":1,
      "items":{"$ref":"#/$defs/_ReplySegment"}
    }
  }
}
```

The complete 18,231-character schema additionally contains these 37 exact
definitions generated from current code:

`ActiveContentKind`, `EvidenceOrigin`, `ModelItemKind`,
`ReconciliationOutcome`, `TargetInteractionKind`, `_AddItem`, `_AddRelation`,
`_AnswerQuestion`, `_Content`, `_ContentSubject`, `_CorrectItem`,
`_CorrectRelation`, `_ExistingItem`, `_ExistingRelation`, `_HumanDiscovery`,
`_ItemMaterialization`, `_ItemSubject`, `_LocalItem`, `_LocalRelation`,
`_MechanismDiscovery`, `_OrdinaryAddressedResolution`, `_OutcomeResolution`,
`_Recognition`, `_RecognitionContent`, `_Reflection`, `_ReflectionContent`,
`_ReinforceItem`, `_ReinforceRelation`, `_RelationMaterialization`,
`_RelationSubject`, `_ReplySegment`, `_RespectPause`, `_StatePatch`,
`_StopExploration`, `_Target`, `_Transition`, `_TransitionContent`.

Its canonical SHA above makes the reconstructed schema identity independently
checkable without duplicating a large generated artifact in this report.

## 10. Evidence

Primary evidence locations:

- startup: `main.py:1-21`;
- Telegram entry and runtime call: `bot/handlers.py:49-82`;
- memory loading and current message: `ai/dialog_engine.py:45-67,222-260`;
- one API call and parameters: `ai/cognitive_core.py:150-197`;
- message assembly: `ai/cognitive_core.py:285-387`;
- ACS handles/history closure: `ai/cognitive_core.py:398-453`;
- base production prompt: `ai/prompts/cognitive_core_system_prompt.txt:1-257`;
- HE instruction/default/retrieval/projection: `ai/human_experience.py:21-57,
  166-201,218-260,271-340`;
- SemanticTurn wire objects/schema: `ai/cognitive_semantic_wire.py:44-319`;
- adapter-only reference validation: `ai/cognitive_semantic_wire.py:322-618`;
- internal cross-field validation: `ai/cognitive_turn.py:374-412,438-553`;
- reconstruction fixture: `tests/test_cognitive_core.py:79-139`.

## 11. Unknowns

1. **UNKNOWN:** actual model snapshot selected server-side for `gpt-5-mini`.
2. **UNKNOWN:** exact HTTP body normalization performed by the installed OpenAI
   SDK after receiving the Python kwargs; audit establishes the kwargs, not a
   network capture.
3. **UNKNOWN:** deployment-specific environment may set
   `HUMAN_EXPERIENCE_ENABLED=false`; current local `.env` does not. The code
   behavior for both paths is confirmed.
4. **UNKNOWN:** server-side behavior for omitted generation parameters. They
   are demonstrably not set by project code.
5. **CONFIRMED LIMIT:** the reconstruction captures the exact current builder
   on synthetic existing test data. It does not expose any real user data and
   does not prove what a live model would return.

## Completion statement

The production model receives, in order: (1) the current base system prompt,
optionally extended with the enabled Human Experience instruction; (2) one
user-role JSON message containing current USER material plus filtered state;
and (3) a strict SemanticTurn JSON Schema as a separate `response_format`
parameter. This composition is produced by the code paths and line references
documented above. No API request was required to establish it.
