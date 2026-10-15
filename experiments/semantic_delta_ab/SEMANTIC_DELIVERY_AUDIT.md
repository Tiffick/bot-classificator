# Cognitive Core — Semantic Delivery Audit

Режим аудита: read-only для production. API-вызовы не выполнялись. Единственное
созданное изменение — этот отчёт.

## 1. Как механизм работает сейчас

### 1.1. Один модельный результат содержит и решение, и пользовательский текст

Production делает один structured-output вызов, получает JSON, парсит его и
передаёт в `adapt_semantic_turn`; отдельного Response Engine или второго шага,
который превращал бы внутреннее решение в ответ, нет
([`ai/cognitive_core.py:162-197`](../../ai/cognitive_core.py),
[`ai/cognitive_core.py:205-248`](../../ai/cognitive_core.py)). Модель в одном
`SemanticTurn` одновременно объявляет:

- `selected_action` — выбранный `DecisionIntent`, `primary_content`,
  `additional_contents` и `response_targets`;
- `reply_segments` — уже готовый пользовательский текст;
- `realizes` — заявленную моделью связь сегмента с текущим `ActionContent`.

Это прямо задано типом `SemanticTurn`
([`ai/cognitive_semantic_wire.py:218-289`](../../ai/cognitive_semantic_wire.py))
и инструкцией production prompt
([`ai/prompts/cognitive_core_system_prompt.txt:133-142`](../../ai/prompts/cognitive_core_system_prompt.txt)).

Следствие: `ActionContent.semantic_content` не является черновиком, из которого
Python строит reply. Это параллельное семантическое представление, а
`ReplySegment.text` — самостоятельный модельный вывод.

### 1.2. Что означает каждый объект

1. `SelectedAction` выбирает функцию следующего хода. Схема требует target хотя
   бы для `HUMAN_DISCOVERY` и `MECHANISM_DISCOVERY`; для `STOP_EXPLORATION`
   запрещает targets. Для `REFLECTION`, `RECOGNITION`, `TRANSITION`,
   `ANSWER_USER_QUESTION` и `RESPECT_PAUSE_OR_REFUSAL` базовый тип допускает как
   наличие, так и отсутствие targets
   ([`ai/cognitive_semantic_wire.py:224-277`](../../ai/cognitive_semantic_wire.py)).
2. `ActionContent` содержит `semantic_content` и ссылки на HumanModel
   ([`ai/cognitive_turn.py:340-350`](../../ai/cognitive_turn.py)).
3. `ResponseTarget` описывает ожидаемое взаимодействие пользователя и указывает
   owning content и subject
   ([`ai/cognitive_turn.py:353-412`](../../ai/cognitive_turn.py)).
4. `ReplySegment` содержит произвольный непустой текст и список content IDs,
   которые он объявляет реализованными
   ([`ai/cognitive_turn.py:415-429`](../../ai/cognitive_turn.py)).
5. Адаптер детерминированно назначает local IDs, разрешает handles и буквально
   переносит `semantic_content`, target interaction, reply text и `realizes` в
   `CognitiveTurnResult`
   ([`ai/cognitive_semantic_wire.py:335-390`](../../ai/cognitive_semantic_wire.py),
   [`ai/cognitive_semantic_wire.py:554-617`](../../ai/cognitive_semantic_wire.py)).

### 1.3. Что код действительно гарантирует

Код и Pydantic гарантируют:

- handle существует, уникален и имеет правильный тип;
- `existing_id` имеет требуемый persistent-ID prefix и существует во входном
  HumanModel;
- owner/subject target ссылаются на существующий current ActionContent;
- CONSENT совместим с `SYSTEM_PROPOSAL`/`SYSTEM_TRANSITION`;
- каждый ID из `realizes` ссылается на известный `ActionContent`;
- каждый **targeted** ActionContent упомянут хотя бы одним `ReplySegment.realizes`;
- обязательное соответствие отдельных intents и content kinds, а также
  наличие/отсутствие targets для Discovery и STOP.

Основные проверки находятся в
[`ai/cognitive_semantic_wire.py:362-390`](../../ai/cognitive_semantic_wire.py),
[`ai/cognitive_semantic_wire.py:563-617`](../../ai/cognitive_semantic_wire.py) и
[`ai/cognitive_turn.py:445-558`](../../ai/cognitive_turn.py).

### 1.4. Что существует только как инструкция модели

Только prompt, не код, требует:

- чтобы `ReplySegment.realizes` означал реальную передачу содержания, а не
  формальную ссылку
  ([`ai/prompts/cognitive_core_system_prompt.txt:255-257`](../../ai/prompts/cognitive_core_system_prompt.txt));
- чтобы intended user interactions были представлены в `response_targets`
  ([`ai/prompts/cognitive_core_system_prompt.txt:138-142`](../../ai/prompts/cognitive_core_system_prompt.txt));
- чтобы Reflection добавляла связь/следствие/паттерн и не превращалась в
  механический пересказ
  ([`ai/prompts/cognitive_core_system_prompt.txt:36-47`](../../ai/prompts/cognitive_core_system_prompt.txt),
  [`ai/prompts/cognitive_core_system_prompt.txt:84-103`](../../ai/prompts/cognitive_core_system_prompt.txt));
- чтобы correction не порождала анкетирование или meta-choice без полезной
  необходимости
  ([`ai/prompts/cognitive_core_system_prompt.txt:52-58`](../../ai/prompts/cognitive_core_system_prompt.txt)).

### 1.5. Чего контракт не обеспечивает

Ни JSON Schema, ни Python сейчас не проверяют:

- что reply передал **все существенные утверждения** `semantic_content`;
- что он не ослабил причинность, значение или полезный вывод;
- что любой вопрос в естественном тексте имеет `ResponseTarget`;
- что вопрос соответствует interaction/subject target;
- что сегмент не добавил новую просьбу или новую ветку, отсутствующую в
  `selected_action`;
- что нетargeted ActionContent вообще реализован в reply (проверяется только
  множество targeted contents, см.
  [`ai/cognitive_turn.py:506-518`](../../ai/cognitive_turn.py)).

Это не post-adapter потеря: адаптер не перефразирует текст. Разрыв создаётся
внутри одного модельного `SemanticTurn`, когда модель отдельно формирует
`semantic_content` и `text`, а затем сама заявляет `realizes`.

## 2. Подтверждённые проблемы

### 2.1. `realizes` проверяет ссылочную целостность, но не semantic delivery

`realizes` является tuple handles без описания покрываемых утверждений
([`ai/cognitive_semantic_wire.py:280-289`](../../ai/cognitive_semantic_wire.py)).
Адаптер проверяет только существование и тип handle, после чего переводит его в
local ID ([`ai/cognitive_semantic_wire.py:592-599`](../../ai/cognitive_semantic_wire.py)).
`CognitiveTurnResult` затем проверяет только, что ID известен и targeted content
формально реализован ([`ai/cognitive_turn.py:506-518`](../../ai/cognitive_turn.py)).

Поэтому R04 и R15 технически корректны даже при различии в информационной
плотности внутреннего и пользовательского представления.

### 2.2. Вопрос без ResponseTarget технически допустим для Reflection

У `_Reflection` нет ограничения `min_length=1` на `response_targets`
([`ai/cognitive_semantic_wire.py:240-243`](../../ai/cognitive_semantic_wire.py)).
`ReplySegment.text` — просто непустая строка; схема не знает, является ли она
вопросом ([`ai/cognitive_semantic_wire.py:280-289`](../../ai/cognitive_semantic_wire.py)).
Поэтому R06 и R20 проходят весь pipeline: текст содержит новый выбор/вопрос, но
`response_targets=[]`.

По текущему **исполняемому** контракту это допустимо. По заявленной архитектуре
и prompt это нарушение: intended interaction должен быть target, а Reflection
без target должна действительно завершаться без запроса ответа. Python не может
выявить нарушение без интерпретации русского текста; сейчас эта обязанность
полностью лежит на модели.

### 2.3. Persistent-ID prefix проверяется после JSON Schema

`ItemId` использует `AfterValidator(_validate_persistent_id(..., "mi"))`
([`ai/cognitive_semantic_wire.py:52-70`](../../ai/cognitive_semantic_wire.py)).
Сгенерированная JSON Schema представляет `_ExistingItem.existing_id` просто как
`string`, то есть prefix `mi_` в JSON Schema не выражен. Поэтому R03, R10 и R26
проходят JSON parsing и JSON Schema, но отклоняются при
`SemanticTurn.model_validate` в адаптере
([`ai/cognitive_semantic_wire.py:330-333`](../../ai/cognitive_semantic_wire.py)).

Prompt уже однозначно различает existing IDs и current handles
([`ai/prompts/cognitive_core_system_prompt.txt:201-214`](../../ai/prompts/cognitive_core_system_prompt.txt)).
Следовательно, это не отсутствие текстового правила; это повторяемая ошибка
исполнения правила, которую серверная strict schema не способна предотвратить в
текущем представлении.

## 3. Разбор указанных ответов

Blinding key данного run связывает IDs с samples следующим образом:
R04=`A.S1.2`, R15=`A.S1.3`, R06=`A.S3.2`, R20=`A.S3.1`,
R03=`A.S1.1`, R10=`B.S1.1`, R26=`B.S1.2`
([`results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDING_KEY.json`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDING_KEY.json)).

### 3.1. R04 — смысл ослаблен, но не потерян

Внутренний `semantic_content` формулирует две добавленные связи:

1. повторяющаяся цепочка делает попытки «устойчиво хрупкими»;
2. критична не единичная еда, а смысл «всё испорчено», запускающий прекращение
   попытки.

Reply сначала подробно перечисляет уже сообщённые три шага, затем всё же
передаёт второй вывод как вопрос: «ключевая часть ... не столько само
переедание, сколько ... “бросание” попытки?». Значит, существенный смысл не
исчез, но его delivery ослаблена:

- вывод утоплен в длинном echo;
- «структурная хрупкость попыток» не передана прямо;
- observation превращена преимущественно в запрос подтверждения.

Классификация: **смысл ослаблен**, а не полностью потерян. Технически sample
полностью валиден. Данные:
[`BLINDED_ACTION_CONTENT.json:130`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDED_ACTION_CONTENT.json).

### 3.2. R15 — delivery корректна, но выбранный insight имеет низкую новизну

Внутреннее содержание говорит: цепочка важнее единичных калорий, потому что
самооценочная реакция прекращает попытку на недели. Reply передаёт практически
тот же тезис: паттерн «объясняет, почему один эпизод часто заканчивает весь
прогресс», затем просит Evaluation.

Здесь нет доказанной semantic delivery loss. Ограничение возникает раньше:
пользователь уже прямо сообщил и мысль «всё испортил», и прекращение на недели,
поэтому Reflection в основном организует известную последовательность, а не
создаёт значительно новое понимание. Классификация: **смысл передан корректно,
но сам selected insight недостаточно полезен/нов**. Это проблема action/content
selection, не передачи через `reply_segments`. Данные:
[`BLINDED_ACTION_CONTENT.json:543`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDED_ACTION_CONTENT.json).

### 3.3. R06 — новый вопрос отсутствует во внутреннем действии

Внутреннее действие: `REFLECTION`, один `SYSTEM_REFLECTION`,
`response_targets=[]`. Reply после принятия correction спрашивает, связать ли
материал с усталостью/эмоциями или оставить фокус на реакции после эпизода.

Этот вопрос:

- создаёт новую развилку разговора;
- не представлен отдельным ActionContent;
- не имеет `OPEN_RESPONSE`, `CLARIFICATION` или другого target;
- тем не менее весь сегмент помечен как реализация Reflection.

Классификация: **response-composition defect и instruction-level contract
violation**, но не техническая невалидность текущей схемы. Pipeline принимает
его, потому что текст не анализируется семантически. Данные:
[`BLINDED_ACTION_CONTENT.json:200`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDED_ACTION_CONTENT.json).

### 3.4. R20 — тот же structural gap, усиленный meta-choice

Внутренне снова выбрана targetless `REFLECTION`. Reply спрашивает: продолжить
исследование факторов возвращения к привычкам или считать уточнение достаточным.
Это не просто риторический вопрос: он просит пользователя выбрать режим
консультации и новую область исследования. В `selected_action` такой выбор не
описан.

Классификация: **response-composition defect**, instruction-level нарушение
правил про internal criterion/meta-conversation и targets, но технически
допустимый `SemanticTurn`. Данные:
[`BLINDED_ACTION_CONTENT.json:735`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDED_ACTION_CONTENT.json).

### 3.5. R03 — local handles ошибочно объявлены existing IDs

State patch объявляет новые item handles `it_problem_pattern`,
`it_previous_attempts`, `it_emotional_response`. `ActionContent.model_item_refs`
затем использует те же строки как `existing_id`, а не как `handle`.

Первая реальная граница отказа: `SemanticTurn.model_validate`; `_ExistingItem`
требует persistent ID с prefix `mi_`. Adapter mapping и
`CognitiveTurnResult` не достигаются. JSON Schema это пропускает по причине из
раздела 2.3. Консультационное содержание не участвует в verdict. Данные:
[`BLINDED_ACTION_CONTENT.json:88`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDED_ACTION_CONTENT.json).

### 3.6. R10 — та же ошибка, независимо от более содержательного reply

Новые handles `it_1_skipped_lunch_binge_cycle`,
`it_2_emotional_response_guilt_ab`, `it_3_pattern_repetition` переданы как
`existing_id`. Первая граница отказа та же: Pydantic SemanticTurn validation.

Reply может быть консультационно полезным, но это не делает turn технически
валидным: production fail-closed корректно не допускает несогласованный graph.
Данные:
[`BLINDED_ACTION_CONTENT.json:338`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDED_ACTION_CONTENT.json).

### 3.7. R26 — тот же local-vs-existing defect

Новые handles `it_cycle_overeating_after_skip` и `it_pattern_frequency` также
помещены в `existing_id`. Отказ происходит до adapter и
`CognitiveTurnResult`. Два reply segments и их корректная ссылка на
`ref_cycle_meaning` не компенсируют ошибочные item references. Данные:
[`BLINDED_ACTION_CONTENT.json:948`](results/semantic-delta-ab-20261013T010202Z-8b1ed0be/BLINDED_ACTION_CONTENT.json).

## 4. Локализация проблем по уровням

| Уровень | Подтверждённые случаи | Что произошло |
|---|---|---|
| Выбор действия / содержания | R15; частично R04 | Выбрана валидная Reflection/Evaluation, но её добавленная ценность мала либо insight не выделен как главный пользовательский результат. |
| Формирование пользовательского ответа | R04, R06, R20 | R04 ослабляет внутренний вывод echo и confirmation burden; R06/R20 добавляют новые вопросы, отсутствующие в targets. |
| Техническая проверка | R03, R10, R26 | Новые local handles ошибочно оформлены как persistent `existing_id`; strict JSON Schema пропускает, Pydantic отклоняет. |
| Post-adapter transformation | ни один из семи | Адаптер не переписывает содержание и не добавляет вопросы; он детерминированно переносит модельный текст и разрешает ссылки. |

Техническая валидность и консультационное качество независимы: R03/R10/R26
могут содержать полезный текст и всё равно должны fail closed; R06/R20 полностью
валидны технически, хотя нарушают предполагаемую semantic ownership вопросов.

## 5. Возможные варианты исправления — без решения и реализации

### Вариант 1. Уточнить semantic-delivery contract и добавить targeted regression

**Идея.** Явно потребовать, чтобы reply сохранял consequential delta каждого
реализуемого ActionContent, а любой фактический запрос ответа был представлен
target-bearing content/ResponseTarget; закрепить это prompt-contract tests и
frozen live regression.

**Преимущества:** минимальное изменение, сохраняет one-call и естественную речь;
не переносит понимание русского текста в Python.

**Недостатки:** остаётся instruction-only гарантией; offline string/contract tests
не докажут live compliance.

**Риски:** слишком буквальная формулировка может усилить механический пересказ
`semantic_content` или заставить модель добавлять targets к риторическим фразам.

### Вариант 2. Сделать user-facing request отдельной явной семантической единицей

**Идея.** Рассмотреть более строгую связь между question/request content,
ResponseTarget и reply segment, чтобы модель не могла формально отнести новую
развилку к Reflection content. Это может быть узкое уточнение существующего
content/target contract, а не новый Engine.

**Преимущества:** лучше отделяет understanding от запроса следующего материала;
упрощает structural auditing.

**Недостатки:** schema всё равно не может определить по русскому тексту, что в
нём вопрос. Полная детерминированная гарантия потребовала бы либо ограничить
генерацию текста, либо добавить semantic judge — оба шага меняют нынешний
дизайн.

**Риски:** усложнение SemanticTurn, увеличение технических ошибок и возможная
неестественность составных conversational moves.

### Вариант 3. Закрыть local-vs-existing gap на API-facing schema boundary

**Идея A.** Выразить persistent-ID prefixes/формат в самой JSON Schema, а не
только `AfterValidator`, чтобы Structured Outputs не принимал `it_*` как
`existing_id`.

**Идея B.** По аналогии с previous target handles рассмотреть input-scoped
handles для existing HumanModel objects и детерминированное разрешение Python,
чтобы модель не копировала storage IDs.

**Преимущества A:** узко предотвращает точный класс R03/R10/R26 раньше и не
ослабляет fail-closed. **Преимущества B:** сильнее соответствует принципу
«LLM решает WHAT, Python оформляет HOW».

**Недостатки A:** устраняет лишь prefix-class ошибок; неверный, но существующий
`mi_` всё равно требует adapter validation. **Недостатки B:** заметнее меняет
API-facing contract и fixtures.

**Риски:** несовместимость frozen artifacts/tests; чрезмерно жёсткий regex может
отклонить легитимный будущий формат IDs. Выбор между A и B принадлежит владельцу
архитектуры.

## Evidence and limitations

- Аудит использовал текущие production prompt/code/schema и сохранённые raw
  outputs/validation events run
  `semantic-delta-ab-20261013T010202Z-8b1ed0be`.
- Причины technical failures воспроизводимы офлайн из сохранённого JSON.
- Почему модель внутренне выбрала конкретную формулировку, установить нельзя:
  rationale/chain-of-thought отсутствует по контракту.
- Переданные владельцем средние blind scores (A=3.1, B=2.3) приняты как внешний
  итог оценки; отдельного score sheet в package на момент аудита не найдено,
  поэтому этот отчёт не пересчитывает оценки и не выводит из них причинность.
- Семь разобранных samples доказывают наличие конкретных failure modes, но не их
  частоту вне данного run.

`PRODUCTION CHANGES = 0`
`API CALLS = 0`
`COMMIT = NO`
`PUSH = NO`
