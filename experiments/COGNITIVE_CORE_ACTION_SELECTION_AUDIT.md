# Cognitive Core — Action Selection Audit

Дата аудита: 2026-10-12
Режим: read-only для production; без API-вызовов и новых экспериментов.

## 1. FACTS

### 1.1. Где фактически принимается решение

Текущий production-механизм не содержит отдельного Python-маршрутизатора,
который выбирает между вопросом, Reflection, Recognition и остановкой. Один
`CognitiveCore.propose()`:

1. собирает system prompt и JSON-контекст;
2. при включённом Human Experience добавляет отдельную системную инструкцию;
3. делает один structured-output вызов;
4. получает уже выбранный `decision_intent`, `selected_action` и
   `reply_segments`;
5. детерминированно переводит semantic handles в технические IDs и валидирует
   результат.

Это непосредственно видно в `ai/cognitive_core.py:150-197`,
`ai/cognitive_core.py:220-248` и `ai/cognitive_core.py:285-322`. Сам вход
содержит текущую реплику, Human Model, previous ACS, DialogueHistory и Human
Experience (`ai/cognitive_core.py:333-387`). Production prompt прямо говорит:
LLM решает **что** означает разговор и какое действие выбрано, Python только
технически оформляет и проверяет результат
(`ai/prompts/cognitive_core_system_prompt.txt:1-11`).

Следовательно, фактическая логика выбора — это совокупность качественных правил
prompt/source of truth, применяемых одной моделью. Сохраняемого списка
кандидатов, ranking, rationale или отдельного policy evaluator в production нет.
Это является намеренным selected-only boundary
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:559-601`). Поиск по
production-коду не обнаружил использования `SelectedActionEvaluator`; он
остаётся Conversation Lab/test tooling.

### 1.2. Фактические типы действий

В API-facing `SemanticTurn` нет отдельного типа `QUESTION` и нет отдельного
типа «профессиональная гипотеза». Фактические `DecisionIntent`:

- `HUMAN_DISCOVERY`;
- `MECHANISM_DISCOVERY`;
- `REFLECTION`;
- `RECOGNITION`;
- `TRANSITION`;
- `RESPECT_PAUSE_OR_REFUSAL`;
- `STOP_EXPLORATION`;
- `ANSWER_USER_QUESTION`.

Они определены в `ai/cognitive_turn.py:70-80`. Вопрос выражается не отдельным
intent, а `ResponseTarget` с interaction `OPEN_RESPONSE`, `CLARIFICATION`,
`EVALUATION` или `CONSENT` (`ai/prompts/cognitive_core_system_prompt.txt:169-184`).
Профессиональная гипотеза выражается как смысл текущего ActionContent — обычно
`SYSTEM_REFLECTION` или `RECOGNITION_OPTION` — и/или как system-proposed версия
для `EVALUATION`; отдельного intent «HYPOTHESIS» нет.

### 1.3. Условия и ограничения по каждому интересующему действию

#### REFLECTION

**Определение.** `REFLECTION` должна возвращать связную рабочую картину, которая
добавляет смысл, и требует `SYSTEM_REFLECTION` content
(`ai/prompts/cognitive_core_system_prompt.txt:153-155`,
`ai/cognitive_semantic_wire.py:240-243`). Source of truth определяет её как
соединение материала в целостную рабочую картину, а не повторение с вопросом
«правильно ли я понял?»
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:418-425`).

**Когда выбирать.** Она уместна, когда уже существует достаточно связанная
картина и возвращение связи, следствия, паттерна или нового понимания создаёт
ценность (`ai/prompts/cognitive_core_system_prompt.txt:36-47`;
`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:811-833`). После
коррекции Reflection также может показать, что содержательно изменилось, но
reconciliation/state update сами по себе не являются следующим действием
(`ai/prompts/cognitive_core_system_prompt.txt:52-58`).

**Ограничения.** Reflection не обязана создавать вопрос или ResponseTarget.
`EVALUATION` нужна только для genuinely new consequential hypothesis
(`ai/prompts/cognitive_core_system_prompt.txt:42-47`, `100-103`). Схема требует
наличие `SYSTEM_REFLECTION`, но не умеет проверять, добавляет ли текст смысл или
лишь перефразирует пользователя (`ai/cognitive_turn.py:520-530`).

#### RECOGNITION

**Определение.** `RECOGNITION` предлагает context-close working versions или
варианты и требует `RECOGNITION_OPTION`
(`ai/prompts/cognitive_core_system_prompt.txt:156-157`,
`ai/cognitive_semantic_wire.py:245-248`). Source of truth разрешает 1–3
контекстно близких проявления из Human Experience либо осторожную версию,
непосредственно обоснованную пользовательским материалом
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:383-398`).

**Когда выбирать.** Когда grounded гипотеза помогает человеку узнать возможный
механизм или проявление, оставаясь легко исправимой. User material, а не Human
Experience, должен давать grounding (`ai/prompts/cognitive_core_system_prompt.txt:48-69`).

**Ограничения.** Human Experience не evidence; версия не становится фактом,
не требует согласия и не должна возвращаться после rejection без нового
основания (`ai/prompts/cognitive_core_system_prompt.txt:104-111`;
`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:645-673`). Schema
проверяет только нужный content kind, а не мягкость, grounding или
correctability версии.

#### Уточняющий или исследующий вопрос

**Представление.** Отдельного `QUESTION` intent нет. Продолжение Human или
Mechanism Discovery выражается intent плюс ResponseTarget; для
`HUMAN_DISCOVERY` и `MECHANISM_DISCOVERY` target обязателен
(`ai/cognitive_semantic_wire.py:230-238`, `ai/cognitive_turn.py:549-553`).
`OPEN_RESPONSE` запрашивает открытый материал, `CLARIFICATION` — уточнение
адресованного смысла (`ai/prompts/cognitive_core_system_prompt.txt:169-173`).

**Когда выбирать.** Уточнение предпочтительно только при существенно разных
правдоподобных трактовках, если ответ способен materially изменить рабочее
понимание или следующий Decision. Снижение неопределённости системы не имеет
самостоятельной пользовательской ценности
(`ai/prompts/cognitive_core_system_prompt.txt:27-35`;
`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:498-505`, `523-548`).
Для отдельного эпизода проверка повторяемости допустима, но не обязательна
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:428-444`).

**Ограничения.** Один ход обычно содержит один focal request; несколько частей
допустимы лишь для одной когнитивной цели. Нельзя спрашивать уже отвеченное или
превращать разговор в анкету
(`ai/prompts/cognitive_core_system_prompt.txt:31-35`, `86-103`). Эти ограничения
семантические: JSON Schema не определяет, сколько независимых вопросов находится
в строке reply.

#### Профессиональная гипотеза о механизме

Это не отдельный action type. Human Experience поступает как optional input во
всё тот же LLM call (`ai/cognitive_core.py:293-320`). Карты названы
профессиональными working versions, не фактами; модель может их игнорировать и
должна использовать только при реальном grounding
(`ai/human_experience.py:37-57`). Prompt разрешает grounded версию именно для
meaningful understanding, а не для механического пересказа
(`ai/prompts/cognitive_core_system_prompt.txt:59-69`).

Гипотезу можно сформулировать без предварительного дополнительного вопроса:
source of truth прямо допускает мягкое, легко исправляемое предположение без
отдельной проверки при низкой цене ошибки
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:325-332`). Если версия
действительно новая и consequential, модель может создать `EVALUATION`; это не
обязательная церемония для каждой гипотезы.

#### STOP

Фактическое имя — `STOP_EXPLORATION`. Оно означает отсутствие нового
исследовательского вопроса, когда дальнейшее исследование уже не добавляет
значимого понимания
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:487-491`). Prompt
говорит, что intent завершает текущий Discovery и не создаёт ResponseTarget
(`ai/prompts/cognitive_core_system_prompt.txt:161-165`). Это также жёстко
проверяет contract (`ai/cognitive_semantic_wire.py:259-262`;
`ai/cognitive_turn.py:544-548`).

STOP не является автоматическим следствием определённого количества вопросов.
Он конкурирует с содержательной Reflection, Recognition, сменой области и
естественным продвижением, когда механизм уже достаточно grounded
(`ai/prompts/cognitive_core_system_prompt.txt:36-41`).

### 1.4. Как задаётся относительный приоритет

В source of truth базовый вопрос Decision: «Что сейчас является наиболее
полезным следующим действием?»; Value/Sufficiency входит в Decision
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:345-357`). Ни одна
линия не обязательна первой, и выбор должен зависеть от Value/Sufficiency
конкретной ситуации (`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:45-53`).

Наиболее операционное правило сейчас таково:

- если recurring mechanism, его значение и последствия уже grounded,
  уточнение имеет преимущество только при material delta;
- иначе выбрать meaningful Reflection, Recognition или уместное продвижение;
- если дальнейшее исследование не полезно, допустимы другая область, завершение
  исследования или Transition.

См. `Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:530-555` и
`ai/prompts/cognitive_core_system_prompt.txt:27-41`.

Это реальный, но **частичный** порядок. Он отсекает вопросы с нулевой
ожидаемой ценностью, однако не задаёт полного tie-breaker между Reflection,
Recognition, STOP и «другой уместной областью», когда несколько действий
формально допустимы. Такое отсутствие полного порядка согласуется с намерением
сохранить профессиональное суждение модели, но оставляет выбор вариативным.

### 1.5. Что предотвращает пересказ и лишние вопросы

В текущих инструкциях уже есть прямые guards:

- цель — полезное понимание, не полнота данных
  (`ai/prompts/cognitive_core_system_prompt.txt:16-30`);
- нельзя механически повторять ясное высказывание; вступление должно добавлять
  connection/consequence/pattern/insight (`:84-90`);
- нельзя превращать ход в анкету (`:91-92`);
- нельзя автоматически подтверждать Reflection или начинать каждый turn
  ритуальным «Понял/Спасибо» (`:96-103`);
- «не знаю» достаточно, а следующая область нужна лишь при существенной ценности
  (`:79-81`).

Соответствующие offline tests подтверждают присутствие этих правил в prompt,
например `tests/test_cognitive_core.py:517-600` и согласованность с source of
truth в `tests/test_cognitive_core.py:603-655`. Но это в основном prompt-contract
tests: они доказывают наличие текста, не фактический выбор модели. Frozen test
сам подчёркивает, что проверяет только доступность recurring-mechanism context,
а не live policy compliance (`tests/test_cognitive_core.py:326-327`).

### 1.6. Что проверяет схема, а что остаётся модели

`SemanticTurn` требует четыре части: state patch, reconciliation,
selected action и reply segments (`ai/cognitive_semantic_wire.py:285-289`).
Adapter детерминированно:

- проверяет существующие ссылки и типы handles;
- переводит handles в local IDs;
- проверяет compatibility reconciliation/interaction;
- связывает ActionContent, targets и `reply_segments.realizes`;
- конструирует валидный `CognitiveTurnResult`.

См. `ai/cognitive_semantic_wire.py:322-449`, `451-519`, `554-613`.
Внутренний contract дополнительно требует, чтобы targeted content был реализован
reply segment и чтобы Reflection/Recognition имели правильный content kind
(`ai/cognitive_turn.py:506-553`).

Однако schema/adapter **не могут установить**:

- является ли Reflection новым пониманием или красивым пересказом;
- нужен ли вопрос;
- действительно ли вопрос уже отвечен;
- содержит ли строка один или несколько независимых вопросов;
- соответствует ли `OPEN_RESPONSE` фактическому запросу в тексте;
- выражает ли reply весь заявленный `semantic_content`, а не только ссылается на
  его handle;
- было ли другое допустимое действие ценнее.

Это остаётся семантической ответственностью LLM и предметом human/live
evaluation, а не структурной валидации.

## 2. CONFLICTS / AMBIGUITIES

### 2.1. «Отразить» и «не пересказывать» одновременно

Reflection обязана вернуть coherent working picture, но запрещён механический
пересказ. Граница сформулирована содержательно («добавляет связь, следствие,
паттерн или новое понимание»), однако в output contract нет наблюдаемого
минимального semantic delta. Поэтому модель может назвать уже дословно
описанную пользователем последовательность «устойчивым циклом» и формально
считать это Reflection. Структурно это проходит schema.

Это не логическое противоречие policy, но operational ambiguity: одна и та же
реплика может быть оценена моделью и как новая интеграция, и как echo.

### 2.2. Продвижение после correction допускает несколько равно законных ходов

Prompt требует молча обновить картину и затем выбрать действие по value. Он
разрешает: перейти к другой значимой части, дать synthesis изменившегося или
остановить ветку (`ai/prompts/cognitive_core_system_prompt.txt:52-58`). Source of
truth формулирует то же самое
(`Research/Human_Experience/Weight/12_Cognitive_Cycle.txt:333-340`).

Тем самым явное правило по смыслу задачи существует: если после correction
дополнительная информация не нужна, можно принять её, обновить понимание и
закончить ветку. Но оно **не является обязательным correction-specific
tie-breaker**. Модель всё ещё может решить, что новая Human Discovery area имеет
ценность, и задать вопрос. Policy запрещает лишь вопросы с недостаточной
ценностью; оценка этой ценности не материализуется и не проверяется Python.

### 2.3. Human Experience одновременно приглашает к гипотезе и усиливает
осторожность

С одной стороны:

- HE может направлять Recognition (`12_Cognitive_Cycle.txt:92-108`);
- prompt просит использовать professional version для meaningful understanding,
  а не paraphrase (`cognitive_core_system_prompt.txt:62-69`);
- HE instruction разрешает grounded observation без обязательного вопроса
  (`ai/human_experience.py:42-55`).

С другой стороны:

- HE не evidence, нельзя усиливать причинность или скрытые мотивы;
- cards не scripts;
- Reflection lifecycle говорит, что Reflection не должна строиться из
  неподтверждённых элементов HE
  (`12_Cognitive_Cycle.txt:814-823`).

Это совместимо при аккуратном чтении: grounded low-stakes hypothesis можно
предложить как гипотезу, а не как факт. Но для модели остаётся неоднозначной
граница между «полезной grounded версией» и «неподтверждённым элементом HE».
Самый консервативный способ выполнить обе группы правил — повторить только
user-grounded материал. Это возможный механизм, а не доказанная причина выбора.

### 2.4. Target может быть структурно валиден без явного пользовательского
запроса в reply

Reflection может иметь optional target. Если модель добавляет `OPEN_RESPONSE`,
schema проверяет валидность его owner/subject, но не наличие фактического вопроса
или открытого приглашения в тексте. Поэтому ActionContent/target/reply могут быть
формально связаны через `realizes`, хотя interaction не проявлен в языке.
R04 — конкретный пример такого расхождения.

### 2.5. `STOP_EXPLORATION` семантически сильнее локального «ничего больше не
спрашивать сейчас»

Prompt определяет intent как завершение текущего Discovery, тогда как
correction-policy говорит «останови эту ветку». Reflection без target может
выполнять локальный stop, но отдельного intent для «принять correction и не
открывать следующий вопрос на этом ходу» нет. Архитектурно это не дефект:
Reflection без target уже выразима. Однако название STOP может делать её менее
естественным выбором при желании продолжить консультацию позднее. Это гипотеза
о восприятии model-facing taxonomy, не наблюдаемый факт.

### 2.6. Structural burden конкурирует с консультационным выбором

Модель в одном output одновременно обновляет Human Model, reconciles ACS,
выбирает action, объявляет handles/references/targets и пишет reply. A/B pilot
подтвердил, что даже structurally plausible output может ошибаться в типе
handle или compatibility target/content
(`experiments/semantic_reference_ab/FINAL_REPORT.md:53-65`). Он не доказывает,
что эта нагрузка вызывает paraphrase, но показывает, что action selection
происходит одновременно с существенной contract-задачей, а не в изоляции.

## 3. OBSERVATIONS

### 3.1. Ограничение сравнения A/B

Condition A — production assembly; B — тот же prompt плюс reference appendix с
таблицей и примерами `handle`/`existing_id`/`realizes`
(`experiments/semantic_reference_ab/FINAL_REPORT.md:25-34`). Appendix не меняет
action policy напрямую (`experiments/semantic_reference_ab/treatment_appendix.txt:1-26`),
но меняет длину и содержание prompt. Поэтому различия ниже — наблюдения о шести
saved outputs, а не доказательство причинности treatment или скрытого reasoning.
Модель не сохраняла rationale; prompt прямо запрещает его.

Mapping R-labels подтверждён в
`experiments/semantic_reference_ab/results/semantic-ref-ab-20261012T042853Z-9c26c00d/BLINDING_KEY.json:5-16`.

### 3.2. S1

#### R12 / A.S1.3 — содержательная гипотеза

Полный raw SemanticTurn сохранён в
`experiments/semantic_reference_ab/results/semantic-ref-ab-20261012T042853Z-9c26c00d/checkpoint.jsonl:26`.

- `decision_intent = reflection`.
- Primary `SYSTEM_REFLECTION` соединяет цепочку «рабочая усталость → пропуск
  обеда → переедание → “всё испортил” → прекращение попыток» и делает шаг к
  механизму: ощущение обнуления поддерживает цикл больше, чем сам эпизод еды.
- Additional `SYSTEM_REFLECTION` формулирует профессиональную версию «эпизод как
  приговор всей попытке», явно как возможное объяснение, не факт.
- `EVALUATION` target просит поддержать/частично поддержать/отвергнуть версию.
- Все три reply segment связаны с ActionContent через `realizes`.

Это пример того, что текущая архитектура **позволяет** использовать HE и перейти
от фактов к новой проверяемой версии без предварительного вопроса. Но ответ
нарушает отдельные style guards: начинается с ритуального «Спасибо» и публично
называет «рабочую гипотезу», хотя HE instruction просит не объявлять этот
технический ярлык. Кроме того, утверждение о большей роли эмоциональной реакции
остаётся системной гипотезой, а не установленной причинностью.

#### R04 / B.S1.1 — практически чистый пересказ

Raw SemanticTurn: тот же checkpoint, строка 5.

- `decision_intent = reflection`.
- Primary content называет последовательность «повторяющимся циклом» и внутри
  `semantic_content` упоминает нестабильность/«обнуление».
- Additional contents отсутствуют.
- Создан `OPEN_RESPONSE` target.
- Единственный reply segment почти дословно собирает уже полностью сообщённую
  цепочку и не содержит вопроса или открытого приглашения.

Структурная валидация прошла. Этот случай показывает две границы contract:
`realizes=[reflection_cycle]` доказывает только заявленную связь reply с content,
но не фактическую передачу всего semantic delta; валидный `OPEN_RESPONSE` не
доказывает наличие request в тексте. Policy guard против echo не был превращён
в структурный guard — и не может быть превращён без семантического анализа.

#### R09 / B.S1.2 — пересказ плюс уточняющий вопрос

Raw SemanticTurn: checkpoint, строка 14.

- `decision_intent = reflection`.
- Primary reflection повторяет полный цикл и добавляет, что он делает усилия
  прерывистыми и может усиливать чувство бесполезности.
- `OPEN_RESPONSE` спрашивает, что происходило в первые часы/на следующий день.
- Additional contents отсутствуют.

В отличие от R04, interaction проявлен в reply. Но USER уже сообщил ключевую
последовательность: после переедания думает «всё испортил» и бросает попытку на
недели. Из сохранённого output нельзя установить, почему модель считала детали
первых часов material для следующего Decision. Ответ иллюстрирует допустимую
форму Reflection+question, но не демонстрирует ожидаемый useful delta этого
вопроса. Он также близок к questionnaire continuation, хотя содержит один
связанный compound request, а не несколько полностью независимых тем.

### 3.3. S3

#### R03 / B.S3.2 — принятие correction без вопроса

Raw SemanticTurn: checkpoint, строка 23.

- `decision_intent = reflection`.
- `ordinary_addressed(previous_target_1)` корректно reconciles предыдущий
  response-seeking target.
- StatePatch сохраняет явную correction как current-user evidence.
- Reflection показывает изменение working picture: не длительная пауза, а
  возвращение к контролю.
- ResponseTargets отсутствуют.

Это фактический пример локального STOP без выбора `STOP_EXPLORATION`: Reflection
без target принимает correction, обновляет картину и не заставляет человека
отвечать снова. Ответ всё ещё начинается с ритуального «Понял» и заметно
пересказывает correction, но добавляет последствие для рабочей картины.

#### R06 / B.S3.1 — краткое принятие correction

Raw SemanticTurn: checkpoint, строка 8.

Структура почти та же: `reflection`, корректный `ordinary_addressed`, новый item,
нет target. Reply короче R03 и заканчивается выводом «это меняет понимание
паттерна». Это тоже подтверждает, что текущие schema и prompt уже позволяют
принять correction без нового вопроса. Ритуальное «Понял» остаётся нарушением
style guard, а смысловой прирост минимален, но ответ не создаёт questionnaire.

#### R11 / A.S3.1 — correction с дальнейшим анкетированием

Raw SemanticTurn: checkpoint, строка 11.

- Модель заявила `decision_intent = human_discovery`.
- `primary_content = null`, additional contents отсутствуют.
- `OPEN_RESPONSE` спрашивает, выражается ли контроль уменьшением порций,
  «лёгкой» едой или отслеживанием калорий.
- `ordinary_addressed(previous_target_1)` и StatePatch correction выбраны по
  смыслу верно.
- Но owner/subject target и `reply_segments.realizes` используют item handle как
  content handle.

Adapter отклонил output как wrong-type semantic handle; это не доставленный
валидный production turn (`experiments/semantic_reference_ab/FINAL_REPORT.md:53-58`).
Поэтому он является evidence о raw selected intent/text, но не о прошедшем весь
runtime ответе. Вопрос действительно выглядит questionnaire-like: он предлагает
три заранее заданных варианта детализации после correction, не объясняя, как
ответ materially изменит working understanding или Decision. Из raw data нельзя
узнать скрытое основание выбора.

### 3.4. Что доказывает совокупность наблюдений

1. Одинаковая production policy способна породить и содержательную grounded
   гипотезу (R12), и почти чистый echo (R04), и echo+question (R09).
2. После correction модель способна остановиться без вопроса (R03/R06), но может
   выбрать дальнейшее анкетирование (raw R11).
3. Это соответствует неполному qualitative ordering: инструкции запрещают
   low-value moves, но окончательная оценка value остаётся модели.
4. Эти шесть outputs не раскрывают hidden reasoning и не позволяют доказать,
   что именно prompt length, HE caution, schema burden или stochasticity вызвали
   конкретный выбор.

## 4. HUMAN EXPERIENCE В S1

### 4.1. Доступная профессиональная версия

Conservative retrieval поддерживает сейчас только P001/P006/P007
(`ai/human_experience.py:218-250`). S1 удовлетворяет P001: в тексте есть
переедание/срыв, «всё испортил/обнуление» и прекращение попыток
(`ai/human_experience.py:271-289`). Карта P001 в
`Research/Human_Experience/Weight/experimental/Human_Experience_Pain_Map_V2/human_experience_pain_map_v2.json:30`
содержит:

- professional hypothesis: один эпизод превращается в приговор всей попытке;
  боль не в калориях, а в ощущении обнуления труда;
- alternatives: усталость, жёсткие правила, действительный отказ от плана;
- guard: не выдавать за диагноз, скрытую истину или установленный факт.

Модели передаётся не весь card/script, а bounded view:
`professional_hypothesis`, `alternative_explanations`, `use_when`, `avoid`
(`ai/human_experience.py:323-330`). Human Experience включён по умолчанию с
emergency override и безопасно abstains при отсутствии/ошибке каталога
(`ai/human_experience.py:184-238`).

### 4.2. Разрешено ли использовать карту без ещё одного вопроса

Да. Production prompt требует использовать professional version, если она
grounded и создаёт meaningful understanding (`:62-69`). HE instruction разрешает
один grounded, легко исправляемый observation и прямо говорит, что полезная
Reflection/Recognition может закончиться без вопроса
(`ai/human_experience.py:45-55`). Source of truth также допускает low-cost мягкое
предположение без отдельной проверки (`12_Cognitive_Cycle.txt:325-332`). R12
практически демонстрирует эту возможность.

### 4.3. Является ли HE только материалом «учесть, но не использовать»

Нет прямого текста, который требовал бы лишь пассивного учёта. Напротив, prompt
просит применять доступную версию для meaningful understanding. Однако
осторожность сформулирована в нескольких местах и включает запрет строить
Reflection из неподтверждённого HE. Поэтому гипотеза о том, что модель иногда
выбирает безопасный пересказ вместо содержательной версии, **правдоподобна, но
не подтверждена**. R12 является прямым контрпримером сильной версии гипотезы
«модель вообще не может/не смеет использовать HE».

## 5. STOP ПОСЛЕ CORRECTION

Явное правило в дословной форме «после исправления всегда достаточно принять
его и остановиться» отсутствует — и такое глобальное правило противоречило бы
value-based policy. Но функциональный эквивалент для случая без дальнейшей
ценности есть:

- correction молча меняет working picture;
- reconciliation/update не определяют reply;
- нельзя расходовать turn на подтверждение/пересказ;
- можно показать содержательно изменившуюся связь или остановить исчерпанную
  ветку;
- уточнение допускается только при material delta.

См. `12_Cognitive_Cycle.txt:325-340`, `498-555` и
`cognitive_core_system_prompt.txt:27-58`.

Почему вопросы всё равно допустимы: prompt также разрешает перейти к ещё не
понятой значимой части истории или естественно продолжить разговор. Решение о
том, material ли новый вопрос, остаётся у модели; contract не требует rationale
и не имеет deterministic threshold. R03/R06 показывают, что no-question path
доступен, R11 — что raw model selection может оценить новую детализацию как
Human Discovery даже там, где human reviewer видит questionnaire continuation.

## 6. HYPOTHESES

### H1. Номинальный «паттерн» позволяет echo выглядеть как Reflection

**Наблюдаемый факт.** R04 структурно валиден: он объявляет reflection и называет
последовательность циклом, хотя user-facing текст почти полностью повторяет
USER.

**Предполагаемый механизм.** Качественное требование semantic delta не имеет
наблюдаемой границы в schema; маркировка «паттерн» может казаться достаточным
выполнением Reflection.

**Объясняемый симптом.** Механический пересказ вместо нового понимания.

**Не доказано.** Что модель именно так интерпретировала правило; rationale нет.

**Минимальная проверка.** Frozen matched S1: текущая policy против варианта с
одним узким contrastive rule/example «название уже явно описанной цепочки
паттерном само по себе не semantic delta»; несколько independent selections,
human blind rating semantic delta, отдельно structural validity.

### H2. Qualitative value rule недостаточно различает полезный вопрос и
questionnaire continuation

**Наблюдаемый факт.** R09 и raw R11 задают дальнейшие детали, хотя ключевой
механизм/коррекция уже сообщены. Prompt запрещает low-value clarification, но не
материализует expected delta.

**Предполагаемый механизм.** Формулировки «другая значимая область» и
«естественное продвижение» позволяют почти любой тематически связанный вопрос
считать потенциально полезным; полного tie-breaker нет.

**Объясняемый симптом.** Серия уточнений и questionnaire behavior.

**Не доказано.** Что ответы на эти вопросы действительно не изменили бы ни
понимание, ни следующий Decision; follow-up не проводился.

**Минимальная проверка.** Frozen S1/S3 comparison: до выбора вопроса требовать
только observable semantic contract «вопрос должен адресовать одну оставшуюся
неопределённость, разные ответы на которую ведут к различному следующему ходу»
без раскрытия rationale; human evaluator проверяет этот hinge по выбранному
action. Не менять schema на первом этапе.

### H3. Осторожность Human Experience делает paraphrase локально безопаснее

**Наблюдаемый факт.** Инструкции одновременно поощряют grounded hypothesis и
запрещают неподтверждённый HE/усиление причинности. R12 использует гипотезу, R04
и R09 почти не используют её.

**Предполагаемый механизм.** При сомнении модель минимизирует epistemic risk и
возвращает только user-stated chain.

**Объясняемый симптом.** HE присутствует во входе, но не создаёт нового
понимания.

**Не доказано.** Выбор мог быть обычной stochastic variation или следствием
reference appendix/общей нагрузки. R12 опровергает невозможность использования.

**Минимальная проверка.** Matched S1 with/without one clarified boundary:
«grounded, explicitly tentative, easily rejectable connection is allowed
without prior clarification; do not assert causality». Сохранять одну карту,
одинаковые inputs/schema/model; blind human rating grounding, novelty and
epistemic safety.

### H4. Одновременная contract-задача отнимает внимание у action quality

**Наблюдаемый факт.** В S3 control возникли две технические ошибки, одна —
wrong-type handle; B с reference examples прошёл 6/6. Model в том же response
должна решить консультационную и reference задачи.

**Предполагаемый механизм.** Высокая структурная нагрузка повышает вероятность
выбора простого текста/действия.

**Объясняемый симптом.** Короткие эхо-ответы или плохо связанные action/target.

**Не доказано.** Pilot был мал, B также увеличивал prompt, а R04/R09 как раз
принадлежат B. Техническая валидность не коррелирует однозначно с качеством.

**Минимальная проверка.** Уже подготовленный human blind review всех 12 outputs:
сравнить consultation delta с технической валидностью, не объявляя condition
winner. Если связи нет, не развивать эту гипотезу.

## 7. MINIMAL CHANGE CANDIDATES

Ниже — варианты для последующего обсуждения, не выбранное решение.

### Candidate 1 — узко определить минимальный semantic delta Reflection

**Изменение.** В source/prompt уточнить: назвать дословно уже описанную цепочку
«паттерном» недостаточно; Reflection должна добавить хотя бы одну обоснованную
связь, consequence, contrast или interpretation, которой ещё не было в ясной
USER-формулировке. Если добавить нечего, не маскировать echo как Reflection —
выбрать другой полезный action или закончить ход/ветку.

**Адресует.** R04 и частично R06/R09.

**Сохраняет.** Текущий intent, one-call и selected-only architecture, право на
естественную Reflection.

**Риск.** Модель может недооценить ценность краткого эмпатического synthesis или
излишне придумывать «новизну».

**Проверка.** Frozen S1 matched run + blind human rubric: semantic delta,
grounding, epistemic safety, naturalness; отдельно проверить отсутствие роста
unsupported hypotheses.

### Candidate 2 — correction-specific value tie-breaker без обязательного STOP

**Изменение.** После correction явно предпочитать: (a) короткую содержательную
фиксацию того, что изменилось, без target; либо (b) `STOP_EXPLORATION`/смену
ветки, если material unresolved question отсутствует. Новый вопрос допустим
только если его возможные ответы реально ведут к разным working pictures или
следующим actions. Это не правило «всегда остановиться после correction».

**Адресует.** Raw R11 и повторное анкетирование после уже ясной поправки.

**Сохраняет.** Reconciliation/state separation, возможность важного follow-up,
существующие intents/schema.

**Риск.** Преждевременное завершение, когда correction открывает новую важную
неопределённость.

**Проверка.** Frozen S3 с набором пар: correction, после которой вопрос не
нужен, и correction, где один discriminating question materially важен. Обе
категории должны оставаться достижимы.

### Candidate 3 — один contrastive action-selection example вместо новых правил

**Изменение.** Добавить компактный, schema-valid контраст на одном и том же
grounded контексте: плохой echo/questionnaire; хорошая tentative hypothesis или
meaningful Reflection; допустимый no-question stop. Не добавлять candidate list
в output и не менять architecture.

**Адресует.** Неоднозначность между формально допустимыми действиями и
неравномерное применение уже существующих правил.

**Сохраняет.** Текущую value policy и свободу суждения; не вводит quota или
жёсткий routing.

**Риск.** Prompt-length/anchoring effect, сценарное переобучение и новая
конкуренция с reference examples.

**Проверка.** Frozen A/B на S1 и S3, одинаковые schema/model/input/order;
основные human metrics — useful understanding, unnecessary question burden,
grounding. Structural validity считать отдельно, не подменять ей качество.

## 8. Итог аудита

Текущая policy уже содержит правильный базовый принцип: вопрос должен выигрывать
только при material useful delta; Reflection/Recognition/STOP полноценны; HE
можно использовать для grounded, correctable understanding без предварительного
опроса. Архитектура свободно выражает все эти варианты.

Основной установленный разрыв находится не в отсутствии action types, а между
качественной policy и структурной проверкой: модель сама оценивает semantic
delta и expected value, а schema проверяет лишь заявленную форму. Когда несколько
ходов допустимы, полного tie-breaker нет. Поэтому почти чистый пересказ может
быть оформлен как валидная Reflection, а тематически связанный вопрос — как
продолжение Discovery, даже если human reviewer считает его низкоценным.

Это объясняет **возможность** наблюдаемого поведения, но не устанавливает
скрытую причину каждого output. Сохранённые данные не содержат rationale;
малый A/B pilot и различие prompts не позволяют приписать выбор одному фактору.

`PRODUCTION CHANGES = 0`
`API CALLS = 0`
`AUDIT STATUS = COMPLETE`
