# Cognitive Core: аудит профессиональных знаний и консультационной методологии

Дата аудита: 2026-10-13
Режим: исследование и проектирование; production не изменялся; API-вызовы не выполнялись.

## 1. Краткий вывод

Cognitive Core сейчас является сильнее спроектированным **контуром ведения
диалога и фиксации индивидуального понимания**, чем системой профессиональной
консультации по снижению веса. Он умеет хранить факты, интерпретации и
отвергнутые версии, выбирать следующий ход, создавать ResponseTarget и
применять результат детерминированно. Однако фактический production-вход почти
не содержит предметной базы знаний о физиологических, поведенческих,
контекстных и клинически значимых механизмах набора веса и неудачных попыток.

Доступное модели профессиональное содержание ограничено:

1. общей консультационной политикой в system prompt;
2. полным Human Model и предыдущим ACS;
3. фрагментом DialogueHistory;
4. максимум тремя Human Experience-картами, причём operational retrieval
   существует только для `P001`, `P006`, `P007`.

Методология из `Diagnostic_Consultation_Methodology_AI_Lead_Bot` значительно
богаче: она требует конкурирующих рабочих объяснений, различающих наблюдений,
многофакторной картины, корректной работы с медицинскими и психологическими
red flags и перехода от сбора фактов к проверяемому улучшению понимания. Но это
сейчас **документ**, а не runtime Knowledge Base.

Главный ответ на вопрос аудита: чтобы профессиональное знание действительно
применялось, недостаточно ещё одного prompt-правила. Нужен отдельный от Human
Experience, небольшой и проверяемый **Professional Knowledge Layer**,
поставляющий Cognitive Core релевантные механизмы, альтернативы, различающие
наблюдения, ограничения применимости и red flags. Он должен оставаться входом
для Formulation, а не источником фактов о пользователе. Существующие one-call,
selected-only, Human Model, ACS и deterministic adapter следует сохранить.

## 2. Карта источников знаний

| Источник | Что содержит | Фактический статус | Доходит до production-модели |
|---|---|---|---|
| `Consultation_Methodology/Diagnostic_Consultation_Methodology_AI_Lead_Bot.docx` и `.pdf` | Синтез MI/OARS, функционального анализа, CBT formulation, COM-B, Elicit–Provide–Elicit, 5As, осторожного SPIN; механизмов и red flags | Новый методологический документ; DOCX и PDF — две формы одного содержания | Нет |
| `Research/Human_Experience/Weight/12_Cognitive_Cycle.txt` | Authoritative цикл Context → Formulation → Decision → Action → Response → Evaluation; value/sufficiency; Recognition/Reflection/Transition | Активный source of truth | Косвенно: вручную сжат в production prompt, не загружается как файл |
| `Research/Human_Experience/Weight/01…11, 13…17` | Опыт пользователя, boundaries, transition, HM/ACS/memory contracts | Source-of-truth семейство | Только через реализованные prompt/contract/code положения |
| `Research/.../experimental/Human_Experience_Pain_Map_V2/human_experience_pain_map_v2.json` | 30 авторских карт возможного внутреннего опыта | Сам каталог помечен `DESIGN_ONLY_NOT_PRODUCTION` | Частично: только `P001/P006/P007` через `ai/human_experience.py` |
| `ai/human_experience.py` | Валидация 30 карт, безопасная проекция, эвристический retrieval | Активен по умолчанию; emergency override через env | Да, максимум три selected cards |
| `ai/prompts/cognitive_core_system_prompt.txt` | Консультационная политика и SemanticTurn contract | Активный system prompt | Да полностью |
| `HumanModel` | Индивидуальные факты, значения, попытки, механизмы, связи, provenance/status | Активное persistent state | Да полностью в динамическом JSON |
| `ActiveConversationState` | Выбранное действие, активное содержание, ожидания ответа | Активное transient/persistent state | Да через input-scoped представление |
| `DialogueHistory` | Релевантная история разговора | Активен; closure последних сообщений и evidence refs | Да, отфильтрованный фрагмент |
| `ai/prompts/knowledge_base.json` | Старые сегменты по килограммам, типовые чувства/симптомы/влияния | Legacy | Нет: загружается legacy `ResponseEngine`, отсутствующим в active path |
| `ai/prompts/human_experience_layer.json` | Старые canned micro-moments/hope fragments | Legacy | Нет в active Cognitive Core path |
| `docs/*` | Раннее продуктовое видение и прежняя архитектура | История/vision, не current source of truth | Нет |

Проверенные контрольные суммы методологии:

- DOCX: `3ac17dfa8e86d3659c787730b0171dd95ca154392c9a15012a75aee0b2b8469c`;
- PDF: `048809c20869a6cc7c30909644e326a2a4e6075a9895c864f2786d050d7e3889`.

DOCX и PDF не следует считать двумя независимыми источниками доказательств.
Это один авторский синтез в двух форматах. Сам документ корректно оговаривает,
что он не является валидированным клиническим протоколом, а иллюстрации не
доказывают эффективность конкретной формулировки бота.

## 3. Разделение четырёх функций

### 3.1. Consultation Methodology — как консультировать

Это правила процесса: как строить formulation, когда уточнять, когда дать
working version, как работать с несогласием, как не превращать беседу в анкету
и где заканчивается Discovery. Они представлены в Cognitive Cycle и новом
методологическом документе. В production prompt уже есть значительная часть
этого слоя: positive consultation policy, prohibitions и Product Boundary
([`ai/prompts/cognitive_core_system_prompt.txt:14-130`](../ai/prompts/cognitive_core_system_prompt.txt)).

### 3.2. Professional Knowledge Base — что профессионально известно

Это проверяемые доменные знания: возможные механизмы, условия применимости,
альтернативные объяснения, различающие наблюдения, противопоказания/red flags,
границы уверенности и источники. Такого активного слоя в текущем Cognitive Core
нет. Методология описывает содержание, из которого его можно спроектировать,
но документ не является исполняемой базой.

Старый `knowledge_base.json` не подходит: он сегментирует человека по числу
набранных килограммов и привязывает к диапазонам типовые чувства, симптомы и
социальные последствия. Это грубое и потенциально недостоверное выведение
индивидуальной жизни из веса; активировать его нельзя.

### 3.3. Human Experience — как механизм может переживаться

Human Experience — карта возможных узнаваемых проявлений и смыслов, а не
клиническая или причинная база. Source of truth явно говорит, что он не является
доказательством о пользователе
([`12_Cognitive_Cycle.txt:79-131`](../Research/Human_Experience/Weight/12_Cognitive_Cycle.txt)).
Runtime-инструкция также называет карты optional working versions, запрещает
скрытые мотивы и требует оставить отвергнутую карту
([`ai/human_experience.py:37-57`](../ai/human_experience.py)).

Каталог содержит 30 карт, но selector реально рассматривает только `P001`,
`P006`, `P007` ([`ai/human_experience.py:243-250`](../ai/human_experience.py)).
Поэтому фраза «Human Experience интегрирован» технически верна, но создаёт
ложное впечатление полноты. Остальные 27 карт неоперациональны.

### 3.4. Individual Human Model — что известно именно об этом человеке

Human Model — индивидуальная память с provenance и epistemic status. Модель
различает `USER_PROVIDED`, `USER_INTERPRETATION`, `SYSTEM_PROPOSED` и статусы
поддержки/отвержения/неопределённости
([`ai/discovery_data_model.py:75-106`](../ai/discovery_data_model.py)).
Это правильная граница: Knowledge Base и Human Experience предлагают варианты,
а Human Model хранит только то, что имеет конкретное основание в диалоге.

### 3.5. Что сейчас смешивается

- Prompt одновременно выступает методологией, policy, contract manual и
  частичным surrogate доменной экспертизы.
- Human Experience называется «professional working versions», хотя его карты
  преимущественно описывают возможный внутренний опыт, а не полный механизм.
- Новый методологический документ совмещает метод процесса и обзор предметных
  механизмов; для runtime эти части нужно разделить.
- Legacy Knowledge Base смешивает общую статистическую типичность и утверждения
  о конкретном пользователе.

## 4. Что модель фактически получает

Production path создаёт ровно один `CognitiveCore.propose()` с клиентом
`OpenAI(max_retries=0)` ([`ai/dialog_engine.py:242-259`](../ai/dialog_engine.py)).
Вызов использует strict JSON Schema
([`ai/cognitive_core.py:165-232`](../ai/cognitive_core.py)).

System input состоит из:

1. полного production prompt;
2. `HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION`, если HE включён
   ([`ai/cognitive_core.py:289-316`](../ai/cognitive_core.py));
3. JSON-контекста с текущей USER-репликой;
4. полной проекции Human Model items/relations;
5. предыдущего ACS с semantic handles;
6. history closure: последние восемь сообщений плюс сообщения, на которые
   ссылаются HM/ACS;
7. максимум трёх выбранных HE-карт.

Из HE-карты передаются только `id`, `professional_hypothesis`,
`alternative_explanations`, `use_when`, `avoid`
([`ai/human_experience.py:323-330`](../ai/human_experience.py)). Это разумная
компактная проекция, но в неё не входят evidence/source, red flags,
discriminating observation и уровень уверенности.

Модель **не получает**:

- новый методологический DOCX/PDF;
- полный Cognitive Cycle;
- клинические рекомендации и safety routes;
- структурированный каталог механизмов;
- 27 неоперациональных HE-карт;
- legacy Knowledge Base в active runtime.

Следовательно, модель вынуждена заполнять предметные пробелы собственным
pretraining либо выбирать более безопасный формально допустимый ход — пересказ
или уточнение. Это правдоподобный механизм, но причинность нельзя считать
доказанной без matched knowledge-availability experiment.

## 5. Что уже работает и не требует замены

1. **One-call и selected-only architecture.** Один модельный результат содержит
   понимание, state patch, выбранное действие и reply; Python не выбирает смысл.
2. **Разделение LLM-WHAT / Python-HOW.** Input-scoped handles разрешаются
   детерминированно; неизвестные или wrong-type handles fail closed
   ([`ai/cognitive_semantic_wire.py:337-391`](../ai/cognitive_semantic_wire.py)).
3. **Индивидуальная эпистемика.** `SYSTEM_PROPOSED` не становится фактом только
   потому, что модель его сформулировала; materialization зависит от ответа
   пользователя.
4. **Value-based Decision.** Cognitive Cycle требует выбирать ожидаемый полезный
   delta, а не снижать неопределённость ради системы
   ([`12_Cognitive_Cycle.txt:346-397`](../Research/Human_Experience/Weight/12_Cognitive_Cycle.txt)).
5. **Correctability.** Recognition/Reflection можно поддержать, частично
   поддержать, отвергнуть или оставить uncertain; отвергнутая версия не должна
   возвращаться без нового основания.
6. **Product Boundary и consent.** Discovery заканчивается до проектирования
   персонального решения; Transition требует meaningful consent
   ([`12_Cognitive_Cycle.txt:875-889`](../Research/Human_Experience/Weight/12_Cognitive_Cycle.txt)).
7. **Safe abstention HE.** Ошибочный/отсутствующий каталог не должен ронять бота;
   selector возвращает пустой набор и пишет диагностику.
8. **Fail-closed contract.** Технически неконсистентный SemanticTurn не
   записывается частично.

## 6. Что не работает или работает неполно

### 6.1. Нет operational Professional Knowledge Base — CRITICAL

Prompt требует содержательного Recognition и mechanism discovery, но не
поставляет систематизированных механизмов. Три HE-карты не заменяют знания о
сне, медикаментах, эндокринных факторах, доступности пищи, экономике, рабочих
условиях, семейной среде, расстройствах пищевого поведения и рисках.

### 6.2. Human Experience имеет узкое и эвристическое покрытие — HIGH

Retrieval трёх карт основан на regex/lexical predicates и последних USER
текстах. Это не делает Python semantic interpreter, но даёт узкое покрытие и
не измеренную вероятность ложного/пропущенного попадания. Каталог одновременно
помечен `DESIGN_ONLY_NOT_PRODUCTION` ([`ai/human_experience.py:147-155`](../ai/human_experience.py)),
README говорит «не включено в production», а код включает его по умолчанию
([`ai/human_experience.py:184-201`](../ai/human_experience.py)). Это статусная и
документационная противоречивость.

### 6.3. Consultation Methodology живёт вне runtime — HIGH

Принцип «2–3 конкурирующие гипотезы + различающее наблюдение или осторожная
версия» отсутствует как поставляемая модели структура. Есть общая политика
expected useful delta, но нет профессиональных alternatives/contraindications,
на которых она могла бы работать.

### 6.4. Один output перегружен независимыми обязанностями — HIGH

Модель одновременно должна выбрать действие, обновить граф, правильно адресовать
existing/local объекты, reconciliate ACS, создать ActionContent/targets и написать
естественный reply. Это не означает, что one-call ошибочен, но создаёт конкуренцию
между консультационной содержательностью и структурной корректностью.

### 6.5. Semantic delivery проверяется структурно, не содержательно — HIGH

`ReplySegment.realizes` гарантирует существование ссылки, но не то, что reply
действительно передал важный вывод. Сохранённый audit показывает ослабление
смысла в R04 и вопросы в R06/R20 при пустых ResponseTargets
([`SEMANTIC_DELIVERY_AUDIT.md:107-224`](../experiments/semantic_delta_ab/SEMANTIC_DELIVERY_AUDIT.md)).

### 6.6. Safety knowledge неполно операционализировано — HIGH

Product Boundary запрещает диагноз и решение, но prompt не даёт явного
операционного маршрута для eating-disorder, self-harm, резких необъяснимых
изменений веса, лекарственных/эндокринных рисков. Новый документ эти red flags
содержит; runtime — нет. Нельзя отправлять такие случаи обычному sales/weight
consultant handoff как эквивалент медицинской помощи.

### 6.7. Старое vision конфликтует с новой эпистемикой — MEDIUM

`docs/00_PROJECT MANIFESTO.txt:348` обещает «рассказал обо мне больше, чем я сам
успел рассказать». В сочетании со старой идеей вероятностно выводить несказанное
это опасно близко к внушению. Текущий Cognitive Cycle и boundaries правильнее:
Human Experience — источник проверяемых версий, не фактов. Старые docs следует
явно маркировать историческими, а не пытаться совместить буквально.

### 6.8. Legacy Knowledge Base содержит предметные ошибки/необоснованные выводы — HIGH

Категоризация чувств и последствий по диапазону набранных килограммов не имеет
достаточного индивидуального основания. Она может создавать стыд, ложные
причины и медицинские утверждения. Этот файл правильно отсутствует в active
path и не должен становиться основой нового слоя.

## 7. Подтверждённые причины

### C1. Production не получает методологический документ

**Факт:** assembly состоит из prompt, HE instruction и динамического JSON;
методологические файлы не загружаются (`ai/cognitive_core.py:289-398`).
**Объясняет:** почему новые механизмы/дифференциальные вопросы документа не могут
стабильно направлять выбор.
**Не доказывает:** что простая загрузка всего документа улучшит ответы.

### C2. Operational HE ограничен тремя картами

**Факт:** selector перечисляет только `P001/P006/P007`
(`ai/human_experience.py:243-250`).
**Объясняет:** отсутствие профессиональной опоры вне трёх паттернов.
**Не доказывает:** что включение всех 30 безопасно или полезно.

### C3. Professional Knowledge Base в active path отсутствует

**Факт:** legacy `knowledge_base.json` загружается только legacy
`ai/engines/response_engine.py:60,109`; `DialogEngine` вызывает Cognitive Core
напрямую.
**Объясняет:** reliance на pretraining модели для доменной экспертизы.

### C4. Контракт допускает semantic-delivery gaps

**Факт:** Python проверяет references и kinds, но не эквивалентность содержания
ActionContent и текста reply. R04/R06/R20 это демонстрируют.
**Объясняет:** полезное внутреннее содержание может ослабляться, а вопрос —
появляться без target.
**Не доказывает:** что нужен второй LLM-вызов.

### C5. Техническая нагрузка создаёт измеряемые отказы

В `semantic_delta_ab` обе версии дали только `10/15` полностью валидных
результатов. Ошибки включали local handle в `existing_id`, несовместимый
`ordinary_addressed` и неизвестный `realizes` handle
([`TECHNICAL_REPORT.md:54-74`](../experiments/semantic_delta_ab/results/semantic-delta-ab-20261013T010202Z-8b1ed0be/TECHNICAL_REPORT.md)).
Это подтверждает проблему надёжности, хотя не устанавливает, что именно размер
prompt является причиной.

### C6. Добавление semantic-delta tie-breaker не решило проблему

Варианты A и B дали одинаковую техническую валидность `10/15`; сообщённая
владельцем средняя слепая оценка была `3.1/5` для A и `2.3/5` для B. B отклонён.
Следовательно, добавлять ещё один общий приоритет «новизны» поверх prompt без
предметного знания не обосновано.

## 8. Вероятные причины, требующие проверки

### H1. Knowledge availability ограничивает Formulation

**Наблюдение:** содержательные гипотезы появляются нестабильно, а model input
не содержит KB.
**Механизм:** без bounded mechanisms модель выбирает безопасный echo/question.
**Минимальная проверка:** matched frozen scenarios, control vs 2–3
релевантных knowledge cards; неизменные prompt/schema/model; human blind review
и technical validity отдельно.

### H2. Full HM + contract burden вытесняет консультационную задачу

**Наблюдение:** один вызов выполняет много графовых и текстовых задач.
**Механизм:** ограниченное attention расходуется на reference bookkeeping.
**Проверка:** не redesign production, а lab factorial: current full output vs
семантически эквивалентный упрощённый wire при одинаковой policy/knowledge.

### H3. Human Experience воспринимается как материал «учесть», а не применить

**Наблюдение:** инструкция разрешает карты, но не требует рассмотреть их
конкурентно с alternatives.
**Механизм:** осторожность приводит к отказу от содержательной версии.
**Проверка:** trace-only lab condition с явным formulation choice между
selected card, alternative explanation и no-card; без требования раскрывать CoT.

### H4. One-call planner/writer coupling ухудшает delivery

**Наблюдение:** R04 содержит более сильный ActionContent, чем reply.
**Механизм:** план и естественный текст генерируются совместно, но contract не
гарантирует semantic completeness.
**Проверка:** сначала усилить узкий delivery contract/examples в lab. Только
если это не помогает, сравнить one-call с двухшаговой архитектурой.

### H5. Default model alias и отсутствие pinned inference budget повышают variance

**Наблюдение:** production использует alias/config, тогда как эксперименты
фиксируют snapshot и параметры.
**Проверка:** matched snapshot experiment без изменения knowledge/policy.

## 9. Соответствие Prompt Engineering Rules R01–R20

Матрица ниже опирается на текущую production assembly, а не на лабораторные
варианты. `PARTIAL` не означает автоматически дефект; это указатель риска.

| Правило | Оценка | Существенное свидетельство | Риск |
|---|---|---|---|
| R01 Наблюдаемая цель | PARTIAL | Цель consultation описана, но одновременно требуется state graph и wire | Средний |
| R02 Разделять инструкцию и данные | PARTIAL | JSON-контекст отделён, но нет явной trust-boundary маркировки embedded text/cards | Высокий |
| R03 Ясная роль | PARTIAL | Роль консультанта ясна; доменная компетентность не обеспечена знаниями | Высокий |
| R04 Явный порядок приоритетов | PARTIAL | Есть expected useful delta, но нет полного tie-break между Recognition/Reflection/Question/Stop | Высокий |
| R05 Конкретные ограничения | PARTIAL | Много точных guards, часть семантических запретов непроверяема кодом | Средний |
| R06 Минимум противоречий | PARTIAL | Policy консолидирована, но осторожность и содержательное продвижение всё ещё конкурируют | Высокий |
| R07 Примеры критической формы | FAIL | Production prompt не содержит проверенных end-to-end wire examples | Высокий |
| R08 Делить сложную задачу | PARTIAL | Секции ясны, но один output объединяет независимые cognitive/graph/reply задачи | Высокий |
| R09 Давать критерии успеха | PARTIAL | Contract success точен; консультационная полезность описана качественно | Средний |
| R10 Structured output | PASS | Strict JSON Schema + Pydantic + adapter | Низкий |
| R11 Ограничивать свободный формат | PASS | Enums/unions/handles; fail-closed | Низкий |
| R12 Согласовать schema и текст | PARTIAL | В основном согласованы; prefix semantics виднее Python, чем JSON Schema | Высокий |
| R13 Grounding | PASS | Exact current-user source quotes и provenance/status | Низкий |
| R14 Few-shot examples | FAIL | Нет production examples; reference pilot показал потенциальную пользу для валидности | Высокий |
| R15 Не просить скрытый CoT | PASS | Модель отдаёт selected result, не rationale | Низкий |
| R16 Управлять объёмом контекста | PARTIAL | History bounded closure, но Human Model целиком и без budget | Средний |
| R17 Версионирование model/config | FAIL | Production alias/config не даёт экспериментальной воспроизводимости | Средний |
| R18 Проверять инструкции эмпирически | PARTIAL | Есть лабораторные runs, но мало knowledge-specific и multi-turn quality eval | Высокий |
| R19 Разделять техническую и качественную оценку | PARTIAL | Новые эксперименты разделяют, production monitoring недостаточен | Средний |
| R20 Failure/recovery design | PARTIAL | Fail-closed/no retry безопасны, но valid transport + invalid semantic output завершает turn без recovery | Высокий |

Полный предыдущий compliance audit содержит построчное обоснование
([`COGNITIVE_CORE_PROMPT_COMPLIANCE_AUDIT.md:64-327`](../COGNITIVE_CORE_PROMPT_COMPLIANCE_AUDIT.md)).

## 10. Результаты дополнительных экспериментов

Новых API-экспериментов для этого аудита не проводилось: существующих данных
достаточно, чтобы подтвердить отсутствие KB и structural/delivery gaps, но
недостаточно для выбора новой knowledge architecture. API calls: `0`, cost: `$0`.

### Semantic reference A/B pilot

- 12/12 API responses, все JSON Schema и initial SemanticTurn validation прошли.
- Control A: 4/6 complete validity.
- Treatment B с компактными reference rules/examples: 6/6 complete validity.
- Совокупно adapter вернул результат в 10/12 случаях.
- Малый sample и отсутствие завершённой независимой quality evaluation не
  позволяют объявить examples общим победителем.

Источник: [`experiments/semantic_reference_ab/FINAL_REPORT.md:36-137`](../experiments/semantic_reference_ab/FINAL_REPORT.md).

### Semantic delta A/B

- 30 вызовов; A 10/15, B 10/15 технически валидных.
- Токены: 254,636 total; зафиксированная стоимость около `$0.0451612`.
- B имел иную scenario distribution, но не повысил общую техническую валидность.
- По сообщённой owner blind evaluation B хуже; менять production на B нельзя.
- R04/R15 разделили «потерю смысла» и «низкую новизну»; R06/R20 подтвердили
  возможность вопроса без target; R03/R10/R26 — local-vs-existing failures.

Эти эксперименты проверяли reference prompting и semantic delta, **не** наличие
профессиональной KB. Они не отвечают, повысит ли curated knowledge качество.

## 11. Пробелы в знаниях

### 11.1. Предметные пробелы

Нужны проверенные карточки как минимум по:

- энергетическому балансу без редукции проблемы к «силе воли»;
- restriction → hunger/overeating и различению физического/эмоционального;
- sleep/fatigue, сменной/сидячей работе и food environment;
- устойчивости изменений, lapse vs relapse, rigid rules/all-or-nothing;
- habit/cue/reward и социальному контексту;
- medication/endocrine/rapid unexplained change boundaries;
- binge-eating и другим eating-disorder red flags;
- shame/self-criticism без навязывания скрытого мотива;
- корректной маршрутизации к медицинской/психологической помощи.

Каждая карточка должна содержать не «готовую фразу», а:

`mechanism`, `evidence_basis`, `applicability_signals`,
`alternative_explanations`, `discriminating_observations`, `avoid`,
`red_flags`, `uncertainty`, `scope/referral`.

### 11.2. Evidence gaps нового документа

- Он является профессионально правдоподобным synthesis, но не прошёл внешнюю
  экспертную/клиническую рецензию.
- Часть источников — authoritative guidelines, часть — методы коммуникации и
  sales discovery. SPIN допустим только как структура исследования, не как
  техника давления на конверсию.
- Основания четырёх книг для HE доступны лишь как открытые descriptions/TOCs;
  нельзя представлять derived карты как прямые утверждения авторов книг.
- Нет доказательства, что предложенные реплики повышают outcomes или conversion.

### 11.3. Evaluation gaps

- Нет gold/consensus разметки профессиональными специалистами.
- Нет safety suite для red flags.
- Нет систематического multi-turn теста false recognition/recovery.
- Нет метрик calibration: сколько предложенных механизмов поддержано,
  частично поддержано, отвергнуто и повторно ошибочно возвращено.
- Нет knowledge provenance telemetry на конкретный action.

## 12. Варианты архитектуры

### Вариант A — эволюция текущего Cognitive Core (рекомендуется)

Добавить один read-only Professional Knowledge Layer перед `propose()`:

`HM + ACS + history → deterministic/bounded retrieval → 2–4 knowledge cards + HE cards → existing CognitiveCore call`.

Formulation и Decision остаются внутри LLM. Python только извлекает кандидаты
по прозрачным признакам и передаёт их; он не решает, что истинно о человеке.
State application и wire не меняются. Knowledge IDs можно логировать как
provenance использованного ActionContent, не записывая knowledge как user fact.

**Плюсы:** минимальный архитектурный риск; сохраняет one-call; непосредственно
закрывает главный пробел; позволяет постепенно валидировать карточки.
**Минусы:** retrieval требует качества; один вызов остаётся нагруженным;
необходимо избежать превращения карточек в scripts.

### Вариант B — существенный redesign Cognitive Core

Разделить formulation/selection и response/state packaging на два модельных
шага либо две специализированные модели. Первый выбирает competing
explanations/next move, второй пишет SemanticTurn/reply.

**Плюсы:** лучше observability; можно независимо оценивать formulation и
delivery; меньше конкуренции задач внутри одного output.
**Минусы:** два вызова, latency/cost, риск расхождения planner/writer, новый
failure surface, нарушение нынешнего one-call product constraint. Данных пока
недостаточно, чтобы оправдать production redesign.

### Вариант C — agentic/RAG architecture

Отдельные knowledge, safety, formulation и dialogue agents/tools с retrieval
по внешнему корпусу.

**Плюсы:** максимальная расширяемость и specialist routing.
**Минусы:** высокая сложность, cost/latency, труднее обеспечить selected-only и
единый provenance, больше способов противоречия. Это преждевременно для
текущего масштаба и экспериментальной зрелости.

## 13. Рекомендуемый вариант

Рекомендуется **вариант A+**: эволюция текущей архитектуры с curated
Professional Knowledge Layer и отдельной программой evaluation.

Принципы реализации:

1. **Не смешивать четыре слоя.** Methodology управляет выбором; Knowledge Base
   предлагает профессиональные механизмы; Human Experience предлагает способы
   переживания/узнавания; Human Model хранит индивидуально подтверждённое.
2. **Начать с малого.** Не 30 HE-карт и не весь документ в prompt, а 10–20
   рецензированных mechanism cards с источниками и safety limits.
3. **Передавать alternatives.** Для каждой релевантной версии дать модели хотя
   бы одно materially different explanation и discriminating observation.
4. **Не материализовать автоматически.** Knowledge/HE только расширяет
   Formulation; HM меняется по текущим grounding/provenance rules.
5. **Сделать использование наблюдаемым.** Записывать в lab telemetry IDs
   доступных и фактически выбранных knowledge/HE cards без скрытого CoT.
6. **Сохранить один вызов на первом этапе.** Рассматривать вариант B только при
   доказанном failure A+ в matched experiment.
7. **Отдельная safety routing.** Медицинские/ED/crisis red flags должны вести к
   подходящему типу помощи, не к generic sales handoff.
8. **Упростить техническую часть точечно.** Сделать ID constraints видимыми в
   API schema и дать проверенные короткие examples; не добавлять общий
   semantic-delta appendix, который уже не подтвердился.

Почему это не просто prompt fix: prompt уже говорит «понимай», «не повторяй» и
«выбирай useful delta». Он не может надёжно заменить отсутствующий, версионируемый
и проверяемый предметный материал. Почему это не новый Engine: слой retrieval
имеет отдельную ответственность — безопасно отобрать knowledge candidates, но
не принимает conversational Decision.

## 14. Следующая последовательность работ

1. **Knowledge specification.** Зафиксировать JSON/data contract mechanism card,
   provenance/source requirements, uncertainty и safety taxonomy.
2. **Expert review.** Рецензировать первые 10–20 карточек специалистом по
   evidence-based weight management; отдельно проверить red flags.
3. **Offline admissibility.** Валидировать, что ни одна карточка не утверждает
   индивидуальную причину, диагноз или эффективность без основания.
4. **Lab retrieval.** Реализовать bounded retrieval только в experiment package;
   production не менять.
5. **Frozen matched evaluation.** Сценарии должны покрывать механизм доступен,
   механизм неприменим, конкурирующие объяснения, correction/rejection, red flag
   и no-knowledge abstention. Сравнивать current production vs current + cards.
6. **Раздельные метрики.** Technical validity, consultation value, novelty,
   groundedness, correctability, safety и burden оценивать отдельно; единица
   повторяемости — независимый production selection, не evaluator vote.
7. **Human expert blind review.** Не использовать только LLM evaluator; заранее
   заморозить rubric и stopping rule.
8. **Pilot gate.** Расширять knowledge/HE coverage только если нет роста false
   recognition, unsupported causality и hard-stop rate.
9. **Production integration A+.** После gate подключить read-only knowledge
   projection к существующему input assembly; one-call и StateApplier оставить.
10. **Post-integration monitoring.** Отслеживать card availability/use,
    supported/partial/rejected outcomes, повторное возвращение rejected versions,
    safety routing и contract failures.
11. **Architecture decision gate.** Только если A+ не улучшает delivery либо
    technical reliability, провести лабораторный one-call vs two-stage тест
    варианта B. До этого redesign не обоснован.

### Итоговый статус

- Главный knowledge gap: **подтверждён**.
- Неспособность текущей модели использовать качественно поданное знание:
  **не доказана**.
- Необходимость нового Engine или multi-call architecture: **не доказана**.
- Необходимость curated Professional Knowledge Layer и safety evaluation:
  **обоснована архитектурой, материалами и экспериментами**.
- Production changes: `0`.
- API calls during audit: `0`; actual cost: `$0`.
- Commit/push: не выполнялись.
