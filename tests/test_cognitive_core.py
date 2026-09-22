import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from ai.cognitive_core import (
    CognitiveCore,
    CognitiveCoreError,
    _normalize_strict_schema,
)
from ai.cognitive_semantic_wire import SEMANTIC_TURN_JSON_SCHEMA, adapt_semantic_turn
from test_cognitive_semantic_wire import (
    COMPLEX_TEXT,
    MI_A,
    MI_B,
    MSG_A,
    RT_A,
    SIMPLE_QUOTE,
    SIMPLE_TEXT,
    _acs as semantic_acs,
    _complex as semantic_complex,
    _model as semantic_model,
    _simple as semantic_simple,
)
from ai.cognitive_turn import (
    CognitiveTurnResult,
    DecisionIntent,
)
from ai.discovery_data_model import (
    ActiveContentItem,
    ActiveContentKind,
    ActiveConversationState,
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    ItemStatus,
    ModelItem,
    ModelItemKind,
    Provenance,
    Relation,
    ResponseTarget,
    SourceReference,
    TargetInteractionKind,
    TargetReference,
    TargetSubjectKind,
)
from ai.discovery_state_applier import apply_cognitive_turn


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def _message(role, text, sequence):
    return DialogueMessage(role=role, text=text, sequence=sequence)


def _fixture(with_acs=True):
    history_messages = tuple(
        message
        for turn in range(1, 7)
        for message in (
            _message(DialogueRole.USER, f"старое user {turn}", (turn * 2) - 1),
            _message(DialogueRole.SYSTEM, f"старое system {turn}", turn * 2),
        )
    )
    old_user = history_messages[0]
    old_system = history_messages[1]
    item = ModelItem(
        kind=ModelItemKind.EXPERIENCE,
        content="старый значимый опыт",
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.REJECTED,
        source_refs=(SourceReference(message_id=old_user.id),),
        created_turn=1,
        updated_turn=1,
    )
    corrected = ModelItem(
        kind=ModelItemKind.EXPERIENCE,
        content="скорректированный опыт",
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.CORRECTED,
        source_refs=(SourceReference(message_id=old_user.id),),
        created_turn=1,
        updated_turn=2,
    )
    relation = Relation(
        meaning="пользователь видит связь",
        source_item_ids=(item.id,),
        target_item_ids=(corrected.id,),
        provenance=Provenance.USER_INTERPRETATION,
        status=ItemStatus.UNCERTAIN,
        source_refs=(SourceReference(message_id=old_user.id),),
        created_turn=1,
        updated_turn=2,
    )
    model = HumanModel(items={item.id: item, corrected.id: corrected}, relations={relation.id: relation})
    acs = None
    if with_acs:
        content = ActiveContentItem(
            kind=ActiveContentKind.SYSTEM_PROPOSAL,
            content="предыдущая системная версия",
            model_item_ids=(item.id,),
            source_message_id=old_system.id,
        )
        target = ResponseTarget(
            active_content_id=content.id,
            subject=TargetReference(kind=TargetSubjectKind.MODEL_ITEM, id=item.id),
            interaction=TargetInteractionKind.EVALUATION,
        )
        acs = ActiveConversationState(
            created_turn=1,
            source_system_message_id=old_system.id,
            active_content={content.id: content},
            response_targets={target.id: target},
        )
    current = _message(DialogueRole.USER, "текущая реплика", 13)
    return current, model, acs, DialogueHistory(messages=history_messages)


def _semantic_baseline():
    payload = semantic_simple()
    payload["state_patch"] = {
        "item_changes": [], "relation_changes": [], "proposal_materializations": []
    }
    payload["reconciliation"] = []
    payload["selected_action"]["primary_content"]["model_item_refs"] = []
    return payload


def _core(result=None):
    client = FakeClient(json.dumps(_semantic_baseline(), ensure_ascii=False))
    return CognitiveCore(client=client), client


def _payload(client):
    return json.loads(client.completions.calls[0]["messages"][1]["content"])


def _objects_with_properties(schema):
    if isinstance(schema, dict):
        if isinstance(schema.get("properties"), dict):
            yield schema
        for value in schema.values():
            yield from _objects_with_properties(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects_with_properties(value)


def _selected_action_schema(intent):
    for branch in SEMANTIC_TURN_JSON_SCHEMA["properties"]["selected_action"][
        "anyOf"
    ]:
        definition = SEMANTIC_TURN_JSON_SCHEMA["$defs"][
            branch["$ref"].rsplit("/", 1)[-1]
        ]
        if definition["properties"]["decision_intent"]["const"] == intent.value:
            return definition
    raise AssertionError(f"Missing wire branch for {intent}")


def test_input_view_includes_current_message_model_and_disabled_he():
    current, model, acs, history = _fixture()
    core, client = _core()

    core.propose(current, model, acs, history)

    payload = _payload(client)
    assert payload["current_user_message"] == {
        "id": current.id,
        "sequence": 13,
        "text": "текущая реплика",
    }
    assert payload["human_experience"] == {"enabled": False}
    assert {item["status"] for item in payload["human_model"]["items"]} == {
        "rejected",
        "corrected",
    }
    relation = payload["human_model"]["relations"][0]
    assert relation["status"] == "uncertain"
    assert relation["source_message_ids"] == [history.messages[0].id]


def test_previous_acs_view_and_system_prompt_are_sent():
    current, model, acs, history = _fixture()
    core, client = _core()

    core.propose(current, model, acs, history)

    payload = _payload(client)
    assert payload["previous_active_conversation_state"]["active_content"][0]["id"] == next(iter(acs.active_content))
    assert payload["previous_active_conversation_state"]["response_targets"][0]["interaction"] == "evaluation"
    assert "один полный Cognitive Discovery turn" in client.completions.calls[0]["messages"][0]["content"]


def test_system_prompt_includes_critical_conditional_contract_rules():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "CONTRACT RULES",
        "ADD item/relation declares a new handle",
        "CORRECT declares a new handle AND explicitly names the existing object",
        "REINFORCE names only the existing object",
        "source_items, target_items and meaning",
        "AMBIGUOUS reconciliation names multiple distinct previous targets",
        "Every targeted content must be realized by a segment",
    ):
        assert instruction in prompt


def test_system_prompt_requires_shared_handles_and_exact_reference_reuse():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    assert "semantic handles" in prompt
    assert "одном общем namespace" in prompt
    assert "каждый объявленный handle уникален" in prompt
    assert "A handle is valid only for its declaration in this response" in prompt
    assert "exact persistent IDs from previous ACS" in prompt


def test_system_prompt_defines_item_and_relation_reference_ownership():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "exact existing_id from HumanModel OR the handle of a current ADD/CORRECT declaration",
        "never as an alias for an existing HumanModel object",
        "confirmation or Reflection without new operations",
        "existing IDs or use empty reference lists",
    ):
        assert instruction in prompt


def test_system_prompt_explains_active_content_target_ownership():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "owner_content handle names the current content carrying the interaction",
        "subject independently names current content",
        "Owner and subject may differ",
        "ReplySegment.realizes names the current content",
    ):
        assert instruction in prompt


def test_system_prompt_distinguishes_consent_from_open_response():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "OPEN_RESPONSE asks the user for their own open material or answer",
        "It is not a request to accept or decline a proposed action",
        "CLARIFICATION asks the user to clarify the addressed meaning",
        "EVALUATION asks the user to evaluate a system-proposed version",
        "CONSENT asks the user to accept or decline a concrete proposed action",
        "ResponseTarget MUST use interaction=CONSENT, not OPEN_RESPONSE",
        "semantic examples, not keyword triggers",
        "CONSENTED and DECLINED resolve only a CONSENT target",
        "Do not use REFUSED as a substitute for DECLINED",
        "AMBIGUOUS means the user's response cannot be assigned to one specific previous target",
    ):
        assert instruction in prompt


def test_system_prompt_uses_simple_consultant_policy():
    prompt = " ".join(CognitiveCore._system_prompt().split())
    for principle in (
        "опытным дружелюбным консультантом",
        "сам, внутренне реши, нужна ли дополнительная информация",
        "Уменьшение неопределённости системы само по себе не является ценностью",
        "мягкое и легко исправляемое предположение",
        "не предлагая отдельную Evaluation каждого такого предположения",
        "Если пользователь поправил, прими поправку",
        "существенно разные правдоподобные трактовки",
        "разумное предположение может заметно увести его не туда",
        "Ясно сказанное не требует",
        "Отдельный эпизод можно принять как достаточный контекст",
        "не расследуй автоматически его точный механизм",
        "можно коротко спросить, повторяется ли ситуация",
        "это не обязательный вопрос после каждого эпизода",
        "существенный или повторяющийся паттерн",
        "Reflection нужна, когда она соединяет части истории",
        "Не повторяй последнюю понятную реплику ради подтверждения",
        "Не добывай ту же деталь снова",
        "Можно перейти к другой важной части истории",
        "ни один тип действия не имеет глобального приоритета",
    ):
        assert principle in prompt

    for removed_concept in (
        "OPERATIONAL CANDIDATE-SELECTION PROTOCOL",
        "GENERATE DISTINCT ELIGIBLE CANDIDATES",
        "COMPARE MARGINAL DECISION VALUE",
        "decision hinge",
        "strongest eligible alternative",
        "each likely response class",
        "COMMIT THE WINNER",
    ):
        assert removed_concept not in prompt
    assert "FSB-" not in prompt


def test_more_information_criterion_is_internal_not_user_facing_meta_choice():
    source = (
        Path(__file__).parents[1]
        / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
    ).read_text(encoding="utf-8")
    policy_sections = (
        CognitiveCore._system_prompt().split("CONSULTANT POLICY:", 1)[1]
        .split("PRODUCT BOUNDARY:", 1)[0],
        source.split("VALUE / SUFFICIENCY ВНУТРИ DECISION", 1)[1]
        .split("PRODUCTION DECISION BOUNDARY", 1)[0],
    )
    for section in policy_sections:
        policy = " ".join(section.casefold().split())
        assert "внутренне" in policy
        assert "нужна ли дополнительная информация" in policy
        assert "не вопрос пользователю" in policy
        assert "не проговар" in policy
        assert "уточнять или продолжать" in policy
        assert "разговор о процессе консультации" in policy
        assert "уменьшение неопределённости системы само по себе" in policy
        assert "существенно разные" in policy
        assert "сначала спроси" not in policy
        assert "сначала спросить" not in policy


def test_cognitive_cycle_uses_working_memory_not_decision_optimizer():
    source = (
        Path(__file__).parents[1]
        / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
    ).read_text(encoding="utf-8")
    normalized = " ".join(source.split())
    for principle in (
        "опытный дружелюбный консультант",
        "Human Model служит рабочей памятью консультанта",
        "мягкое предположение можно использовать в разговоре без отдельной проверки",
        "Уточнение нужно, когда одновременно есть существенно разные",
        "Отдельный эпизод не требует расследования точной причинной цепочки",
        "Такой вопрос не обязателен для каждого эпизода",
        "Stop Exploration является допустимым действием",
    ):
        assert principle.lower() in normalized.lower()
    for removed_concept in (
        "decision hinge",
        "классы ответа пользователя наиболее вероятны",
        "лучшей содержательно отличающейся альтернативы",
    ):
        assert removed_concept not in normalized


def test_simple_policy_preserves_safety_and_structural_guards():
    prompt = " ".join(CognitiveCore._system_prompt().split())
    for guard in (
        "серьёзная медицинская или психологическая гипотеза не является фактом",
        "Human Experience отключён и не является evidence",
        "rejected не возвращай в иной формулировке",
        "bare yes при нескольких targets не подтверждает их все",
        "отказ, паузу и коррекцию уважай",
        "SYSTEM_TRANSITION должен быть явным и иметь CONSENT target",
        "не выбирай за человека привычку, практику, изменение среды",
        "SYSTEM_PROPOSAL не разрешает скрытый solution design",
        "после DECLINED не продавливай соседний вариант",
        "Отказ от предложения сам по себе не доказывает BARRIER",
        "meaningful consent, не добавляя параллельное AI-предложение решения",
    ):
        assert guard in prompt
    assert set(SEMANTIC_TURN_JSON_SCHEMA["properties"]) == {
        "state_patch",
        "reconciliation",
        "selected_action",
        "reply_segments",
    }
    assert not set(CognitiveTurnResult.model_json_schema()["properties"]).intersection(
        {"candidates", "candidate_rankings", "rationale", "scores"}
    )


def test_system_prompt_enforces_solution_action_product_boundary():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "Discovery помогает понять проблему и существенные паттерны",
        "Общее объяснение обоснованного механизма допустимо ради понимания",
        "не применяй его как персональное решение",
        "не выбирай за человека привычку, практику, изменение среды",
        "SYSTEM_PROPOSAL не разрешает скрытый solution design",
        "ANSWER_USER_QUESTION не отменяет Product Boundary",
        "сначала дай полезное обоснованное понимание проблемы",
        "а не персональную рекомендацию",
        "не делай переход автоматическим по форме вопроса",
        "meaningful consent, не добавляя параллельное AI-предложение решения",
        "Never disguise a concrete experiment, recommendation, plan or practice as Discovery",
    ):
        assert instruction in prompt


def test_direct_solution_question_keeps_explanation_inside_product_boundary():
    policy = " ".join(
        CognitiveCore._system_prompt()
        .split("PRODUCT BOUNDARY:", 1)[1]
        .split("DECISION INTENT:", 1)[0]
        .split()
    ).casefold()

    # The direct question has a useful, grounded answer route, not a forced handoff.
    assert "answer_user_question не отменяет product boundary" in policy
    assert "сначала дай полезное обоснованное понимание проблемы" in policy
    assert "опирайся на уже известный материал" in policy
    assert "можно объяснить общий принцип" in policy
    assert "не делай переход автоматическим по форме вопроса" in policy

    # General education cannot become invented user-specific causes or an AI offer
    # to select a personal solution from a menu of techniques.
    assert "не выдавай типовые причины за установленную причину именно этого человека" in policy
    assert "не превращай объяснение в меню способов действий" in policy
    assert "ai выберет, адаптирует или разработает для него персональное решение" in policy
    assert "если следующий полезный шаг требует такого выбора" in policy
    assert "transition к живому консультанту с meaningful consent" in policy


def test_source_of_truth_separates_explanation_from_personal_solution_design():
    root = Path(__file__).parents[1] / "Research/Human_Experience/Weight"
    boundaries = " ".join((root / "11_BOUNDARIES.txt").read_text(encoding="utf-8").split())
    cycle = " ".join((root / "12_Cognitive_Cycle.txt").read_text(encoding="utf-8").split())
    transition = " ".join(
        (root / "10_TRANSITION_TO_HUMAN_CONSULTANT.txt")
        .read_text(encoding="utf-8").split()
    )

    assert "Общее объяснение не равно проектированию решения" in boundaries
    assert "Текущий AI не должен: - выбирать конкретные привычки" in boundaries
    assert "Прямой вопрос пользователя не отменяет эту границу" in boundaries
    assert "Прямой вопрос не отменяет Product Boundary" in cycle
    assert "не выбирать практику или план для этого человека" in cycle
    assert "не делая переход автоматическим по форме вопроса" in cycle
    assert "AI не подбирает программу питания или лечения" in cycle
    assert "на слое Discovery не должна" in transition
    assert "не должна обязательно" not in transition


def test_system_prompt_respects_declined_without_inventing_problem_barrier():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "Отказ и паузу уважай",
        "после DECLINED не продавливай соседний вариант",
        "Отказ от предложения сам по себе не доказывает BARRIER проблемы",
    ):
        assert instruction in prompt


def test_system_prompt_defines_ephemeral_decision_intents_and_structural_limits():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "Choose Decision before Response",
        "architectural function of the already selected next Cognitive Cycle action",
        "not the topic of its text",
        "HUMAN_DISCOVERY continues understanding the person and Meaningful Change",
        "MECHANISM_DISCOVERY continues understanding Previous Attempts, Barriers",
        "REFLECTION returns a coherent working picture",
        "RECOGNITION offers context-close working versions or options",
        "TRANSITION explicitly offers movement beyond the current Discovery boundary",
        "RESPECT_PAUSE_OR_REFUSAL does not continue a declined or unwanted direction",
        "STOP_EXPLORATION ends the current Discovery",
        "ANSWER_USER_QUESTION answers a direct user question",
        "decision_intent does not prove semantic Product Boundary compliance",
        "Never disguise a concrete experiment, recommendation, plan or practice as Discovery",
        "A CONSENT target subject must be current SYSTEM_PROPOSAL or SYSTEM_TRANSITION",
        "SYSTEM_REFLECTION is evaluated, not accepted or declined",
    ):
        assert instruction in prompt


def test_api_schema_is_strict_semantic_turn():
    schema = SEMANTIC_TURN_JSON_SCHEMA
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {
        "state_patch", "reconciliation", "selected_action", "reply_segments"
    }
    assert "anyOf" not in schema
    assert all(
        node.get("additionalProperties") is False
        and set(node["required"]) == set(node["properties"])
        for node in _objects_with_properties(schema)
    )
    assert len(schema["properties"]["selected_action"]["anyOf"]) == len(DecisionIntent)
    assert {
        _selected_action_schema(intent)["properties"]["decision_intent"]["const"]
        for intent in DecisionIntent
    } == {intent.value for intent in DecisionIntent}
    serialized = json.dumps(schema)
    for obsolete in ("source_span", "local_item_id", "local_relation_id", "local_id"):
        assert obsolete not in serialized
    for unsupported in ('"oneOf"', '"discriminator"', '"default"'):
        assert unsupported not in serialized


def test_production_request_uses_semantic_schema_after_one_call():
    current, model, acs, history = _fixture()
    core, client = _core()
    result = core.propose(current, model, acs, history)
    request = client.completions.calls[0]
    assert len(client.completions.calls) == 1
    assert request["response_format"]["json_schema"] == {
        "name": "semantic_turn", "strict": True, "schema": SEMANTIC_TURN_JSON_SCHEMA
    }
    assert isinstance(result, CognitiveTurnResult)
    assert result.decision_intent is DecisionIntent.HUMAN_DISCOVERY


def _semantic_context(text, *, multiple=False, proposal=False):
    history_message = DialogueMessage(
        id=MSG_A, role=DialogueRole.SYSTEM, text="Предыдущий вопрос", sequence=1
    )
    current = DialogueMessage(role=DialogueRole.USER, text=text, sequence=2)
    return current, semantic_model(), semantic_acs(multiple=multiple, proposal=proposal), DialogueHistory(messages=(history_message,))


def _run_semantic(payload, *, text=SIMPLE_TEXT, multiple=False, proposal=False):
    current, model, acs, history = _semantic_context(
        text, multiple=multiple, proposal=proposal
    )
    client = FakeClient(json.dumps(payload, ensure_ascii=False))
    core = CognitiveCore(client)
    return core, client, current, model, acs, history


def test_valid_simple_semantic_response_maps_losslessly():
    payload = semantic_simple()
    core, client, current, model, acs, history = _run_semantic(payload)
    result = core.propose(current, model, acs, history)
    expected = adapt_semantic_turn(
        payload, current_user_text=current.text, human_model=model, previous_acs=acs
    )
    assert result == expected
    assert result.state_patch.item_operations[0].local_id.startswith("item:")
    assert result.reconciliation[0].previous_response_target_ids == (RT_A,)
    assert len(client.completions.calls) == 1


def test_valid_complex_semantic_response_maps_losslessly():
    payload = semantic_complex()
    core, client, current, model, acs, history = _run_semantic(
        payload, text=COMPLEX_TEXT, multiple=True, proposal=True
    )
    result = core.propose(current, model, acs, history)
    expected = adapt_semantic_turn(
        payload, current_user_text=current.text, human_model=model, previous_acs=acs
    )
    assert result == expected
    target = result.system_action.response_targets[0]
    assert target.active_content_local_id != target.subject.active_content_local_id
    assert len(result.state_patch.item_operations) == 3
    assert len(result.reconciliation) == 2
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("defect", ("duplicate", "unknown", "wrong_type", "bad_existing", "bad_quote", "final_contract"))
def test_semantic_boundary_fail_closed_after_one_call(defect):
    payload = semantic_simple()
    if defect == "duplicate":
        payload["selected_action"]["primary_content"]["handle"] = "evening_hunger"
    elif defect == "unknown":
        payload["selected_action"]["response_targets"][0]["owner_content"] = "unknown"
    elif defect == "wrong_type":
        payload["selected_action"]["primary_content"]["model_item_refs"] = [{"handle": "new_relation"}]
        payload["state_patch"]["relation_changes"].append({
            "operation": "add", "handle": "new_relation",
            "source_items": [{"existing_id": MI_A}], "target_items": [{"existing_id": MI_B}],
            "meaning": "Связь", "evidence_origin": "current_user_interpretation",
            "source_quote": SIMPLE_QUOTE,
        })
    elif defect == "bad_existing":
        payload["selected_action"]["primary_content"]["model_item_refs"] = [{"existing_id": "mi_" + "f" * 32}]
    elif defect == "bad_quote":
        payload["state_patch"]["item_changes"][0]["source_quote"] = "несуществующая цитата"
    else:
        payload["selected_action"]["response_targets"][0]["interaction"] = "consent"
    core, client, current, model, acs, history = _run_semantic(payload)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize(("content", "category"), (
    ("", "structured_output_failure"),
    ("not json", "parsing_failure"),
    ('{"state_patch": {}}', "validation_failure"),
))
def test_invalid_output_is_not_retried(content, category):
    current, model, acs, history = _fixture()
    client = FakeClient(content)
    core = CognitiveCore(client)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == category
    assert len(client.completions.calls) == 1


def test_api_failure_is_reported_without_retry():
    current, model, acs, history = _fixture()
    client = FakeClient("unused")

    def fail(**kwargs):
        client.completions.calls.append(kwargs)
        raise OSError("offline transport failure")

    client.completions.create = fail
    core = CognitiveCore(client)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == "api_failure"
    assert len(client.completions.calls) == 1


def test_semantic_dto_validation_failure_is_reported():
    payload = _semantic_baseline()
    payload["selected_action"]["primary_content"]["kind"] = "not_a_kind"
    current, model, acs, history = _fixture()
    client = FakeClient(json.dumps(payload))
    with pytest.raises(CognitiveCoreError) as error:
        CognitiveCore(client).propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("quote", ("missing", "о", ""))
def test_production_exact_quote_failure_does_not_mutate_response(quote):
    payload = semantic_simple()
    payload["state_patch"]["item_changes"][0]["source_quote"] = quote
    original = deepcopy(payload)
    core, client, current, model, acs, history = _run_semantic(payload)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert payload == original
    assert len(client.completions.calls) == 1


def _quote_case(branch, text, quote):
    current, model, acs, history = _fixture()
    current = _message(DialogueRole.USER, text, 13)
    payload = _semantic_baseline()
    if branch == "item":
        payload["state_patch"]["item_changes"] = [{
            "operation": "reinforce", "existing_id": next(iter(model.items)),
            "evidence_origin": "current_user_material", "source_quote": quote,
        }]
    elif branch == "relation":
        payload["state_patch"]["relation_changes"] = [{
            "operation": "reinforce", "existing_id": next(iter(model.relations)),
            "evidence_origin": "current_user_material", "source_quote": quote,
        }]
    else:
        payload["reconciliation"] = [{
            "previous_target_ids": [next(iter(acs.response_targets))],
            "outcome": "supported", "source_quote": quote,
        }]
    return payload, current, model, acs, history


@pytest.mark.parametrize("branch", ("item", "relation", "reconciliation"))
@pytest.mark.parametrize(("text", "quote"), (
    ("Начало. Хочется есть вечером.", "Хочется есть вечером."),
    ("🙂 е\u0308ж — текст!", "е\u0308ж"),
    (" a  b ", " a  b "),
))
def test_exact_quote_resolution_through_production(branch, text, quote):
    payload, current, model, acs, history = _quote_case(branch, text, quote)
    original = deepcopy(payload)
    client = FakeClient(json.dumps(payload, ensure_ascii=False))
    result = CognitiveCore(client).propose(current, model, acs, history)
    assert payload == original
    entry = (
        result.state_patch.item_operations[0] if branch == "item" else
        result.state_patch.relation_operations[0] if branch == "relation" else
        result.reconciliation[0]
    )
    assert entry.source_span.char_start == text.index(quote)
    assert entry.source_span.char_end == text.index(quote) + len(quote)
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("quote", (
    "нет", "да", "aa", "", None, 123, "текст", "a b", "e\u0301", "Текст!", " Текст "
))
def test_invalid_quotes_fail_closed_through_production(quote):
    text = (
        "да, да" if quote == "да" else "aaaa" if quote == "aa" else
        "a  b" if quote == "a b" else "é" if quote == "e\u0301" else "Текст"
    )
    payload, current, model, acs, history = _quote_case("item", text, quote)
    client = FakeClient(json.dumps(payload, ensure_ascii=False))
    with pytest.raises(CognitiveCoreError) as error:
        CognitiveCore(client).propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert len(client.completions.calls) == 1


def test_numeric_source_span_cannot_bypass_semantic_quote_boundary():
    payload = semantic_simple()
    payload["state_patch"]["item_changes"][0]["source_span"] = {
        "char_start": 0, "char_end": len(SIMPLE_TEXT)
    }
    core, client, current, model, acs, history = _run_semantic(payload)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert len(client.completions.calls) == 1


def test_final_internal_contract_rejects_semantic_consent_mismatch():
    payload = semantic_simple()
    payload["selected_action"]["response_targets"][0]["interaction"] = "consent"
    core, client, current, model, acs, history = _run_semantic(payload)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert error.value.__cause__.__cause__.__class__.__name__ == "ValidationError"
    assert len(client.completions.calls) == 1


def test_missing_structural_history_reference_fails_before_api_call():
    current, model, acs, history = _fixture()
    missing = next(iter(model.items.values()))
    foreign = DialogueMessage(role=DialogueRole.USER, text="нет", sequence=99)
    invalid = missing.model_copy(update={
        "source_refs": (SourceReference(message_id=foreign.id),)
    })
    invalid_model = model.with_updates(items={**model.items, invalid.id: invalid})
    core, client = _core()
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, invalid_model, acs, history)
    assert error.value.category == "history_reference_failure"
    assert client.completions.calls == []


def test_result_can_be_applied_by_unchanged_state_applier():
    current = _message(DialogueRole.USER, "Мне тяжело", 1)
    history = DialogueHistory()
    core, client = _core()
    result = core.propose(current, HumanModel(), None, history)
    system = _message(DialogueRole.SYSTEM, "Что в этом для вас было самым трудным?", 2)
    applied = apply_cognitive_turn(HumanModel(), None, history, current, system, result)
    assert isinstance(result, CognitiveTurnResult)
    assert applied.new_acs is not None
    assert len(client.completions.calls) == 1


def test_input_view_history_closure_and_manual_state_application():
    current, model, acs, history = _fixture()
    core, client = _core()
    result = core.propose(current, model, acs, history)
    selected = _payload(client)["dialogue_history"]
    sequences = [message["sequence"] for message in selected]
    assert sequences == sorted(sequences)
    assert len({message["id"] for message in selected}) == len(selected)
    assert history.messages[0].id in {message["id"] for message in selected}
    assert history.messages[1].id in {message["id"] for message in selected}
    assert isinstance(result, CognitiveTurnResult)


def test_prompt_describes_semantic_responsibilities_without_old_wire_bookkeeping():
    prompt = " ".join(CognitiveCore._system_prompt().split())
    for instruction in (
        "SEMANTIC WIRE", "SEMANTIC CONTRACT", "semantic handles",
        "exact persistent IDs", "ADD item/relation", "CORRECT",
        "REINFORCE", "source_items", "target_items", "reconciliation",
        "proposal_materializations", "primary_content", "owner_content",
        "subject", "interaction", "ReplySegment.realizes", "source_quote",
    ):
        assert instruction.lower() in prompt.lower()
    for obsolete in (
        "LOCAL ID NAMESPACES", "ItemOperation.local_id",
        "existing_item_id=null", "realizes_action_content_ids",
        "local_item_id", "local_relation_id", "content:<name>",
    ):
        assert obsolete not in prompt


def test_schema_normalization_does_not_mutate_pydantic_source_schema():
    from ai.cognitive_semantic_wire import SemanticTurn
    raw = SemanticTurn.model_json_schema()
    original = deepcopy(raw)
    normalized = _normalize_strict_schema(raw)
    assert raw == original
    assert normalized == raw
    assert normalized == SEMANTIC_TURN_JSON_SCHEMA
