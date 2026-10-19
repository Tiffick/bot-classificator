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
from ai.cognitive_semantic_wire import (
    SEMANTIC_TURN_JSON_SCHEMA,
    adapt_semantic_turn,
    semantic_turn_json_schema,
)
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
from ai.human_experience import HumanExperienceSelection
from ai.knowledge_retrieval import (
    KNOWLEDGE_RETRIEVAL_FLAG,
    PROFESSIONAL_KNOWLEDGE_SYSTEM_INSTRUCTION,
    KnowledgeRetrievalError,
)


@pytest.fixture(autouse=True)
def _isolate_semantic_retrieval(monkeypatch):
    """Unit tests opt in explicitly; production defaults are tested separately."""
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "false")


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


def test_default_input_view_includes_current_message_model_and_enabled_he(monkeypatch):
    current, model, acs, history = _fixture()
    core, client = _core()
    monkeypatch.delenv("HUMAN_EXPERIENCE_ENABLED", raising=False)

    core.propose(current, model, acs, history)

    payload = _payload(client)
    assert payload["current_user_message"] == {
        "id": current.id,
        "sequence": 13,
        "text": "текущая реплика",
    }
    assert payload["human_experience"]["enabled"] is True
    assert payload["human_experience"]["abstained"] is True
    assert {item["status"] for item in payload["human_model"]["items"]} == {
        "rejected",
        "corrected",
    }
    relation = payload["human_model"]["relations"][0]
    assert relation["status"] == "uncertain"
    assert relation["source_message_ids"] == [history.messages[0].id]


def test_human_experience_flag_off_keeps_base_prompt_and_disabled_input(monkeypatch):
    current, model, acs, history = _fixture()
    monkeypatch.setenv("HUMAN_EXPERIENCE_ENABLED", "false")
    explicit_off = CognitiveCore._messages(current, model, acs, history)

    assert explicit_off[0]["content"] == CognitiveCore._system_prompt()
    assert json.loads(explicit_off[1]["content"])["human_experience"] == {
        "enabled": False
    }


def test_human_experience_flag_off_does_not_load_or_retrieve(monkeypatch):
    current, model, acs, history = _fixture()
    core, client = _core()
    monkeypatch.setenv("HUMAN_EXPERIENCE_ENABLED", "false")

    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("retrieval must not run while the flag is off")

    monkeypatch.setattr("ai.cognitive_core.select_human_experience", unexpected_retrieval)

    core.propose(current, model, acs, history)

    assert len(client.completions.calls) == 1


def test_human_experience_flag_on_adds_bounded_p001_hypothesis(monkeypatch):
    history = DialogueHistory(messages=(
        _message(
            DialogueRole.USER,
            "После тяжёлой недели пропускаю обед и вечером переедаю.",
            1,
        ),
        _message(DialogueRole.SYSTEM, "Что происходит после этого?", 2),
        _message(DialogueRole.USER, "Думаю, что всё испортил.", 3),
        _message(DialogueRole.SYSTEM, "И что вы тогда делаете?", 4),
        _message(DialogueRole.USER, "Бросаю попытку на несколько недель.", 5),
        _message(DialogueRole.SYSTEM, "Это повторяется?", 6),
    ))
    current = _message(DialogueRole.USER, "Да, почти каждый раз.", 7)
    monkeypatch.delenv("HUMAN_EXPERIENCE_ENABLED", raising=False)

    messages = CognitiveCore._messages(current, HumanModel(), None, history)
    payload = json.loads(messages[1]["content"])

    assert "EXPERIMENTAL HUMAN EXPERIENCE MODE" in messages[0]["content"]
    assert "Human Experience отключён" not in messages[0]["content"]
    assert "профессиональные working hypotheses" in messages[0]["content"]
    assert payload["human_experience"]["enabled"] is True
    assert payload["human_experience"]["status"] == (
        "experimental_hypotheses_not_evidence"
    )
    assert [
        card["id"] for card in payload["human_experience"]["selected_cards"]
    ] == ["P001"]
    card = payload["human_experience"]["selected_cards"][0]
    assert "professional_hypothesis" in card
    assert "recognition_example" in card
    assert "discriminating_question" in card
    serialized_context = json.dumps(payload["human_experience"], ensure_ascii=False)
    assert "Возможно, сложнее всего не сам трудный вечер" in serialized_context
    assert "Что именно кажется потерянным после такого вечера" in serialized_context
    assert "original_v1" not in payload["human_experience"]


def test_enabled_human_experience_can_abstain_without_cards(monkeypatch):
    current, model, acs, history = _fixture(with_acs=False)
    monkeypatch.setenv("HUMAN_EXPERIENCE_ENABLED", "true")

    messages = CognitiveCore._messages(current, model, acs, history)
    context = json.loads(messages[1]["content"])["human_experience"]

    assert context["enabled"] is True
    assert context["abstained"] is True
    assert context["selected_cards"] == []
    assert context["abstain_reason"] == "no_reliable_match"


def test_enabled_human_experience_keeps_catalog_failure_as_abstention(monkeypatch):
    current, model, acs, history = _fixture(with_acs=False)
    monkeypatch.setenv("HUMAN_EXPERIENCE_ENABLED", "true")
    monkeypatch.setattr(
        "ai.cognitive_core.select_human_experience",
        lambda *args, **kwargs: HumanExperienceSelection(
            cards=(), abstain_reason="catalog_unavailable"
        ),
    )

    messages = CognitiveCore._messages(current, model, acs, history)
    context = json.loads(messages[1]["content"])["human_experience"]

    assert context["abstained"] is True
    assert context["abstain_reason"] == "catalog_unavailable"


def test_human_experience_does_not_mask_ordinary_addressed_contract_failure(
    monkeypatch,
):
    current, model, acs, history = _fixture()
    payload = _semantic_baseline()
    payload["reconciliation"] = [{
        "resolution_kind": "ordinary_addressed",
        "previous_target_handle": "previous_target_1",
        "source_quote": current.text,
    }]
    client = FakeClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setenv("HUMAN_EXPERIENCE_ENABLED", "true")

    with pytest.raises(CognitiveCoreError) as error:
        CognitiveCore(client).propose(current, model, acs, history)

    assert error.value.category == "validation_failure"
    assert "ordinary_addressed requires an OPEN_RESPONSE or CLARIFICATION" in str(
        error.value.__cause__
    )
    assert len(client.completions.calls) == 1


def test_frozen_recurring_mechanism_context_is_available_for_decision():
    """Input regression only: this does not assert live LLM policy compliance."""
    messages = (
        _message(
            DialogueRole.USER,
            "Несколько раз худел на 5–7 кг за полтора-два месяца.",
            1,
        ),
        _message(DialogueRole.SYSTEM, "Что обычно происходило дальше?", 2),
        _message(
            DialogueRole.USER,
            "После тяжёлой рабочей недели пропускал обед и вечером переедал.",
            3,
        ),
        _message(DialogueRole.SYSTEM, "Что значил для вас такой вечер?", 4),
        _message(
            DialogueRole.USER,
            "Считал один неудачный вечер провалом всей попытки и всё бросал на недели.",
            5,
        ),
        _message(DialogueRole.SYSTEM, "Это был разовый случай или повторялось?", 6),
        _message(
            DialogueRole.USER,
            "Почти каждый раз повторялся тот же сценарий.",
            7,
        ),
        _message(DialogueRole.SYSTEM, "Что происходило после первого срыва?", 8),
    )
    items = tuple(
        ModelItem(
            kind=kind,
            content=content,
            provenance=Provenance.USER_PROVIDED,
            status=ItemStatus.RECORDED,
            source_refs=(SourceReference(message_id=messages[index].id),),
            created_turn=1,
            updated_turn=4,
        )
        for kind, content, index in (
            (
                ModelItemKind.PREVIOUS_ATTEMPT,
                "несколько успешных стартов со снижением веса на 5–7 кг",
                0,
            ),
            (
                ModelItemKind.EXPERIENCE,
                "тяжёлая рабочая неделя, пропущенный обед и вечернее переедание",
                2,
            ),
            (
                ModelItemKind.BARRIER,
                "после одного неудачного вечера считал попытку проваленной и прекращал усилия",
                4,
            ),
            (
                ModelItemKind.USER_INTERPRETATION,
                "тот же механизм повторялся почти каждый раз",
                6,
            ),
        )
    )
    model = HumanModel(items={item.id: item for item in items}, relations={})
    current = _message(DialogueRole.USER, "Да, именно так и происходит.", 9)

    payload = CognitiveCore._input_view(
        current,
        model,
        None,
        DialogueHistory(messages=messages),
    )

    assert [item["content"] for item in payload["human_model"]["items"]] == [
        item.content for item in items
    ]
    assert [entry["text"] for entry in payload["dialogue_history"] if entry["sequence"] % 2] == [
        messages[index].text for index in (0, 2, 4, 6)
    ]
    assert payload["human_experience"] == {"enabled": False}


def test_knowledge_retrieval_flag_off_preserves_previous_production_assembly(
    monkeypatch,
):
    current, model, acs, history = _fixture()
    monkeypatch.delenv("HUMAN_EXPERIENCE_ENABLED", raising=False)
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "false")

    messages = CognitiveCore._messages(current, model, acs, history)
    payload = json.loads(messages[1]["content"])

    assert "professional_knowledge" not in payload
    assert PROFESSIONAL_KNOWLEDGE_SYSTEM_INSTRUCTION not in messages[0]["content"]
    assert payload["human_experience"]["enabled"] is True


def test_enabled_knowledge_retrieval_adds_separate_blocks_and_keeps_one_call(
    monkeypatch,
):
    _, model, acs, history = _fixture()
    current = _message(
        DialogueRole.USER,
        "После тяжёлой недели пропускаю обед, вечером переедаю и всё бросаю.",
        13,
    )
    core, client = _core()
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "true")
    monkeypatch.delenv("HUMAN_EXPERIENCE_ENABLED", raising=False)
    selection = SimpleNamespace(
        professional_input_view=lambda: {
            "enabled": True,
            "status": "pilot_not_expert_validated",
            "selected_cards": [{"id": "mechanism-1"}],
        },
        human_experience_input_view=lambda *, enabled: {
            "enabled": enabled,
            "selected_cards": [{"id": "P001"}],
            "diagnostics": {"indexed_human_experience_count": 30},
        },
    )
    monkeypatch.setattr(
        "ai.cognitive_core.retrieve_professional_context",
        lambda *args, **kwargs: selection,
    )

    result = core.propose(current, model, acs, history)

    payload = _payload(client)
    assert len(client.completions.calls) == 1
    assert payload["professional_knowledge"]["enabled"] is True
    assert payload["professional_knowledge"]["selected_cards"]
    assert payload["professional_knowledge"]["status"] == (
        "pilot_not_expert_validated"
    )
    assert payload["human_experience"]["enabled"] is True
    assert payload["human_experience"]["diagnostics"][
        "indexed_human_experience_count"
    ] == 30
    assert PROFESSIONAL_KNOWLEDGE_SYSTEM_INSTRUCTION in (
        client.completions.calls[0]["messages"][0]["content"]
    )
    assert result.state_patch.item_operations == ()


def test_retrieval_failure_logs_and_falls_back_to_previous_context_path(
    monkeypatch, caplog
):
    current, model, acs, history = _fixture()
    core, client = _core()
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "true")

    def fail(*args, **kwargs):
        raise KnowledgeRetrievalError("broken local index")

    monkeypatch.setattr("ai.cognitive_core.retrieve_professional_context", fail)

    core.propose(current, model, acs, history)

    payload = _payload(client)
    assert len(client.completions.calls) == 1
    assert "professional_knowledge" not in payload
    assert payload["human_experience"]["enabled"] is True
    assert "using the previous production context path" in caplog.text


def test_previous_acs_view_and_system_prompt_are_sent():
    current, model, acs, history = _fixture()
    core, client = _core()

    core.propose(current, model, acs, history)

    payload = _payload(client)
    assert payload["previous_active_conversation_state"]["active_content"][0]["id"] == next(iter(acs.active_content))
    previous_target = payload["previous_active_conversation_state"]["response_targets"][0]
    assert previous_target["handle"] == "previous_target_1"
    assert "id" not in previous_target
    assert previous_target["interaction"] == "evaluation"
    assert "один полный Cognitive Discovery turn" in client.completions.calls[0]["messages"][0]["content"]


def test_system_prompt_includes_critical_conditional_contract_rules():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "C. SEMANTIC OUTPUT CONTRACT",
        "ADD item/relation declares a new handle",
        "CORRECT declares a new handle AND explicitly names the existing object",
        "REINFORCE names only the existing object",
        "source_items, target_items and meaning",
        "AMBIGUOUS reconciliation names multiple distinct previous targets",
        "Every targeted content must be realized by a reply segment",
    ):
        assert instruction in prompt


def test_system_prompt_requires_shared_handles_and_semantic_previous_targets():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    assert "semantic handles" in prompt
    assert "одном общем namespace" in prompt
    assert "каждый объявленный handle уникален" in prompt
    assert "A handle is valid only for its declaration in this response" in prompt
    assert "Previous ACS ResponseTargets have input-scoped handles" in prompt
    assert "do not copy rt_ or aci_ storage IDs" in prompt
    assert "Python resolves the handle" in prompt
    assert "resolution_kind=ordinary_addressed" in prompt
    assert "previous_target_handle" in prompt
    assert "Do not choose an internal outcome" in prompt
    assert "OPEN_RESPONSE or CLARIFICATION target" in prompt
    assert "records ANSWERED" in prompt


def test_system_prompt_separates_target_support_from_item_reinforcement():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    assert "Reconciliation and persistent HumanModel evidence updates are separate" in prompt
    assert "does not automatically REINFORCE every ModelItem" in prompt
    assert "each specific existing item that the CURRENT USER message itself supports" in prompt
    assert "A SUPPORTED target may therefore have zero item operations" in prompt
    assert "multiple REINFORCE operations remain valid" in prompt


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


def test_system_prompt_ends_cognitive_discovery_after_transition_consent():
    prompt = " ".join(CognitiveCore._system_prompt().split()).casefold()
    for instruction in (
        "согласием на предыдущий SYSTEM_TRANSITION",
        "reconciliate его как CONSENTED",
        "не утверждай, что консультант уже уведомлён",
        "не спрашивай про канал, время или видеозвонок",
        "deterministic runtime отдельно запросит и обработает телефон",
    ):
        assert instruction.casefold() in prompt


def test_system_prompt_uses_simple_consultant_policy():
    prompt = " ".join(CognitiveCore._system_prompt().split())
    for principle in (
        "A. HOW TO CONDUCT THE CONSULTATION",
        "опытный дружелюбный консультант",
        "сам, внутренне реши, нужна ли дополнительная информация",
        "Уменьшение неопределённости системы само по себе не является ценностью",
        "Версия остаётся гипотезой, легко исправляется и не требует согласия",
        "Поправку пользователя спокойно прими",
        "существенно разные правдоподобные трактовки",
        "Отдельный эпизод может быть достаточным контекстом",
        "проверяй повторяемость только когда она действительно меняет",
        "существенный или повторяющийся mechanism",
        "REFLECTION соединяет существенные части пользовательской истории",
        "B. STRICT BEHAVIORAL PROHIBITIONS",
        "Не повторяй и не перефразируй механически ясное высказывание пользователя",
        "Не начинай обычный factual turn ритуальным",
        "Чувствительность темы сама по себе не оправдывает",
        "Вступление допустимо только если добавляет meaningful connection",
        "Один turn обычно содержит один focal conversational request",
        "Не превращай разговор в анкету",
        "expected consultation value",
        "Не исчерпывай её вопросами ради полноты",
        "разным working pictures или следующим действиям",
        "meaningful REFLECTION / RECOGNITION, ответ, STOP_EXPLORATION или TRANSITION",
        "не обязательная Reflection после фиксированного числа ходов",
        "Не задавай вопрос, на который разговор уже дал достаточный ответ",
        "C. SEMANTIC OUTPUT CONTRACT",
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


def test_system_prompt_has_one_ordered_policy_and_contract_structure():
    prompt = CognitiveCore._system_prompt()
    sections = (
        "A. HOW TO CONDUCT THE CONSULTATION",
        "B. STRICT BEHAVIORAL PROHIBITIONS",
        "C. SEMANTIC OUTPUT CONTRACT",
    )

    assert all(prompt.count(section) == 1 for section in sections)
    assert [prompt.index(section) for section in sections] == sorted(
        prompt.index(section) for section in sections
    )
    for obsolete_heading in (
        "CONSULTANT POLICY:",
        "PRODUCT BOUNDARY:",
        "WHAT YOU SHOULD DO:",
        "WHAT YOU MUST NOT DO:",
        "WHAT YOU SHOULD DO INSTEAD WHEN THE BOUNDARY IS REACHED:",
    ):
        assert obsolete_heading not in prompt


def test_system_prompt_prohibits_questionnaire_echo_and_automatic_reflection_confirmation():
    policy = " ".join(
        CognitiveCore._system_prompt().split("C. SEMANTIC OUTPUT CONTRACT", 1)[0].split()
    ).casefold()

    for invariant in (
        "не повторяй и не перефразируй механически ясное высказывание пользователя",
        "не проси подтвердить факт, который пользователь уже ясно сообщил",
        "не задавай вопрос, на который разговор уже дал достаточный ответ",
        "не превращай разговор в анкету",
        "не выноси наружу внутреннее рассуждение",
        "не используй gender-placeholder expressions",
        "не превращай reflection автоматически в просьбу подтвердить пересказ",
        "не выдумывай факты, причины, чувства, мотивы",
    ):
        assert invariant in policy
    assert "genuinely new consequential hypothesis" in policy
    assert "materially влияет на рабочее понимание или следующий decision" in policy


def test_consultation_policy_requires_professional_initiative_and_meaningful_delta():
    prompt = " ".join(
        CognitiveCore._system_prompt().split("C. SEMANTIC OUTPUT CONTRACT", 1)[0].split()
    ).casefold()
    source = " ".join(
        (
            Path(__file__).parents[1]
            / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
        ).read_text(encoding="utf-8").split()
    ).casefold()

    for policy in (prompt, source):
        assert "профессиональн" in policy and "инициатив" in policy
        assert "ещё не" in policy and "переживан" in policy
        assert "не проверенн" in policy and "не выдаётся за факт" in policy
        assert "mechanism" in policy and "personal meaning" in policy
        assert "повтор" in policy
        assert "недостаточно" in policy or "не являются достаточным" in policy
        assert "уже запросил объяснение" in policy
        assert "объясни её сейчас" in policy or "объяснить сейчас" in policy
        assert (
            "самому выбрать" in policy
            or "не перекладывай на человека выбор" in policy
        )
        assert "mechanism" in policy


def test_consultation_policy_separates_user_owned_fact_from_system_delta():
    prompt = " ".join(
        CognitiveCore._system_prompt().split("C. SEMANTIC OUTPUT CONTRACT", 1)[0].split()
    ).casefold()
    source = " ".join(
        (
            Path(__file__).parents[1]
            / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
        ).read_text(encoding="utf-8").split()
    ).casefold()

    for policy in (prompt, source):
        assert "что пользователь уже прямо сообщил" in policy or (
            "что человек уже прямо сообщил" in policy
        )
        assert "какой новый профессиональный смысл" in policy
        assert "существенная неизвестность" in policy
        assert "пересказ" in policy and "новую системную гипотезу" in policy
        assert "evaluation" in policy
        assert "разные" in policy and "working pictures" in policy
        assert "не созда" in policy and "responsetarget" in policy


def test_v2_policy_prioritizes_professional_value_support_and_memory():
    prompt = " ".join(CognitiveCore._system_prompt().split()).casefold()
    source = " ".join(
        (
            Path(__file__).parents[1]
            / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
        ).read_text(encoding="utf-8").split()
    ).casefold()

    for policy in (prompt, source):
        assert "профессиональн" in policy
        assert "поддерж" in policy
        assert "мотивац" in policy
        assert "желаем" in policy
        assert "довер" in policy and "себ" in policy
        assert "без жёсткой очередности" in policy or "не линейная лестница" in policy
    assert "сохраняй через statepatch" in prompt
    assert "если хотите, могу объяснить" in prompt
    assert "должны дойти до человека" in prompt
    assert "соответствующем responsetarget" in prompt
    assert "уверенность относится к профессиональному действию" in prompt
    assert "не возвращайся к уже объяснённой ветке" in prompt
    assert "отсутствие responsetarget" in prompt
    assert "закончи его декларативно" in prompt


def test_prompt_uses_contrastive_examples_without_turning_them_into_templates():
    prompt = " ".join(CognitiveCore._system_prompt().split()).casefold()

    assert "контрастные ориентиры" in prompt
    assert "не шаблоны обязательных фраз" in prompt
    assert "ощущение, что прежние усилия обнулились" in prompt
    assert "если гипотеза подтверждена" in prompt
    assert "если отвергнута, откажись от неё" in prompt
    assert "дисфункциональная когнитивная генерализация" in prompt
    assert "один трудный вечер начинает выглядеть как доказательство" in prompt


def test_consultation_value_precedes_nonrequested_transition():
    prompt = " ".join(
        CognitiveCore._system_prompt().split("C. SEMANTIC OUTPUT CONTRACT", 1)[0].split()
    ).casefold()
    source = " ".join(
        (
            Path(__file__).parents[1]
            / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
        ).read_text(encoding="utf-8").split()
    ).casefold()

    for policy in (prompt, source):
        assert "несколько mechanisms" in policy
        assert "спросил, что делать" in policy
        assert "профессиональн" in policy and "ценност" in policy
        assert "явное" in policy and "живому консультанту" in policy and "приоритет" in policy
        assert "практическ" in policy
        assert "формальн" in policy and "reflection" in policy and "не обязательн" in policy


def test_more_information_criterion_is_internal_not_user_facing_meta_choice():
    source = (
        Path(__file__).parents[1]
        / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
    ).read_text(encoding="utf-8")
    policy_sections = (
        CognitiveCore._system_prompt().split("C. SEMANTIC OUTPUT CONTRACT", 1)[0],
        source.split("VALUE / SUFFICIENCY ВНУТРИ DECISION", 1)[1]
        .split("PRODUCTION DECISION BOUNDARY", 1)[0],
    )
    for index, section in enumerate(policy_sections):
        policy = " ".join(section.casefold().split())
        assert "внутренне" in policy
        assert "нужна ли дополнительная информация" in policy
        if index == 0:
            assert "не выноси наружу" in policy
            assert "не спрашивай пользователя" in policy
            assert "нужно ли тебе знать больше, уточнять дальше или продолжать" in policy
        else:
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
        "не как привычное начало ходов",
        "Для любого DecisionIntent вступление",
        "один focal conversational request",
        "по ожидаемой консультационной ценности",
        "можно вернуться к более раннему пользовательскому материалу",
        "достаточно надёжно устанавливает повторяющийся mechanism",
        "существенно изменить рабочее понимание или следующий Decision",
        "не обязательная Reflection после фиксированного числа ходов",
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
    prompt = " ".join(CognitiveCore._system_prompt().split()).casefold()
    for guard in (
        "серьёзная медицинская или психологическая гипотеза не является фактом",
        "human experience предоставляет профессиональные working hypotheses",
        "не приписывай человеку скрытые мотивы",
        "не возвращай rejected version в иной формулировке",
        "bare yes при нескольких targets не подтверждает их все",
        "уважай коррекцию, отказ, паузу",
        "system_transition content and uses consent",
        "не составляй программу",
        "system_proposal не обходит эту границу",
        "соседний вариант после declined",
        "отказ от предложения сам по себе не доказывает barrier",
        "не добавляй к transition параллельное ai-предложение решения",
    ):
        assert guard.casefold() in prompt
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
        "A. HOW TO CONDUCT THE CONSULTATION",
        "Помогай человеку яснее увидеть собственную ситуацию",
        "Explanation помогает понять, что происходит, почему это важно",
        "B. STRICT BEHAVIORAL PROHIBITIONS",
        "Направление работы называет, ЧТО должно измениться и ПОЧЕМУ",
        "Даже одна конкретная практика или рекомендация",
        "Product Boundary проходит не перед любым практическим смыслом",
        "перечень options не становится допустимым",
        "Не составляй программу, подробный рацион",
        "Когда следующий полезный шаг требует выбрать, адаптировать или спроектировать",
        "Форма вопроса сама по себе не требует Transition",
        "предложи TRANSITION к живому консультанту",
        "не добавляй к Transition параллельное AI-предложение решения",
        "Не маскируй персональную программу",
    ):
        assert instruction in prompt


def test_direct_solution_question_keeps_explanation_inside_product_boundary():
    policy = " ".join(
        CognitiveCore._system_prompt().split("C. SEMANTIC OUTPUT CONTRACT", 1)[0].split()
    ).casefold()

    # The direct question has a useful, grounded answer route, not a forced handoff.
    assert "прямой вопрос пользователя имеет приоритет" in policy
    assert "grounded synthesis" in policy
    assert "descriptive explanation" in policy
    assert "форма вопроса сама по себе не требует transition" in policy
    assert "если на него можно честно ответить через понимание, объяснение" in policy

    # A bounded general orientation is allowed, but it cannot become an invented
    # user-specific cause or personalized solution design.
    assert "не выдумывай факты, причины, чувства, мотивы" in policy
    assert "направления профессиональной работы" in policy
    assert "остановись до выбора действий" in policy
    assert "не составляй программу, подробный рацион" in policy
    assert "когда следующий полезный шаг требует выбрать, адаптировать или спроектировать" in policy
    assert "предложи transition к живому консультанту с meaningful consent" in policy


def test_product_boundary_allows_bounded_help_but_rejects_personal_design():
    root = Path(__file__).parents[1] / "Research/Human_Experience/Weight"
    prompt = " ".join(
        CognitiveCore._system_prompt().split("C. SEMANTIC OUTPUT CONTRACT", 1)[0].split()
    ).casefold()
    boundaries = " ".join(
        (root / "11_BOUNDARIES.txt").read_text(encoding="utf-8").split()
    ).casefold()
    cycle = " ".join(
        (root / "12_Cognitive_Cycle.txt").read_text(encoding="utf-8").split()
    ).casefold()
    transition = " ".join(
        (root / "10_TRANSITION_TO_HUMAN_CONSULTANT.txt")
        .read_text(encoding="utf-8")
        .split()
    ).casefold()

    # Understanding and a non-prescriptive professional direction remain inside the product.
    assert "explanation помогает понять, что происходит, почему это важно" in prompt
    assert "descriptive explanation наблюдаемого mechanism" in prompt
    assert "профессиональная перспектива" in boundaries
    assert "не выбирает конкретную практику" in boundaries
    assert "направления профессиональной работы" in cycle
    assert "bounded answer может на этом закончиться" in transition
    assert "останавливается до выбора конкретной практики" in transition

    # Any concrete action recommendation, menu or adaptation remains outside; relabelling does not bypass the boundary.
    assert "даже одна конкретная практика" in prompt
    assert "конкретную рекомендацию" in boundaries
    assert "generic menu" in cycle and "конкретной рекомендации" in cycle
    assert "один такой action option" in transition
    for policy in (boundaries, cycle, transition):
        assert "общ" in policy and "принцип" in policy

    # Personal selection/adaptation/design still requires Transition.
    assert "не продолжай discovery только для обхода product boundary" in prompt
    assert "выбрать, адаптировать или спроектировать" in prompt
    assert "transition" in prompt and "meaningful consent" in prompt
    assert "transition с осмысленным согласием" in transition


def test_source_of_truth_separates_professional_direction_from_personal_design():
    root = Path(__file__).parents[1] / "Research/Human_Experience/Weight"
    boundaries = " ".join((root / "11_BOUNDARIES.txt").read_text(encoding="utf-8").split())
    cycle = " ".join((root / "12_Cognitive_Cycle.txt").read_text(encoding="utf-8").split())
    transition = " ".join(
        (root / "10_TRANSITION_TO_HUMAN_CONSULTANT.txt")
        .read_text(encoding="utf-8").split()
    )

    assert "не равны проектированию решения" in boundaries
    assert "не выбирает конкретную практику" in boundaries
    assert "Текущий AI не должен: - выбирать или проектировать систему привычек" in boundaries
    assert "Прямой вопрос пользователя не отменяет эту границу" in boundaries
    assert "Прямой вопрос не отменяет Product Boundary" in cycle
    assert "направления профессиональной работы" in cycle
    assert "не делая переход автоматическим по форме вопроса" in cycle
    assert "не подбирает программу питания или лечения" in cycle
    assert "останавливается до выбора конкретной" in transition
    assert "не должна обязательно" not in transition


def test_system_prompt_respects_declined_without_inventing_problem_barrier():
    prompt = " ".join(CognitiveCore._system_prompt().split()).casefold()

    for instruction in (
        "уважай коррекцию, отказ, паузу",
        "соседний вариант после declined",
        "отказ от предложения сам по себе не доказывает barrier",
    ):
        assert instruction.casefold() in prompt


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
        "Не маскируй персональную программу",
        "A CONSENT target subject must be current SYSTEM_PROPOSAL or SYSTEM_TRANSITION",
        "SYSTEM_REFLECTION is actually targeted for evaluation",
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


def test_production_request_without_acs_forbids_unavailable_reconciliation():
    current, model, acs, history = _fixture(with_acs=False)
    core, client = _core()

    core.propose(current, model, acs, history)

    request_schema = client.completions.calls[0]["response_format"]["json_schema"][
        "schema"
    ]
    assert request_schema == semantic_turn_json_schema(None)
    assert request_schema["properties"]["reconciliation"]["maxItems"] == 0


def test_user_correction_without_previous_acs_completes_and_is_preserved():
    first_user = _message(
        DialogueRole.USER,
        "После тяжёлой недели пропускаю обед, вечером переедаю и потом бросаю попытку.",
        1,
    )
    first_system = _message(
        DialogueRole.SYSTEM,
        "Рабочая гипотеза: один вечер ощущается как провал всей попытки.",
        2,
    )
    history = DialogueHistory(messages=(first_user, first_system))
    current = _message(
        DialogueRole.USER,
        "Нет, для меня один вечер не означает провал, и после него я обычно продолжаю.",
        3,
    )
    payload = _semantic_baseline()
    payload["state_patch"]["item_changes"] = [{
        "operation": "add",
        "handle": "explicit_correction",
        "kind": "user_interpretation",
        "content": "Один вечер не означает для пользователя провал; после него он обычно продолжает.",
        "evidence_origin": "current_user_material",
        "source_quote": current.text,
    }]
    payload["selected_action"] = {
        "decision_intent": "mechanism_discovery",
        "primary_content": {
            "handle": "next_useful_question",
            "kind": "system_question",
            "semantic_content": (
                "Выяснить, что действительно приводит к длительной паузе, "
                "если не один вечер с перееданием."
            ),
            "model_item_refs": [{"handle": "explicit_correction"}],
            "relation_refs": [],
        },
        "additional_contents": [],
        "response_targets": [{
            "owner_content": "next_useful_question",
            "subject": {"content_handle": "next_useful_question"},
            "interaction": "open_response",
        }],
    }
    payload["reply_segments"] = [{
        "text": (
            "Если один такой вечер не останавливает попытку, что обычно всё-таки "
            "приводит к паузе на несколько недель?"
        ),
        "realizes": ["next_useful_question"],
    }]
    client = FakeClient(json.dumps(payload, ensure_ascii=False))

    result = CognitiveCore(client).propose(current, HumanModel(), None, history)
    system = _message(
        DialogueRole.SYSTEM,
        "".join(segment.text for segment in result.reply_segments),
        4,
    )
    applied = apply_cognitive_turn(
        HumanModel(), None, history, current, system, result
    )

    assert result.reconciliation == ()
    assert result.decision_intent is DecisionIntent.MECHANISM_DISCOVERY
    assert applied.new_acs is not None
    assert next(iter(applied.new_acs.response_targets.values())).interaction is (
        TargetInteractionKind.OPEN_RESPONSE
    )
    assert any(
        "один вечер не означает" in item.content.casefold()
        and "обычно продолжает" in item.content.casefold()
        for item in applied.updated_human_model.items.values()
    )
    assert applied.updated_history.messages[-2:] == (current, system)


def test_prompt_distinguishes_unmaterialized_correction_from_acs_resolution():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    assert "If previous_active_conversation_state is null" in prompt
    assert "reconciliation and proposal_materializations MUST both be empty" in prompt
    assert "treat the USER's explicit correction as current evidence" in prompt
    assert "Do not invent an ACS target or silently discard the correction" in prompt


def test_correction_policy_separates_state_update_from_next_conversational_move():
    prompt = " ".join(CognitiveCore._system_prompt().split())
    source = " ".join(
        (
            Path(__file__).parents[1]
            / "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt"
        ).read_text(encoding="utf-8").split()
    )

    for policy in (prompt, source):
        assert "не определяют пользовательский ответ" in policy or (
            "не являются сами по себе следующим пользовательским ходом" in policy
        )
        assert "не должна расходовать весь ответ на подтверждение или пересказ поправки" in policy or (
            "Не расходуй весь turn на подтверждение или пересказ поправки" in policy
        )
        assert "по consultation value" in policy or (
            "по ожидаемой консультационной ценности" in policy
        )


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
            "previous_target_handles": ["previous_target_1"],
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


def test_capitalization_only_quote_is_canonicalized_to_current_user_text():
    text = (
        "Да, я уже много раз пытался похудеть. Сначала получается, "
        "а потом всё возвращается."
    )
    quote = "сначала получается, а потом всё возвращается"
    payload, current, model, acs, history = _quote_case("item", text, quote)
    original = deepcopy(payload)
    client = FakeClient(json.dumps(payload, ensure_ascii=False))

    result = CognitiveCore(client).propose(current, model, acs, history)

    assert payload == original
    span = result.state_patch.item_operations[0].source_span
    assert text[span.char_start:span.char_end] == (
        "Сначала получается, а потом всё возвращается"
    )
    assert len(client.completions.calls) == 1


def test_unique_exact_quote_remains_preferred_over_case_insensitive_candidates():
    text = "текст и Текст"
    quote = "текст"
    payload, current, model, acs, history = _quote_case("item", text, quote)
    client = FakeClient(json.dumps(payload, ensure_ascii=False))

    result = CognitiveCore(client).propose(current, model, acs, history)

    span = result.state_patch.item_operations[0].source_span
    assert (span.char_start, span.char_end) == (0, len(quote))
    assert text[span.char_start:span.char_end] == quote


def test_ambiguous_case_insensitive_quote_fails_closed():
    text = (
        "СНАЧАЛА получается, а потом всё возвращается. "
        "Сначала получается, а потом всё возвращается."
    )
    quote = "сначала получается, а потом всё возвращается"
    payload, current, model, acs, history = _quote_case("item", text, quote)
    client = FakeClient(json.dumps(payload, ensure_ascii=False))

    with pytest.raises(CognitiveCoreError) as error:
        CognitiveCore(client).propose(current, model, acs, history)

    assert error.value.category == "validation_failure"
    assert "case-insensitive matches=2" in str(error.value.__cause__.__cause__)
    assert len(client.completions.calls) == 1


def test_genuinely_incorrect_case_insensitive_quote_fails_closed():
    text = (
        "Да, я уже много раз пытался похудеть. Сначала получается, "
        "а потом всё возвращается."
    )
    quote = "сначала получается, но потом всё возвращается"
    payload, current, model, acs, history = _quote_case("item", text, quote)
    client = FakeClient(json.dumps(payload, ensure_ascii=False))

    with pytest.raises(CognitiveCoreError) as error:
        CognitiveCore(client).propose(current, model, acs, history)

    assert error.value.category == "validation_failure"
    assert "case-insensitive matches=0" in str(error.value.__cause__.__cause__)
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("quote", (
    "нет", "да", "aa", "", None, 123, "a b", "e\u0301", "Текст!", " Текст "
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
        "SEMANTIC WIRE", "SEMANTIC OUTPUT CONTRACT", "semantic handles",
        "exact existing_id", "ADD item/relation", "CORRECT",
        "REINFORCE", "source_items", "target_items", "reconciliation",
        "proposal_materializations", "primary_content", "owner_content",
        "subject", "interaction", "ReplySegment.realizes", "source_quote",
        "resolution_kind=ordinary_addressed", "previous_target_handle",
        "previous_target_handles", "resolution_target_handle",
        "do not copy rt_ or aci_ storage IDs",
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
