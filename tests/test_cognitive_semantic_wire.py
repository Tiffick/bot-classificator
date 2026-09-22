"""Offline tests for the semantic Cognitive Core wire boundary."""

import pytest

from ai.cognitive_semantic_wire import (
    SEMANTIC_TURN_JSON_SCHEMA,
    SemanticTurn,
    SemanticWireError,
    adapt_semantic_turn,
)
from ai.cognitive_turn import (
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    DecisionIntent,
    EvidenceOrigin,
    ItemOperation,
    ItemOperationKind,
    ItemReference,
    ProposalMaterialization,
    ReconciliationOutcome,
    RelationOperation,
    RelationOperationKind,
    RelationReference,
    ReplySegment,
    SourceSpan,
    StatePatch,
    SystemAction,
    TargetResolution,
)
from ai.discovery_data_model import (
    ActiveContentItem,
    ActiveContentKind,
    ActiveConversationState,
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


MI_A = "mi_" + "a" * 32
MI_B = "mi_" + "b" * 32
REL_A = "rel_" + "d" * 32
RT_A = "rt_" + "1" * 32
RT_B = "rt_" + "2" * 32
ACI_A = "aci_" + "c" * 32
ACI_B = "aci_" + "e" * 32
MSG_A = "msg_" + "3" * 32

SIMPLE_TEXT = (
    "Когда пробовал просто есть меньше, к вечеру после поздней смены я так "
    "сильно голодал, что снова ел больше, чем хотел."
)
SIMPLE_QUOTE = "к вечеру после поздней смены я так сильно голодал, что снова ел больше, чем хотел"
COMPLEX_TEXT = (
    "Нет, это не просто привычка: я раньше неточно сказал — после смены я как раз "
    "сильно голоден. Да, ужинаю поздно. Днём почти не успеваю поесть, и думаю, "
    "поэтому к вечеру голод такой сильный."
)
RELATION_QUOTE = "Днём почти не успеваю поесть, и думаю, поэтому к вечеру голод такой сильный"


def _span(message: str, quote: str) -> SourceSpan:
    start = message.index(quote)
    return SourceSpan(char_start=start, char_end=start + len(quote))


def _model(*, relation: bool = False) -> HumanModel:
    source = (SourceReference(message_id=MSG_A),)
    first = ModelItem(
        id=MI_A,
        kind=ModelItemKind.EXPERIENCE,
        content="После смены сильного голода нет.",
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.RECORDED,
        source_refs=source,
        created_turn=1,
        updated_turn=1,
    )
    second = ModelItem(
        id=MI_B,
        kind=ModelItemKind.LIFE_CHANGE,
        content="После смены обычно ужинаю поздно.",
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.RECORDED,
        source_refs=source,
        created_turn=1,
        updated_turn=1,
    )
    relations = {}
    if relation:
        old = Relation(
            id=REL_A,
            meaning="Старая связь",
            source_item_ids=(MI_A,),
            target_item_ids=(MI_B,),
            provenance=Provenance.USER_INTERPRETATION,
            status=ItemStatus.RECORDED,
            source_refs=source,
            created_turn=1,
            updated_turn=1,
        )
        relations[old.id] = old
    return HumanModel(items={MI_A: first, MI_B: second}, relations=relations)


def _acs(*, multiple: bool = False, proposal: bool = False) -> ActiveConversationState:
    content = ActiveContentItem(
        id=ACI_A,
        kind=ActiveContentKind.SYSTEM_PROPOSAL if proposal else ActiveContentKind.SYSTEM_QUESTION,
        content="Вечерняя еда после смены скорее привычка, чем голод.",
        source_message_id=MSG_A,
    )
    first = ResponseTarget(
        id=RT_A,
        active_content_id=ACI_A,
        subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=ACI_A),
        interaction=TargetInteractionKind.EVALUATION if proposal else TargetInteractionKind.OPEN_RESPONSE,
    )
    contents = {ACI_A: content}
    targets = {RT_A: first}
    if multiple:
        other = ActiveContentItem(
            id=ACI_B,
            kind=ActiveContentKind.SYSTEM_QUESTION,
            content="Когда после смены голод сильнее всего?",
            source_message_id=MSG_A,
        )
        contents[ACI_B] = other
        targets[RT_B] = ResponseTarget(
            id=RT_B,
            active_content_id=ACI_B,
            subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=ACI_B),
            interaction=TargetInteractionKind.OPEN_RESPONSE,
        )
    return ActiveConversationState(
        created_turn=1,
        source_system_message_id=MSG_A,
        active_content=contents,
        response_targets=targets,
    )


def _simple() -> dict:
    return {
        "state_patch": {
            "item_changes": [{
                "operation": "add", "handle": "evening_hunger", "kind": "barrier",
                "content": "Сильный вечерний голод мешал попытке есть меньше.",
                "evidence_origin": "current_user_material", "source_quote": SIMPLE_QUOTE,
            }],
            "relation_changes": [],
            "proposal_materializations": [],
        },
        "reconciliation": [{
            "previous_target_ids": [RT_A], "outcome": "answered", "source_quote": SIMPLE_TEXT,
        }],
        "selected_action": {
            "decision_intent": "human_discovery",
            "primary_content": {
                "handle": "next_question", "kind": "system_question",
                "semantic_content": "Выяснить, что в этом опыте было самым трудным.",
                "model_item_refs": [{"existing_id": MI_A}, {"handle": "evening_hunger"}],
                "relation_refs": [],
            },
            "additional_contents": [],
            "response_targets": [{
                "owner_content": "next_question",
                "subject": {"content_handle": "next_question"},
                "interaction": "open_response",
            }],
        },
        "reply_segments": [{
            "text": "Что в этом для вас было самым трудным?",
            "realizes": ["next_question"],
        }],
    }


def _complex() -> dict:
    return {
        "state_patch": {
            "item_changes": [
                {
                    "operation": "correct", "handle": "evening_hunger", "existing_id": MI_A,
                    "kind": "experience", "content": "После смены пользователь сильно голоден.",
                    "evidence_origin": "current_user_material",
                    "source_quote": "после смены я как раз сильно голоден",
                },
                {
                    "operation": "reinforce", "existing_id": MI_B,
                    "evidence_origin": "current_user_material",
                    "source_quote": "Да, ужинаю поздно",
                },
                {
                    "operation": "add", "handle": "daytime_gap", "kind": "barrier",
                    "content": "Днём пользователь почти не успевает поесть.",
                    "evidence_origin": "current_user_material",
                    "source_quote": "Днём почти не успеваю поесть",
                },
            ],
            "relation_changes": [{
                "operation": "add", "handle": "daytime_to_evening",
                "source_items": [{"handle": "daytime_gap"}],
                "target_items": [{"handle": "evening_hunger"}],
                "meaning": "Пользователь предполагает связь дневного пропуска еды с вечерним голодом.",
                "evidence_origin": "current_user_interpretation",
                "source_quote": RELATION_QUOTE,
            }],
            "proposal_materializations": [{
                "subject_kind": "item", "resolution_target_id": RT_A,
                "previous_active_content_id": ACI_A, "kind": "experience",
                "content": "Вечерняя еда после смены скорее привычка, чем голод.",
            }],
        },
        "reconciliation": [
            {
                "previous_target_ids": [RT_A], "outcome": "rejected",
                "source_quote": "Нет, это не просто привычка",
            },
            {
                "previous_target_ids": [RT_B], "outcome": "answered",
                "source_quote": "к вечеру голод такой сильный",
            },
        ],
        "selected_action": {
            "decision_intent": "reflection",
            "primary_content": {
                "handle": "working_picture", "kind": "system_reflection",
                "semantic_content": "Голод после смены и возможная связь с пропуском еды днём.",
                "model_item_refs": [
                    {"handle": "evening_hunger"}, {"existing_id": MI_B},
                    {"handle": "daytime_gap"},
                ],
                "relation_refs": [{"handle": "daytime_to_evening"}],
            },
            "additional_contents": [{
                "handle": "evaluation_question", "kind": "system_question",
                "semantic_content": "Проверить собранную картину у пользователя.",
                "model_item_refs": [], "relation_refs": [],
            }],
            "response_targets": [{
                "owner_content": "evaluation_question",
                "subject": {"content_handle": "working_picture"},
                "interaction": "evaluation",
            }],
        },
        "reply_segments": [
            {"text": "Похоже, голод после смены связан с тем, что днём не успеваете поесть. ",
             "realizes": ["working_picture"]},
            {"text": "Похоже ли это на ваш опыт?", "realizes": ["evaluation_question"]},
        ],
    }


def _adapt(payload: dict, *, text: str = SIMPLE_TEXT, model=None, acs=None):
    return adapt_semantic_turn(
        payload,
        current_user_text=text,
        human_model=_model() if model is None else model,
        previous_acs=_acs() if acs is None else acs,
    )


def test_simple_turn_is_lossless_against_existing_internal_contract():
    result = _adapt(_simple())
    expected = CognitiveTurnResult(
        decision_intent=DecisionIntent.HUMAN_DISCOVERY,
        state_patch=StatePatch(item_operations=(ItemOperation(
            operation=ItemOperationKind.ADD,
            local_id="item:1",
            kind=ModelItemKind.BARRIER,
            content="Сильный вечерний голод мешал попытке есть меньше.",
            evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
            source_span=_span(SIMPLE_TEXT, SIMPLE_QUOTE),
        ),)),
        reconciliation=(TargetResolution(
            previous_response_target_ids=(RT_A,),
            outcome=ReconciliationOutcome.ANSWERED,
            source_span=_span(SIMPLE_TEXT, SIMPLE_TEXT),
        ),),
        system_action=SystemAction(
            contents=(ActionContent(
                local_id="content:1", kind=ActiveContentKind.SYSTEM_QUESTION,
                semantic_content="Выяснить, что в этом опыте было самым трудным.",
                model_item_refs=(
                    ItemReference(existing_item_id=MI_A),
                    ItemReference(local_item_id="item:1"),
                ),
            ),),
            response_targets=(ActionTarget(
                local_id="target:1", active_content_local_id="content:1",
                subject=ActionSubjectReference(
                    kind=TargetSubjectKind.ACTIVE_CONTENT,
                    active_content_local_id="content:1",
                ),
                interaction=TargetInteractionKind.OPEN_RESPONSE,
            ),),
        ),
        reply_segments=(ReplySegment(
            local_id="segment:1", text="Что в этом для вас было самым трудным?",
            realizes_action_content_ids=("content:1",),
        ),),
    )
    assert result == expected


def test_complex_turn_is_lossless_against_existing_internal_contract():
    result = _adapt(_complex(), text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))
    expected = CognitiveTurnResult(
        decision_intent=DecisionIntent.REFLECTION,
        state_patch=StatePatch(
            item_operations=(
                ItemOperation(
                    operation=ItemOperationKind.CORRECT, local_id="item:1",
                    existing_item_id=MI_A, kind=ModelItemKind.EXPERIENCE,
                    content="После смены пользователь сильно голоден.",
                    evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                    source_span=_span(COMPLEX_TEXT, "после смены я как раз сильно голоден"),
                ),
                ItemOperation(
                    operation=ItemOperationKind.REINFORCE, existing_item_id=MI_B,
                    evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                    source_span=_span(COMPLEX_TEXT, "Да, ужинаю поздно"),
                ),
                ItemOperation(
                    operation=ItemOperationKind.ADD, local_id="item:2",
                    kind=ModelItemKind.BARRIER,
                    content="Днём пользователь почти не успевает поесть.",
                    evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                    source_span=_span(COMPLEX_TEXT, "Днём почти не успеваю поесть"),
                ),
            ),
            relation_operations=(RelationOperation(
                operation=RelationOperationKind.ADD, local_id="relation:1",
                source_items=(ItemReference(local_item_id="item:2"),),
                target_items=(ItemReference(local_item_id="item:1"),),
                meaning="Пользователь предполагает связь дневного пропуска еды с вечерним голодом.",
                evidence_origin=EvidenceOrigin.CURRENT_USER_INTERPRETATION,
                source_span=_span(COMPLEX_TEXT, RELATION_QUOTE),
            ),),
            proposal_materializations=(ProposalMaterialization(
                subject_kind="item", resolution_target_id=RT_A,
                previous_active_content_id=ACI_A, kind=ModelItemKind.EXPERIENCE,
                content="Вечерняя еда после смены скорее привычка, чем голод.",
            ),),
        ),
        reconciliation=(
            TargetResolution(
                previous_response_target_ids=(RT_A,), outcome=ReconciliationOutcome.REJECTED,
                source_span=_span(COMPLEX_TEXT, "Нет, это не просто привычка"),
            ),
            TargetResolution(
                previous_response_target_ids=(RT_B,), outcome=ReconciliationOutcome.ANSWERED,
                source_span=_span(COMPLEX_TEXT, "к вечеру голод такой сильный"),
            ),
        ),
        system_action=SystemAction(
            contents=(
                ActionContent(
                    local_id="content:1", kind=ActiveContentKind.SYSTEM_REFLECTION,
                    semantic_content="Голод после смены и возможная связь с пропуском еды днём.",
                    model_item_refs=(
                        ItemReference(local_item_id="item:1"),
                        ItemReference(existing_item_id=MI_B),
                        ItemReference(local_item_id="item:2"),
                    ),
                    relation_refs=(RelationReference(local_relation_id="relation:1"),),
                ),
                ActionContent(
                    local_id="content:2", kind=ActiveContentKind.SYSTEM_QUESTION,
                    semantic_content="Проверить собранную картину у пользователя.",
                ),
            ),
            response_targets=(ActionTarget(
                local_id="target:1", active_content_local_id="content:2",
                subject=ActionSubjectReference(
                    kind=TargetSubjectKind.ACTIVE_CONTENT,
                    active_content_local_id="content:1",
                ),
                interaction=TargetInteractionKind.EVALUATION,
            ),),
        ),
        reply_segments=(
            ReplySegment(
                local_id="segment:1",
                text="Похоже, голод после смены связан с тем, что днём не успеваете поесть. ",
                realizes_action_content_ids=("content:1",),
            ),
            ReplySegment(
                local_id="segment:2", text="Похоже ли это на ваш опыт?",
                realizes_action_content_ids=("content:2",),
            ),
        ),
    )
    assert result == expected


def test_duplicate_unknown_and_wrong_type_handles_fail_closed():
    duplicate = _simple()
    duplicate["selected_action"]["primary_content"]["handle"] = "evening_hunger"
    with pytest.raises(SemanticWireError, match="Duplicate"):
        _adapt(duplicate)

    unknown = _simple()
    unknown["selected_action"]["primary_content"]["model_item_refs"][1] = {"handle": "missing"}
    with pytest.raises(SemanticWireError, match="Unknown semantic handle"):
        _adapt(unknown)

    wrong = _complex()
    wrong["selected_action"]["primary_content"]["model_item_refs"][0] = {
        "handle": "daytime_to_evening"
    }
    with pytest.raises(SemanticWireError, match="Wrong-type"):
        _adapt(wrong, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))

    reverse = _complex()
    reverse["selected_action"]["primary_content"]["relation_refs"][0] = {
        "handle": "daytime_gap"
    }
    with pytest.raises(SemanticWireError, match="Wrong-type"):
        _adapt(reverse, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))


def test_invalid_existing_id_and_unknown_existing_reference_fail_closed():
    malformed = _simple()
    malformed["selected_action"]["primary_content"]["model_item_refs"][0] = {
        "existing_id": "mi_not-a-uuid"
    }
    with pytest.raises(SemanticWireError):
        _adapt(malformed)
    unknown = _simple()
    unknown["selected_action"]["primary_content"]["model_item_refs"][0] = {
        "existing_id": "mi_" + "f" * 32
    }
    with pytest.raises(SemanticWireError, match="Unknown existing"):
        _adapt(unknown)


@pytest.mark.parametrize("quote", ["not in the message", "Да"])
def test_missing_or_repeated_exact_quote_fails_closed(quote):
    payload = _simple()
    payload["state_patch"]["item_changes"][0]["source_quote"] = quote
    text = SIMPLE_TEXT if quote != "Да" else "Да. Да."
    with pytest.raises(SemanticWireError):
        _adapt(payload, text=text)


def test_quote_is_required_for_state_operations():
    payload = _simple()
    del payload["state_patch"]["item_changes"][0]["source_quote"]
    with pytest.raises(SemanticWireError):
        _adapt(payload)


def test_relation_endpoints_owner_subject_and_segments_are_not_inferred():
    payload = _complex()
    payload["state_patch"]["relation_changes"][0]["source_items"], payload["state_patch"]["relation_changes"][0]["target_items"] = (
        payload["state_patch"]["relation_changes"][0]["target_items"],
        payload["state_patch"]["relation_changes"][0]["source_items"],
    )
    payload["selected_action"]["response_targets"][0]["owner_content"] = "working_picture"
    payload["reply_segments"][0]["realizes"] = ["working_picture", "evaluation_question"]
    result = _adapt(payload, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))
    assert result.state_patch.relation_operations[0].source_items[0].local_item_id == "item:1"
    assert result.state_patch.relation_operations[0].target_items[0].local_item_id == "item:2"
    assert result.system_action.response_targets[0].active_content_local_id == "content:1"
    assert result.system_action.response_targets[0].subject.active_content_local_id == "content:1"
    assert result.reply_segments[0].realizes_action_content_ids == ("content:1", "content:2")


def test_relation_reinforce_and_correct_preserve_existing_and_replacement():
    payload = _simple()
    payload["state_patch"]["relation_changes"] = [
        {
            "operation": "reinforce", "existing_id": REL_A,
            "evidence_origin": "current_user_material", "source_quote": SIMPLE_QUOTE,
        },
        {
            "operation": "correct", "handle": "corrected_relation", "existing_id": REL_A,
            "source_items": [{"existing_id": MI_B}],
            "target_items": [{"handle": "evening_hunger"}],
            "meaning": "Новая направленная связь",
            "evidence_origin": "current_user_interpretation", "source_quote": SIMPLE_QUOTE,
        },
    ]
    result = _adapt(payload, model=_model(relation=True))
    assert result.state_patch.relation_operations[0].operation == RelationOperationKind.REINFORCE
    assert result.state_patch.relation_operations[0].local_id is None
    corrected = result.state_patch.relation_operations[1]
    assert corrected.operation == RelationOperationKind.CORRECT
    assert corrected.existing_relation_id == REL_A
    assert corrected.local_id == "relation:1"
    assert corrected.source_items == (ItemReference(existing_item_id=MI_B),)
    assert corrected.target_items == (ItemReference(local_item_id="item:1"),)


def test_ambiguous_reconciliation_preserves_both_targets_without_materialization():
    payload = _simple()
    payload["reconciliation"] = [{
        "previous_target_ids": [RT_A, RT_B], "outcome": "ambiguous", "source_quote": None,
    }]
    result = _adapt(payload, acs=_acs(multiple=True))
    assert result.reconciliation[0].previous_response_target_ids == (RT_A, RT_B)
    assert result.reconciliation[0].outcome == ReconciliationOutcome.AMBIGUOUS
    assert result.reconciliation[0].source_span is None
    assert not result.state_patch.proposal_materializations


def test_relation_proposal_materialization_preserves_explicit_endpoints():
    payload = _complex()
    payload["state_patch"]["proposal_materializations"] = [{
        "subject_kind": "relation", "resolution_target_id": RT_A,
        "previous_active_content_id": ACI_A,
        "source_items": [{"existing_id": MI_B}],
        "target_items": [{"handle": "evening_hunger"}],
        "meaning": "Системная версия связи",
    }]
    result = _adapt(payload, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))
    materialization = result.state_patch.proposal_materializations[0]
    assert materialization.subject_kind == "relation"
    assert materialization.source_items == (ItemReference(existing_item_id=MI_B),)
    assert materialization.target_items == (ItemReference(local_item_id="item:1"),)


def test_multiple_response_targets_keep_order_and_semantic_subjects():
    payload = _complex()
    payload["selected_action"]["response_targets"].append({
        "owner_content": "working_picture",
        "subject": {"item": {"handle": "evening_hunger"}},
        "interaction": "clarification",
    })
    result = _adapt(payload, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))
    assert [target.local_id for target in result.system_action.response_targets] == [
        "target:1", "target:2"
    ]
    assert result.system_action.response_targets[1].subject.item_reference.local_item_id == "item:1"


@pytest.mark.parametrize(
    "intent,kind,interaction",
    [
        ("respect_pause_or_refusal", None, None),
        ("stop_exploration", None, None),
        ("transition", "system_transition", "consent"),
    ],
)
def test_pause_stop_and_transition_consent(intent, kind, interaction):
    payload = _simple()
    payload["state_patch"]["item_changes"] = []
    payload["reconciliation"] = []
    payload["selected_action"]["decision_intent"] = intent
    if kind is None:
        payload["selected_action"].update(
            primary_content=None, additional_contents=[], response_targets=[]
        )
        payload["reply_segments"][0]["realizes"] = []
    else:
        payload["selected_action"]["primary_content"].update(
            kind=kind, model_item_refs=[]
        )
        payload["selected_action"]["response_targets"][0]["interaction"] = interaction
    result = _adapt(payload)
    assert result.decision_intent.value == intent
    assert len(result.system_action.response_targets) == (1 if interaction else 0)


def test_typed_primary_and_invalid_materialization_fail_closed():
    payload = _complex()
    payload["selected_action"]["primary_content"]["kind"] = "system_question"
    with pytest.raises(SemanticWireError):
        _adapt(payload, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))
    payload = _complex()
    payload["state_patch"]["proposal_materializations"][0]["resolution_target_id"] = RT_B
    with pytest.raises(SemanticWireError):
        _adapt(payload, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))


def test_previous_target_and_content_are_explicit_and_checked():
    payload = _complex()
    payload["reconciliation"][0]["previous_target_ids"] = ["rt_" + "f" * 32]
    with pytest.raises(SemanticWireError, match="Unknown previous response target"):
        _adapt(payload, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))

    payload = _complex()
    payload["state_patch"]["proposal_materializations"][0][
        "previous_active_content_id"
    ] = ACI_B
    with pytest.raises(SemanticWireError, match="target/content mismatch"):
        _adapt(payload, text=COMPLEX_TEXT, acs=_acs(multiple=True, proposal=True))


@pytest.mark.parametrize(
    "intent,kind",
    [("reflection", "system_reflection"), ("recognition", "recognition_option"),
     ("transition", "system_transition")],
)
def test_typed_primary_is_explicit_for_each_typed_decision(intent, kind):
    payload = _simple()
    payload["selected_action"]["decision_intent"] = intent
    payload["selected_action"]["primary_content"]["kind"] = kind
    if intent == "transition":
        payload["selected_action"]["response_targets"][0]["interaction"] = "consent"
    result = _adapt(payload)
    assert result.system_action.contents[0].kind.value == kind
    payload["selected_action"]["primary_content"]["kind"] = "system_statement"
    with pytest.raises(SemanticWireError):
        _adapt(payload)


def test_generated_schema_is_strict_production_schema():
    schema = SEMANTIC_TURN_JSON_SCHEMA
    assert schema["type"] == "object" and "anyOf" not in schema
    assert set(schema["properties"]) == {
        "state_patch", "reconciliation", "selected_action", "reply_segments"
    }

    branch_refs = schema["properties"]["selected_action"]["anyOf"]
    branches = [schema["$defs"][row["$ref"].rsplit("/", 1)[-1]] for row in branch_refs]
    assert len(branches) == 8
    primary_kinds = {
        branch["properties"]["decision_intent"]["const"]:
        schema["$defs"][branch["properties"]["primary_content"]["$ref"].rsplit("/", 1)[-1]]["properties"]["kind"]["const"]
        for branch in branches
        if branch["properties"]["decision_intent"]["const"]
        in {"reflection", "recognition", "transition"}
    }
    assert primary_kinds == {
        "reflection": "system_reflection",
        "recognition": "recognition_option",
        "transition": "system_transition",
    }

    def walk(node):
        if isinstance(node, dict):
            assert "default" not in node
            assert "oneOf" not in node
            assert "discriminator" not in node
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert len(schema.get("$defs", {})) > 0
    assert SemanticTurn.model_validate(_simple())
