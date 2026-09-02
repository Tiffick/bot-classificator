import json
from typing import List, Optional
from uuid import uuid4

import pytest
from pydantic import ValidationError

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
    validate_acs_human_model_references,
)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _source(message_id: Optional[str] = None) -> SourceReference:
    return SourceReference(message_id=message_id or _id("msg"))


def _item(
    *,
    item_id: Optional[str] = None,
    kind: ModelItemKind = ModelItemKind.EXPERIENCE,
    provenance: Provenance = Provenance.USER_PROVIDED,
    status: ItemStatus = ItemStatus.RECORDED,
    supersedes_item_ids: Optional[List[str]] = None,
) -> ModelItem:
    return ModelItem(
        id=item_id or _id("mi"),
        kind=kind,
        content="нормализованный смысл пользовательского материала",
        provenance=provenance,
        status=status,
        source_refs=[_source()],
        supersedes_item_ids=supersedes_item_ids or [],
        created_turn=1,
        updated_turn=1,
    )


def _relation(
    source_item_id: str,
    target_item_id: str,
    *,
    relation_id: Optional[str] = None,
    supersedes_relation_ids: Optional[List[str]] = None,
) -> Relation:
    return Relation(
        id=relation_id or _id("rel"),
        meaning="пользователь воспринимает эти два изменения как связанные",
        source_item_ids=[source_item_id],
        target_item_ids=[target_item_id],
        provenance=Provenance.USER_INTERPRETATION,
        status=ItemStatus.RECORDED,
        source_refs=[_source()],
        supersedes_relation_ids=supersedes_relation_ids or [],
        created_turn=1,
        updated_turn=1,
    )


def _active_content(
    *,
    content_id: Optional[str] = None,
    kind: ActiveContentKind = ActiveContentKind.SYSTEM_QUESTION,
    model_item_ids: Optional[List[str]] = None,
    relation_ids: Optional[List[str]] = None,
) -> ActiveContentItem:
    return ActiveContentItem(
        id=content_id or _id("aci"),
        kind=kind,
        content="адресуемый смысл последнего системного ответа",
        model_item_ids=model_item_ids or [],
        relation_ids=relation_ids or [],
        source_message_id=_id("msg"),
    )


def _target(
    active_content_id: str,
    *,
    target_id: Optional[str] = None,
    subject: Optional[TargetReference] = None,
    interaction: TargetInteractionKind = TargetInteractionKind.EVALUATION,
) -> ResponseTarget:
    return ResponseTarget(
        id=target_id or _id("rt"),
        active_content_id=active_content_id,
        subject=subject
        or TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=active_content_id),
        interaction=interaction,
    )


def _acs(
    content: List[ActiveContentItem], targets: List[ResponseTarget]
) -> ActiveConversationState:
    return ActiveConversationState(
        id=_id("acs"),
        created_turn=1,
        source_system_message_id=_id("msg"),
        active_content={item.id: item for item in content},
        response_targets={target.id: target for target in targets},
    )


def test_direct_user_material_round_trip_preserves_recorded_status():
    item = _item()
    model = HumanModel(items={item.id: item})

    restored = HumanModel.model_validate_json(model.model_dump_json())

    assert restored.items[item.id].provenance is Provenance.USER_PROVIDED
    assert restored.items[item.id].status is ItemStatus.RECORDED


@pytest.mark.parametrize(
    ("status",),
    [
        (ItemStatus.SUPPORTED,),
        (ItemStatus.REJECTED,),
        (ItemStatus.UNCERTAIN,),
    ],
)
def test_system_proposal_preserves_epistemic_status(status):
    item = _item(provenance=Provenance.SYSTEM_PROPOSED, status=status)
    model = HumanModel(items={item.id: item})

    restored = HumanModel.model_validate(json.loads(model.model_dump_json()))

    assert restored.items[item.id].provenance is Provenance.SYSTEM_PROPOSED
    assert restored.items[item.id].status is status


def test_correction_preserves_old_item_and_supersession_reference():
    old_item = _item(status=ItemStatus.CORRECTED)
    new_item = _item(supersedes_item_ids=[old_item.id])

    model = HumanModel(items={old_item.id: old_item, new_item.id: new_item})
    restored = HumanModel.model_validate_json(model.model_dump_json())

    assert restored.items[old_item.id].status is ItemStatus.CORRECTED
    assert restored.items[new_item.id].supersedes_item_ids == [old_item.id]


def test_relation_uses_arbitrary_meaning_without_relation_enum():
    attempt = _item(kind=ModelItemKind.PREVIOUS_ATTEMPT)
    hunger = _item(kind=ModelItemKind.BARRIER)
    relation = _relation(hunger.id, attempt.id)
    relation.meaning = (
        "пользователь считает вечерний голод основной причиной срыва "
        "предыдущей попытки есть меньше"
    )

    model = HumanModel(
        items={attempt.id: attempt, hunger.id: hunger},
        relations={relation.id: relation},
    )

    assert model.relations[relation.id].meaning == relation.meaning


def test_relation_supersession_requires_existing_relation():
    first = _item()
    second = _item()
    old_relation = _relation(first.id, second.id)
    new_relation = _relation(
        first.id,
        second.id,
        supersedes_relation_ids=[old_relation.id],
    )

    model = HumanModel(
        items={first.id: first, second.id: second},
        relations={old_relation.id: old_relation, new_relation.id: new_relation},
    )

    assert model.relations[new_relation.id].supersedes_relation_ids == [old_relation.id]


def test_human_model_rejects_missing_superseded_relation():
    first = _item()
    second = _item()
    relation = _relation(
        first.id,
        second.id,
        supersedes_relation_ids=[_id("rel")],
    )

    with pytest.raises(ValidationError, match="supersedes_relation_ids"):
        HumanModel(
            items={first.id: first, second.id: second},
            relations={relation.id: relation},
        )


def test_partial_reflection_uses_three_independent_targets_without_resolution():
    content = [_active_content(kind=ActiveContentKind.SYSTEM_REFLECTION) for _ in range(3)]
    targets = [_target(item.id) for item in content]

    active_state = _acs(content, targets)
    serialised_target = active_state.response_targets[targets[0].id].model_dump()

    assert len(active_state.active_content) == 3
    assert len(active_state.response_targets) == 3
    assert all(target.interaction is TargetInteractionKind.EVALUATION for target in targets)
    assert "resolution" not in serialised_target


def test_recognition_options_are_independent_addressable_targets():
    content = [
        _active_content(kind=ActiveContentKind.RECOGNITION_OPTION) for _ in range(3)
    ]
    targets = [_target(item.id) for item in content]

    active_state = _acs(content, targets)

    assert {
        target.active_content_id for target in active_state.response_targets.values()
    } == {item.id for item in content}


def test_transition_uses_consent_without_assuming_a_result():
    content = _active_content(kind=ActiveContentKind.SYSTEM_TRANSITION)
    target = _target(
        content.id,
        interaction=TargetInteractionKind.CONSENT,
    )

    active_state = _acs([content], [target])

    assert active_state.response_targets[target.id].interaction is TargetInteractionKind.CONSENT
    assert "resolution" not in active_state.model_dump()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DialogueMessage(role=DialogueRole.USER, text="текст", sequence=1),
        lambda: _item(),
        lambda: _relation(_id("mi"), _id("mi")),
        lambda: _active_content(),
        lambda: _target(_id("aci")),
        lambda: (lambda content: _acs([content], [_target(content.id)]))(_active_content()),
    ],
)
def test_generated_ids_are_stable_prefixed_uuids(factory):
    value = factory()
    identifier = value.id
    prefix = identifier.split("_", maxsplit=1)[0]

    assert prefix in {"msg", "mi", "rel", "aci", "rt", "acs"}
    assert uuid4().__class__(identifier.removeprefix(f"{prefix}_"))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "id": _id("mi"),
            "kind": "experience",
            "content": "материал",
            "provenance": "user_provided",
            "status": "recorded",
            "source_refs": [{"message_id": _id("msg")}],
            "created_turn": 1,
            "updated_turn": 1,
            "unexpected": True,
        },
        {
            "id": _id("rt"),
            "active_content_id": _id("aci"),
            "subject": {"kind": "active_content", "id": _id("aci")},
            "interaction": "evaluation",
            "resolution": "supported",
        },
    ],
)
def test_extra_fields_are_forbidden(payload):
    model_class = ModelItem if payload["id"].startswith("mi_") else ResponseTarget

    with pytest.raises(ValidationError):
        model_class.model_validate(payload)


def test_human_model_rejects_missing_relation_item_reference():
    item = _item()
    relation = _relation(item.id, _id("mi"))

    with pytest.raises(ValidationError, match="Relation item references"):
        HumanModel(items={item.id: item}, relations={relation.id: relation})


def test_human_model_rejects_missing_superseded_item():
    item = _item(supersedes_item_ids=[_id("mi")])

    with pytest.raises(ValidationError, match="supersedes_item_ids"):
        HumanModel(items={item.id: item})


def test_human_model_rejects_dict_key_that_differs_from_item_id():
    item = _item()

    with pytest.raises(ValidationError, match="item key"):
        HumanModel(items={_id("mi"): item})


def test_human_model_rejects_dict_key_that_differs_from_relation_id():
    first = _item()
    second = _item()
    relation = _relation(first.id, second.id)

    with pytest.raises(ValidationError, match="relation key"):
        HumanModel(
            items={first.id: first, second.id: second},
            relations={_id("rel"): relation},
        )


def test_active_state_rejects_target_for_missing_active_content():
    content = _active_content()
    target = _target(_id("aci"))

    with pytest.raises(ValidationError, match="active_content_id"):
        _acs([content], [target])


def test_active_state_rejects_dict_keys_that_do_not_match_object_ids():
    content = _active_content()
    target = _target(content.id)

    with pytest.raises(ValidationError, match="content key"):
        ActiveConversationState(
            id=_id("acs"),
            created_turn=1,
            source_system_message_id=_id("msg"),
            active_content={_id("aci"): content},
            response_targets={target.id: target},
        )

    with pytest.raises(ValidationError, match="target key"):
        ActiveConversationState(
            id=_id("acs"),
            created_turn=1,
            source_system_message_id=_id("msg"),
            active_content={content.id: content},
            response_targets={_id("rt"): target},
        )


def test_active_content_target_reference_must_exist_in_same_acs():
    content = _active_content()
    target = _target(
        content.id,
        subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=_id("aci")),
    )

    with pytest.raises(ValidationError, match="ACTIVE_CONTENT TargetReference"):
        _acs([content], [target])


def test_cross_object_acs_references_are_validated_against_human_model():
    item = _item()
    other_item = _item()
    relation = _relation(item.id, other_item.id)
    model = HumanModel(
        items={item.id: item, other_item.id: other_item},
        relations={relation.id: relation},
    )
    content = _active_content(model_item_ids=[item.id], relation_ids=[relation.id])
    target = _target(
        content.id,
        subject=TargetReference(kind=TargetSubjectKind.RELATION, id=relation.id),
    )
    active_state = _acs([content], [target])

    validate_acs_human_model_references(active_state, model)

    missing_target = _target(
        content.id,
        subject=TargetReference(kind=TargetSubjectKind.MODEL_ITEM, id=_id("mi")),
    )
    missing_state = _acs([content], [missing_target])

    with pytest.raises(ValueError, match="MODEL_ITEM TargetReference"):
        validate_acs_human_model_references(missing_state, model)


def test_dialogue_history_round_trip_uses_stable_message_references():
    message = DialogueMessage(
        role=DialogueRole.USER,
        text="Раньше мне нравилось фотографироваться.",
        sequence=1,
    )
    history = DialogueHistory(messages=[message])
    source = SourceReference(message_id=message.id, char_start=0, char_end=6)

    restored = DialogueHistory.model_validate_json(history.model_dump_json())

    assert restored.messages[0].id == source.message_id
