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
    validate_discovery_memory,
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
    source_refs: Optional[List[SourceReference]] = None,
    created_turn: int = 1,
    updated_turn: int = 1,
) -> ModelItem:
    return ModelItem(
        id=item_id or _id("mi"),
        kind=kind,
        content="нормализованный смысл пользовательского материала",
        provenance=provenance,
        status=status,
        source_refs=source_refs or [_source()],
        supersedes_item_ids=supersedes_item_ids or [],
        created_turn=created_turn,
        updated_turn=updated_turn,
    )


def _relation(
    source_item_id: str,
    target_item_id: str,
    *,
    relation_id: Optional[str] = None,
    supersedes_relation_ids: Optional[List[str]] = None,
    meaning: str = "пользователь воспринимает эти два изменения как связанные",
    source_refs: Optional[List[SourceReference]] = None,
) -> Relation:
    return Relation(
        id=relation_id or _id("rel"),
        meaning=meaning,
        source_item_ids=[source_item_id],
        target_item_ids=[target_item_id],
        provenance=Provenance.USER_INTERPRETATION,
        status=ItemStatus.RECORDED,
        source_refs=source_refs or [_source()],
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
    source_message_id: Optional[str] = None,
) -> ActiveContentItem:
    return ActiveContentItem(
        id=content_id or _id("aci"),
        kind=kind,
        content="адресуемый смысл последнего системного ответа",
        model_item_ids=model_item_ids or [],
        relation_ids=relation_ids or [],
        source_message_id=source_message_id or _id("msg"),
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
    content: List[ActiveContentItem],
    targets: List[ResponseTarget],
    *,
    source_system_message_id: Optional[str] = None,
) -> ActiveConversationState:
    return ActiveConversationState(
        id=_id("acs"),
        created_turn=1,
        source_system_message_id=source_system_message_id or _id("msg"),
        active_content={item.id: item for item in content},
        response_targets={target.id: target for target in targets},
    )


def _valid_acs() -> ActiveConversationState:
    content = _active_content()
    return _acs([content], [_target(content.id)])


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
    assert restored.items[new_item.id].supersedes_item_ids == (old_item.id,)


def test_relation_uses_arbitrary_meaning_without_relation_enum():
    attempt = _item(kind=ModelItemKind.PREVIOUS_ATTEMPT)
    hunger = _item(kind=ModelItemKind.BARRIER)
    relation = _relation(
        hunger.id,
        attempt.id,
        meaning=(
            "пользователь считает вечерний голод основной причиной срыва "
            "предыдущей попытки есть меньше"
        ),
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

    assert model.relations[new_relation.id].supersedes_relation_ids == (old_relation.id,)


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


def test_human_model_allows_multiple_items_of_the_same_kind():
    first_change = _item(kind=ModelItemKind.LIFE_CHANGE)
    second_change = _item(kind=ModelItemKind.LIFE_CHANGE)

    model = HumanModel(items={first_change.id: first_change, second_change.id: second_change})

    assert len(model.items) == 2
    assert {item.kind for item in model.items.values()} == {ModelItemKind.LIFE_CHANGE}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _item(created_turn=2, updated_turn=1),
        lambda: Relation(
            id=_id("rel"),
            meaning="связь",
            source_item_ids=[_id("mi")],
            target_item_ids=[_id("mi")],
            provenance=Provenance.USER_PROVIDED,
            status=ItemStatus.RECORDED,
            source_refs=[_source()],
            created_turn=2,
            updated_turn=1,
        ),
    ],
)
def test_turn_order_rejects_updated_turn_before_created_turn(factory):
    with pytest.raises(ValidationError, match="updated_turn"):
        factory()


@pytest.mark.parametrize(
    ("factory", "prefix"),
    [
        (lambda: DialogueMessage(role=DialogueRole.USER, text="текст", sequence=1), "msg"),
        (lambda: _item(), "mi"),
        (lambda: _relation(_id("mi"), _id("mi")), "rel"),
        (_valid_acs, "acs"),
        (lambda: _active_content(), "aci"),
        (lambda: _target(_id("aci")), "rt"),
    ],
)
def test_all_generated_core_ids_have_prefixed_uuids(factory, prefix):
    value = factory()

    assert value.id.startswith(f"{prefix}_")
    assert uuid4().__class__(value.id.removeprefix(f"{prefix}_"))


@pytest.mark.parametrize(
    ("factory", "wrong_id"),
    [
        (lambda: DialogueMessage(role=DialogueRole.USER, text="текст", sequence=1), _id("mi")),
        (lambda: _item(), _id("rel")),
        (lambda: _relation(_id("mi"), _id("mi")), _id("mi")),
        (_valid_acs, _id("rt")),
        (lambda: _active_content(), _id("msg")),
        (lambda: _target(_id("aci")), _id("mi")),
    ],
)
def test_core_ids_reject_wrong_prefixes(factory, wrong_id):
    value = factory()

    with pytest.raises(ValidationError):
        value.__class__.model_validate({**value.model_dump(), "id": wrong_id})


@pytest.mark.parametrize(
    ("factory", "prefix"),
    [
        (lambda: DialogueMessage(role=DialogueRole.USER, text="текст", sequence=1), "msg"),
        (lambda: _item(), "mi"),
        (lambda: _relation(_id("mi"), _id("mi")), "rel"),
        (_valid_acs, "acs"),
        (lambda: _active_content(), "aci"),
        (lambda: _target(_id("aci")), "rt"),
    ],
)
def test_core_ids_reject_malformed_uuid_parts(factory, prefix):
    value = factory()

    with pytest.raises(ValidationError):
        value.__class__.model_validate(
            {**value.model_dump(), "id": f"{prefix}_not-a-uuid"}
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DialogueHistory(),
        lambda: _active_content(),
        _valid_acs,
        lambda: TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=_id("aci")),
    ],
)
def test_extra_fields_are_forbidden_on_representative_core_models(factory):
    value = factory()

    with pytest.raises(ValidationError):
        value.__class__.model_validate({**value.model_dump(), "unexpected": True})


def test_persistent_collections_reject_in_place_mutation_and_allow_revalidated_copy():
    old_item = _item()
    model = HumanModel(items={old_item.id: old_item})
    new_item = _item()
    content = _active_content()
    target = _target(content.id)
    active_state = _acs([content], [target])

    with pytest.raises(TypeError):
        model.items[_id("mi")] = new_item
    with pytest.raises(TypeError):
        active_state.response_targets[_id("rt")] = target
    with pytest.raises(ValidationError):
        old_item.supersedes_item_ids += (new_item.id,)

    updated = model.with_updates(items={old_item.id: old_item, new_item.id: new_item})

    assert set(updated.items) == {old_item.id, new_item.id}
    assert set(model.items) == {old_item.id}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: (lambda item_id: _item(supersedes_item_ids=[item_id, item_id]))(_id("mi")),
        lambda: (lambda relation_id: Relation(
            id=_id("rel"),
            meaning="связь",
            source_item_ids=[_id("mi")],
            target_item_ids=[_id("mi")],
            provenance=Provenance.USER_PROVIDED,
            status=ItemStatus.RECORDED,
            source_refs=[_source()],
            supersedes_relation_ids=[relation_id, relation_id],
            created_turn=1,
            updated_turn=1,
        ))(_id("rel")),
        lambda: (lambda item_id: Relation(
            id=_id("rel"),
            meaning="связь",
            source_item_ids=[item_id, item_id],
            target_item_ids=[_id("mi")],
            provenance=Provenance.USER_PROVIDED,
            status=ItemStatus.RECORDED,
            source_refs=[_source()],
            created_turn=1,
            updated_turn=1,
        ))(_id("mi")),
        lambda: (lambda item_id: Relation(
            id=_id("rel"),
            meaning="связь",
            source_item_ids=[_id("mi")],
            target_item_ids=[item_id, item_id],
            provenance=Provenance.USER_PROVIDED,
            status=ItemStatus.RECORDED,
            source_refs=[_source()],
            created_turn=1,
            updated_turn=1,
        ))(_id("mi")),
        lambda: (lambda item_id: _active_content(model_item_ids=[item_id, item_id]))(_id("mi")),
        lambda: (lambda relation_id: _active_content(relation_ids=[relation_id, relation_id]))(_id("rel")),
    ],
)
def test_duplicate_references_are_rejected(factory):
    with pytest.raises(ValidationError, match="duplicate"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: (lambda item_id: _item(item_id=item_id, supersedes_item_ids=[item_id]))(_id("mi")),
        lambda: (lambda relation_id: Relation(
            id=relation_id,
            meaning="связь",
            source_item_ids=[_id("mi")],
            target_item_ids=[_id("mi")],
            provenance=Provenance.USER_PROVIDED,
            status=ItemStatus.RECORDED,
            source_refs=[_source()],
            supersedes_relation_ids=[relation_id],
            created_turn=1,
            updated_turn=1,
        ))(_id("rel")),
    ],
)
def test_self_supersession_is_rejected(factory):
    with pytest.raises(ValidationError, match="supersede itself"):
        factory()


def test_cross_object_target_references_validate_existing_item_and_relation():
    item = _item()
    other_item = _item()
    relation = _relation(item.id, other_item.id)
    model = HumanModel(
        items={item.id: item, other_item.id: other_item},
        relations={relation.id: relation},
    )
    content = _active_content(model_item_ids=[item.id], relation_ids=[relation.id])
    item_target = _target(
        content.id,
        subject=TargetReference(kind=TargetSubjectKind.MODEL_ITEM, id=item.id),
    )
    relation_target = _target(
        content.id,
        subject=TargetReference(kind=TargetSubjectKind.RELATION, id=relation.id),
    )

    validate_acs_human_model_references(_acs([content], [item_target, relation_target]), model)


@pytest.mark.parametrize("subject_kind", [TargetSubjectKind.MODEL_ITEM, TargetSubjectKind.RELATION])
def test_cross_object_target_references_reject_missing_item_or_relation(subject_kind):
    item = _item()
    model = HumanModel(items={item.id: item})
    content = _active_content()
    prefix = "mi" if subject_kind is TargetSubjectKind.MODEL_ITEM else "rel"
    target = _target(
        content.id,
        subject=TargetReference(kind=subject_kind, id=_id(prefix)),
    )

    with pytest.raises(ValueError):
        validate_acs_human_model_references(_acs([content], [target]), model)


def test_history_aware_validation_checks_messages_roles_and_offsets():
    user_message = DialogueMessage(role=DialogueRole.USER, text="Пользовательский текст", sequence=1)
    system_message = DialogueMessage(role=DialogueRole.SYSTEM, text="Системный ответ", sequence=2)
    history = DialogueHistory(messages=[user_message, system_message])
    item = _item(source_refs=[_source(user_message.id)])
    related = _item(source_refs=[_source(user_message.id)])
    relation = _relation(item.id, related.id, source_refs=[_source(user_message.id)])
    model = HumanModel(
        items={item.id: item, related.id: related},
        relations={relation.id: relation},
    )
    content = _active_content(
        model_item_ids=[item.id],
        relation_ids=[relation.id],
        source_message_id=system_message.id,
    )
    state = _acs([content], [_target(content.id)], source_system_message_id=system_message.id)

    validate_discovery_memory(model, state, history)

    missing_source_item = _item(source_refs=[_source(_id("msg"))])
    missing_source_model = HumanModel(items={missing_source_item.id: missing_source_item})
    with pytest.raises(ValueError, match="SourceReference.message_id"):
        validate_discovery_memory(missing_source_model, None, history)

    out_of_bounds_item = _item(
        source_refs=[SourceReference(message_id=user_message.id, char_start=0, char_end=999)]
    )
    with pytest.raises(ValueError, match="char_end"):
        validate_discovery_memory(HumanModel(items={out_of_bounds_item.id: out_of_bounds_item}), None, history)

    user_content = _active_content(source_message_id=user_message.id)
    user_state = _acs([user_content], [_target(user_content.id)], source_system_message_id=user_message.id)
    with pytest.raises(ValueError, match="system message"):
        validate_discovery_memory(HumanModel(), user_state, history)


@pytest.mark.parametrize("missing_reference", ["active_content", "active_state"])
def test_history_aware_validation_rejects_missing_system_message_references(missing_reference):
    system_message = DialogueMessage(role=DialogueRole.SYSTEM, text="Ответ", sequence=1)
    history = DialogueHistory(messages=[system_message])
    missing_message_id = _id("msg")
    content = _active_content(
        source_message_id=(missing_message_id if missing_reference == "active_content" else system_message.id)
    )
    state = _acs(
        [content],
        [_target(content.id)],
        source_system_message_id=(missing_message_id if missing_reference == "active_state" else system_message.id),
    )

    with pytest.raises(ValueError, match="must exist in DialogueHistory"):
        validate_discovery_memory(HumanModel(), state, history)
