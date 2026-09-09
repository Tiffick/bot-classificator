from copy import deepcopy

import pytest

from ai.cognitive_turn import (
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    EvidenceOrigin,
    ItemOperation,
    ItemOperationKind,
    ItemReference,
    ProposalMaterialization,
    ReconciliationOutcome,
    RelationOperation,
    RelationOperationKind,
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
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    ItemStatus,
    ModelItem,
    ModelItemKind,
    Provenance,
    ResponseTarget,
    SourceReference,
    TargetInteractionKind,
    TargetReference,
    TargetSubjectKind,
    make_id,
    validate_discovery_memory,
)
from ai.discovery_state_applier import (
    DiscoveryStateApplyError,
    apply_cognitive_turn,
)


def _message(role, text, sequence):
    return DialogueMessage(role=role, text=text, sequence=sequence)


def _span(text):
    return SourceSpan(char_start=0, char_end=len(text))


def _current_pair(text="Пользовательский материал", reply="Что ты сейчас замечаешь?"):
    return (
        _message(DialogueRole.USER, text, 1),
        _message(DialogueRole.SYSTEM, reply, 2),
    )


def _item(*, content="известный материал", kind=ModelItemKind.EXPERIENCE, provenance=Provenance.USER_PROVIDED, status=ItemStatus.RECORDED, message=None, turn=1):
    message = message or _message(DialogueRole.USER, "исходный материал", 1)
    return ModelItem(
        kind=kind,
        content=content,
        provenance=provenance,
        status=status,
        source_refs=(SourceReference(message_id=message.id),),
        created_turn=turn,
        updated_turn=turn,
    )


def _action(*, count=1, interaction=TargetInteractionKind.OPEN_RESPONSE, contents=None):
    contents = contents or tuple(
        ActionContent(
            local_id=f"content:{index}",
            kind=ActiveContentKind.SYSTEM_QUESTION,
            semantic_content=f"уточнить тему {index}",
        )
        for index in range(1, count + 1)
    )
    targets = tuple(
        ActionTarget(
            local_id=f"target:{index}",
            active_content_local_id=content.local_id,
            subject=ActionSubjectReference(
                kind=TargetSubjectKind.ACTIVE_CONTENT,
                active_content_local_id=content.local_id,
            ),
            interaction=interaction,
        )
        for index, content in enumerate(contents, 1)
    )
    return SystemAction(contents=contents, response_targets=targets)


def _result(*, user_text, reply, patch=None, action=None, reconciliation=(), materializations=(), segments=None):
    action = action if action is not None else _action()
    patch = patch or StatePatch()
    if materializations:
        patch = StatePatch(
            item_operations=patch.item_operations,
            relation_operations=patch.relation_operations,
            proposal_materializations=tuple(materializations),
        )
    if segments is None:
        segments = (
            ReplySegment(
                local_id="segment:reply",
                text=reply,
                realizes_action_content_ids=tuple(content.local_id for content in action.contents),
            ),
        )
    return CognitiveTurnResult(
        state_patch=patch,
        reconciliation=tuple(reconciliation),
        system_action=action,
        reply_segments=tuple(segments),
    )


def _add(local_id, text, *, origin=EvidenceOrigin.CURRENT_USER_MATERIAL, kind=ModelItemKind.EXPERIENCE):
    return ItemOperation(
        operation=ItemOperationKind.ADD,
        local_id=local_id,
        kind=kind,
        content=text,
        evidence_origin=origin,
        source_span=_span(text),
    )


def _apply_first_turn(result, user_text, reply=None):
    user_message, system_message = _current_pair(user_text, reply or "Что ты сейчас замечаешь?")
    return apply_cognitive_turn(
        HumanModel(), None, DialogueHistory(), user_message, system_message, result
    )


def _previous_proposal(*, interaction=TargetInteractionKind.EVALUATION):
    previous_user = _message(DialogueRole.USER, "Исходная реплика", 1)
    previous_system = _message(DialogueRole.SYSTEM, "Может быть, это важно?", 2)
    history = DialogueHistory(messages=(previous_user, previous_system))
    content = ActiveContentItem(
        kind=ActiveContentKind.SYSTEM_PROPOSAL,
        content="системная версия",
        source_message_id=previous_system.id,
    )
    target = ResponseTarget(
        active_content_id=content.id,
        subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=content.id),
        interaction=interaction,
    )
    acs = ActiveConversationState(
        created_turn=1,
        source_system_message_id=previous_system.id,
        active_content={content.id: content},
        response_targets={target.id: target},
    )
    return HumanModel(), acs, history, content, target


def _apply_after_proposal(outcome, *, user_text="Да", interaction=TargetInteractionKind.EVALUATION, materialize=True):
    human_model, acs, history, content, target = _previous_proposal(interaction=interaction)
    user_message = _message(DialogueRole.USER, user_text, 3)
    reply = "Спасибо за уточнение."
    system_message = _message(DialogueRole.SYSTEM, reply, 4)
    resolution = TargetResolution(
        previous_response_target_ids=(target.id,),
        outcome=outcome,
        source_span=_span(user_text),
    )
    materializations = ()
    if materialize:
        materializations = (
            ProposalMaterialization(
                resolution_target_id=target.id,
                previous_active_content_id=content.id,
                subject_kind="item",
                kind=ModelItemKind.EXPERIENCE,
                content="системная версия",
            ),
        )
    result = _result(
        user_text=user_text,
        reply=reply,
        action=SystemAction(),
        reconciliation=(resolution,),
        materializations=materializations,
    )
    return apply_cognitive_turn(human_model, acs, history, user_message, system_message, result)


def test_add_new_item():
    text = "Живот мешает двигаться"
    result = _result(
        user_text=text,
        reply="Что в этом изменилось для тебя?",
        patch=StatePatch(item_operations=(_add("item:belly", text),)),
    )

    applied = _apply_first_turn(result, text, "Что в этом изменилось для тебя?")

    item_id = applied.item_id_mapping["item:belly"]
    assert applied.updated_human_model.items[item_id].provenance == Provenance.USER_PROVIDED
    assert applied.updated_human_model.items[item_id].status == ItemStatus.RECORDED


def test_reinforce_existing_item():
    old_user = _message(DialogueRole.USER, "Живот беспокоит", 1)
    existing = _item(content="живот беспокоит", message=old_user)
    human_model = HumanModel(items={existing.id: existing})
    history = DialogueHistory(messages=(old_user, _message(DialogueRole.SYSTEM, "Понимаю.", 2)))
    text = "Да, это всё ещё важно"
    user = _message(DialogueRole.USER, text, 3)
    reply = "Что в этом сейчас наиболее заметно?"
    system = _message(DialogueRole.SYSTEM, reply, 4)
    result = _result(
        user_text=text,
        reply=reply,
        patch=StatePatch(
            item_operations=(
                ItemOperation(
                    operation=ItemOperationKind.REINFORCE,
                    existing_item_id=existing.id,
                    evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                    source_span=_span(text),
                ),
            )
        ),
    )

    applied = apply_cognitive_turn(human_model, None, history, user, system, result)

    assert set(applied.updated_human_model.items) == {existing.id}
    assert len(applied.updated_human_model.items[existing.id].source_refs) == 2


def test_correct_existing_item_creates_superseding_replacement():
    old_user = _message(DialogueRole.USER, "Одежда не важна", 1)
    existing = _item(content="одежда не важна", message=old_user)
    human_model = HumanModel(items={existing.id: existing})
    history = DialogueHistory(messages=(old_user, _message(DialogueRole.SYSTEM, "Понял.", 2)))
    text = "Хотя нет, вещи сидят хуже"
    user, system = _message(DialogueRole.USER, text, 3), _message(DialogueRole.SYSTEM, "Что ты при этом чувствуешь?", 4)
    result = _result(
        user_text=text,
        reply=system.text,
        patch=StatePatch(
            item_operations=(
                ItemOperation(
                    operation=ItemOperationKind.CORRECT,
                    local_id="item:clothes-correction",
                    existing_item_id=existing.id,
                    kind=ModelItemKind.EXPERIENCE,
                    content="вещи сидят хуже",
                    evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                    source_span=_span(text),
                ),
            )
        ),
    )

    applied = apply_cognitive_turn(human_model, None, history, user, system, result)

    replacement = applied.updated_human_model.items[applied.item_id_mapping["item:clothes-correction"]]
    assert applied.updated_human_model.items[existing.id].status == ItemStatus.CORRECTED
    assert replacement.supersedes_item_ids == (existing.id,)


def test_relation_between_two_new_items():
    text = "два связанных переживания"
    patch = StatePatch(
        item_operations=(_add("item:first", text), _add("item:second", text)),
        relation_operations=(
            RelationOperation(
                operation=RelationOperationKind.ADD,
                local_id="relation:link",
                source_items=(ItemReference(local_item_id="item:first"),),
                target_items=(ItemReference(local_item_id="item:second"),),
                meaning="пользователь видит связь",
                evidence_origin=EvidenceOrigin.CURRENT_USER_INTERPRETATION,
                source_span=_span(text),
            ),
        ),
    )
    result = _result(user_text=text, reply="Что в этой связи важнее?", patch=patch)

    applied = _apply_first_turn(result, text, "Что в этой связи важнее?")

    relation = applied.updated_human_model.relations[applied.relation_id_mapping["relation:link"]]
    assert relation.source_item_ids == (applied.item_id_mapping["item:first"],)
    assert relation.target_item_ids == (applied.item_id_mapping["item:second"],)


def test_relation_existing_to_new():
    previous_user = _message(DialogueRole.USER, "Старый материал", 1)
    existing = _item(message=previous_user)
    human_model = HumanModel(items={existing.id: existing})
    history = DialogueHistory(messages=(previous_user, _message(DialogueRole.SYSTEM, "Понял.", 2)))
    text = "Новый материал связан со старым"
    user, system = _message(DialogueRole.USER, text, 3), _message(DialogueRole.SYSTEM, "Как это связано?", 4)
    patch = StatePatch(
        item_operations=(_add("item:new", text),),
        relation_operations=(
            RelationOperation(
                operation=RelationOperationKind.ADD,
                local_id="relation:existing-new",
                source_items=(ItemReference(existing_item_id=existing.id),),
                target_items=(ItemReference(local_item_id="item:new"),),
                meaning="пользователь связывает материалы",
                evidence_origin=EvidenceOrigin.CURRENT_USER_INTERPRETATION,
                source_span=_span(text),
            ),
        ),
    )
    result = _result(user_text=text, reply=system.text, patch=patch)

    applied = apply_cognitive_turn(human_model, None, history, user, system, result)

    relation = applied.updated_human_model.relations[applied.relation_id_mapping["relation:existing-new"]]
    assert relation.source_item_ids == (existing.id,)


def test_user_interpretation_and_other_person_provenance_are_derived():
    text = "Это жена сказала, а я думаю иначе"
    patch = StatePatch(
        item_operations=(
            _add(
                "item:interpretation",
                text,
                origin=EvidenceOrigin.CURRENT_USER_INTERPRETATION,
                kind=ModelItemKind.USER_INTERPRETATION,
            ),
            _add(
                "item:wife-view",
                text,
                origin=EvidenceOrigin.CURRENT_OTHER_PERSON_REPORT,
                kind=ModelItemKind.OTHER_PERSON_VIEW,
            ),
        )
    )
    result = _result(user_text=text, reply="Что из этого ближе тебе?", patch=patch)

    applied = _apply_first_turn(result, text, "Что из этого ближе тебе?")

    assert applied.updated_human_model.items[applied.item_id_mapping["item:interpretation"]].provenance == Provenance.USER_INTERPRETATION
    assert applied.updated_human_model.items[applied.item_id_mapping["item:wife-view"]].provenance == Provenance.OTHER_PERSON


@pytest.mark.parametrize(
    ("outcome", "status"),
    (
        (ReconciliationOutcome.SUPPORTED, ItemStatus.SUPPORTED),
        (ReconciliationOutcome.REJECTED, ItemStatus.REJECTED),
        (ReconciliationOutcome.PARTIALLY_SUPPORTED, ItemStatus.PARTIALLY_SUPPORTED),
        (ReconciliationOutcome.UNCERTAIN, ItemStatus.UNCERTAIN),
    ),
)
def test_system_proposal_materialization_derives_status(outcome, status):
    applied = _apply_after_proposal(outcome, user_text="Реакция")

    created = next(iter(applied.updated_human_model.items.values()))
    assert created.provenance == Provenance.SYSTEM_PROPOSED
    assert created.status == status
    assert len(created.source_refs) == 2


def test_ambiguous_reaction_does_not_materialize_proposal():
    human_model, acs, history, content, first_target = _previous_proposal()
    second_target = ResponseTarget(
        active_content_id=content.id,
        subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=content.id),
        interaction=TargetInteractionKind.EVALUATION,
    )
    acs = ActiveConversationState(
        created_turn=1,
        source_system_message_id=history.messages[-1].id,
        active_content={content.id: content},
        response_targets={first_target.id: first_target, second_target.id: second_target},
    )
    text, reply = "Да", "Спасибо за ответ."
    user, system = _message(DialogueRole.USER, text, 3), _message(DialogueRole.SYSTEM, reply, 4)
    result = _result(
        user_text=text,
        reply=reply,
        action=SystemAction(),
        reconciliation=(
            TargetResolution(
                previous_response_target_ids=(first_target.id, second_target.id),
                outcome=ReconciliationOutcome.AMBIGUOUS,
                source_span=_span(text),
            ),
        ),
    )

    applied = apply_cognitive_turn(human_model, acs, history, user, system, result)

    assert applied.updated_human_model.items == {}


def test_omitted_previous_target_is_not_forced_to_resolve():
    human_model, acs, history, _, _ = _previous_proposal()
    text, reply = "Кстати, я начал больше ходить.", "Это важно заметить."
    user, system = _message(DialogueRole.USER, text, 3), _message(DialogueRole.SYSTEM, reply, 4)
    result = _result(user_text=text, reply=reply, action=SystemAction())

    applied = apply_cognitive_turn(human_model, acs, history, user, system, result)

    assert applied.new_acs is None
    assert applied.updated_human_model.items == {}


def test_refused_is_accepted_for_previous_target():
    human_model, acs, history, _, target = _previous_proposal(
        interaction=TargetInteractionKind.OPEN_RESPONSE
    )
    text, reply = "Не хочу это обсуждать.", "Хорошо, не будем торопиться."
    user, system = _message(DialogueRole.USER, text, 3), _message(DialogueRole.SYSTEM, reply, 4)
    result = _result(
        user_text=text,
        reply=reply,
        action=SystemAction(),
        reconciliation=(
            TargetResolution(
                previous_response_target_ids=(target.id,),
                outcome=ReconciliationOutcome.REFUSED,
                source_span=_span(text),
            ),
        ),
    )

    applied = apply_cognitive_turn(human_model, acs, history, user, system, result)

    assert applied.new_acs is None


def test_consent_target_accepts_declined_reconciliation():
    applied = _apply_after_proposal(
        ReconciliationOutcome.DECLINED,
        user_text="Пока не хочу ничего пробовать.",
        interaction=TargetInteractionKind.CONSENT,
        materialize=False,
    )

    assert applied.new_acs is None
    assert applied.updated_human_model.items == {}


def test_open_response_target_rejects_declined_reconciliation():
    with pytest.raises(
        DiscoveryStateApplyError,
        match="Reconciliation outcome is incompatible with target interaction",
    ):
        _apply_after_proposal(
            ReconciliationOutcome.DECLINED,
            user_text="Пока не хочу ничего пробовать.",
            interaction=TargetInteractionKind.OPEN_RESPONSE,
            materialize=False,
        )


def test_invalid_previous_target_and_existing_item_are_rejected():
    text, reply = "Материал", "Что дальше?"
    user, system = _current_pair(text, reply)
    result = _result(
        user_text=text,
        reply=reply,
        patch=StatePatch(
            item_operations=(
                ItemOperation(
                    operation=ItemOperationKind.REINFORCE,
                    existing_item_id=make_id("mi"),
                    evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                    source_span=_span(text),
                ),
            )
        ),
    )
    with pytest.raises(DiscoveryStateApplyError):
        apply_cognitive_turn(HumanModel(), None, DialogueHistory(), user, system, result)

    missing_target = TargetResolution(
        previous_response_target_ids=(make_id("rt"),),
        outcome=ReconciliationOutcome.ANSWERED,
    )
    result = _result(user_text=text, reply=reply, reconciliation=(missing_target,))
    with pytest.raises(DiscoveryStateApplyError):
        apply_cognitive_turn(HumanModel(), None, DialogueHistory(), user, system, result)


def test_invalid_source_span_is_rejected():
    text, reply = "коротко", "Что дальше?"
    operation = ItemOperation(
        operation=ItemOperationKind.ADD,
        local_id="item:invalid-span",
        kind=ModelItemKind.EXPERIENCE,
        content="материал",
        evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
        source_span=SourceSpan(char_start=0, char_end=len(text) + 1),
    )
    result = _result(user_text=text, reply=reply, patch=StatePatch(item_operations=(operation,)))

    with pytest.raises(DiscoveryStateApplyError):
        _apply_first_turn(result, text, reply)


def test_action_creates_acs_with_exact_system_message_and_multiple_targets():
    text = "Пользовательский материал"
    reply = "Понимаю. Что из этого важнее?"
    action = _action(count=2, interaction=TargetInteractionKind.EVALUATION)
    result = _result(user_text=text, reply=reply, action=action)

    applied = _apply_first_turn(result, text, reply)

    assert applied.new_acs is not None
    assert len(applied.new_acs.active_content) == 2
    assert len(applied.new_acs.response_targets) == 2
    assert applied.new_acs.source_system_message_id == applied.updated_history.messages[-1].id


def test_action_persists_distinct_owner_and_active_content_subject():
    text = "Вечером особенно трудно."
    reply = "Похоже, вечерний голод мешает устойчивости. Насколько это похоже на ваш опыт?"
    reflection = ActionContent(
        local_id="content:reflection",
        kind=ActiveContentKind.SYSTEM_REFLECTION,
        semantic_content="вечерний голод мешает устойчивости",
    )
    question = ActionContent(
        local_id="content:question",
        kind=ActiveContentKind.SYSTEM_QUESTION,
        semantic_content="насколько это похоже на опыт пользователя",
    )
    action = SystemAction(
        contents=(reflection, question),
        response_targets=(
            ActionTarget(
                local_id="target:question",
                active_content_local_id=question.local_id,
                subject=ActionSubjectReference(
                    kind=TargetSubjectKind.ACTIVE_CONTENT,
                    active_content_local_id=reflection.local_id,
                ),
                interaction=TargetInteractionKind.EVALUATION,
            ),
        ),
    )
    result = _result(user_text=text, reply=reply, action=action)

    applied = _apply_first_turn(result, text, reply)

    assert applied.new_acs is not None
    assert len(applied.new_acs.active_content) == 2
    target = next(iter(applied.new_acs.response_targets.values()))
    owner = applied.new_acs.active_content[target.active_content_id]
    subject = applied.new_acs.active_content[target.subject.id]
    assert owner.content == question.semantic_content
    assert subject.content == reflection.semantic_content
    assert target.active_content_id != target.subject.id
    validate_discovery_memory(
        applied.updated_human_model, applied.new_acs, applied.updated_history
    )


def test_no_targets_creates_no_acs():
    text, reply = "Вопрос", "Вот краткий ответ."
    result = _result(user_text=text, reply=reply, action=SystemAction())

    applied = _apply_first_turn(result, text, reply)

    assert applied.new_acs is None


def test_empathy_and_question_segments_and_multiple_contents_are_valid():
    text = "Материал"
    reply = "Понимаю. Что из этого важнее?"
    action = _action(count=2)
    segments = (
        ReplySegment(local_id="segment:empathy", text="Понимаю. "),
        ReplySegment(
            local_id="segment:question",
            text="Что из этого важнее?",
            realizes_action_content_ids=("content:1", "content:2"),
        ),
    )
    result = _result(user_text=text, reply=reply, action=action, segments=segments)

    applied = _apply_first_turn(result, text, reply)

    assert applied.new_acs is not None
    assert len(applied.new_acs.active_content) == 2


def test_system_text_must_equal_joined_reply_segments():
    text = "Материал"
    result = _result(user_text=text, reply="Другой ответ")

    with pytest.raises(DiscoveryStateApplyError):
        _apply_first_turn(result, text, "Не тот ответ")


def test_failure_does_not_mutate_original_inputs_and_exact_pair_is_appended():
    text, reply = "Материал", "Что дальше?"
    user, system = _current_pair(text, reply)
    human_model, history = HumanModel(), DialogueHistory()
    before_model, before_history = deepcopy(human_model), deepcopy(history)
    invalid = _result(
        user_text=text,
        reply=reply,
        patch=StatePatch(
            item_operations=(
                ItemOperation(
                    operation=ItemOperationKind.REINFORCE,
                    existing_item_id=make_id("mi"),
                    evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                    source_span=_span(text),
                ),
            )
        ),
    )
    with pytest.raises(DiscoveryStateApplyError):
        apply_cognitive_turn(human_model, None, history, user, system, invalid)
    assert human_model == before_model
    assert history == before_history

    valid = _result(user_text=text, reply=reply, action=SystemAction())
    applied = apply_cognitive_turn(human_model, None, history, user, system, valid)
    assert applied.updated_history.messages == (user, system)
    validate_discovery_memory(
        applied.updated_human_model, applied.new_acs, applied.updated_history
    )


def test_failure_does_not_mutate_previous_acs_or_history():
    human_model, acs, history, _, _ = _previous_proposal()
    before_model = deepcopy(human_model)
    before_acs = acs.model_dump()
    before_history = deepcopy(history)
    text, reply = "Ответ", "Спасибо за ответ."
    user, system = _message(DialogueRole.USER, text, 3), _message(DialogueRole.SYSTEM, reply, 4)
    result = _result(
        user_text=text,
        reply=reply,
        action=SystemAction(),
        reconciliation=(
            TargetResolution(
                previous_response_target_ids=(make_id("rt"),),
                outcome=ReconciliationOutcome.ANSWERED,
            ),
        ),
    )

    with pytest.raises(DiscoveryStateApplyError):
        apply_cognitive_turn(human_model, acs, history, user, system, result)

    assert human_model == before_model
    assert acs.model_dump() == before_acs
    assert history == before_history
