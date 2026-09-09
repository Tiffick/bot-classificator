import pytest
from pydantic import ValidationError

from ai.cognitive_turn import (
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    EvidenceOrigin,
    ItemOperation,
    ItemOperationKind,
    ItemReference,
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
    ActiveContentKind,
    ModelItemKind,
    TargetInteractionKind,
    TargetSubjectKind,
    make_id,
)


SPAN = SourceSpan(char_start=0, char_end=1)


def _add(local_id="item:new"):
    return ItemOperation(
        operation=ItemOperationKind.ADD,
        local_id=local_id,
        kind=ModelItemKind.EXPERIENCE,
        content="новый пользовательский материал",
        evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
        source_span=SPAN,
    )


def _question_action():
    content = ActionContent(
        local_id="content:question",
        kind=ActiveContentKind.SYSTEM_QUESTION,
        semantic_content="уточнить переживание пользователя",
    )
    target = ActionTarget(
        local_id="target:question",
        active_content_local_id=content.local_id,
        subject=ActionSubjectReference(
            kind=TargetSubjectKind.ACTIVE_CONTENT,
            active_content_local_id=content.local_id,
        ),
        interaction=TargetInteractionKind.OPEN_RESPONSE,
    )
    return SystemAction(contents=(content,), response_targets=(target,))


def _turn(*, patch=None, action=None, segments=None, reconciliation=()):
    action = action or _question_action()
    return CognitiveTurnResult(
        state_patch=patch or StatePatch(),
        reconciliation=reconciliation,
        system_action=action,
        reply_segments=segments
        or (
            ReplySegment(
                local_id="segment:question",
                text="Что для тебя сейчас важнее всего?",
                realizes_action_content_ids=("content:question",),
            ),
        ),
    )


def test_valid_add_contract():
    result = _turn(patch=StatePatch(item_operations=(_add(),)))

    assert result.state_patch.item_operations[0].operation == ItemOperationKind.ADD


def test_add_requires_new_material_fields():
    with pytest.raises(ValidationError):
        ItemOperation(
            operation=ItemOperationKind.ADD,
            local_id="item:new",
            kind=ModelItemKind.EXPERIENCE,
            content="материал",
        )


def test_valid_reinforce_contract():
    operation = ItemOperation(
        operation=ItemOperationKind.REINFORCE,
        existing_item_id=make_id("mi"),
        evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
        source_span=SPAN,
    )

    assert operation.local_id is None


def test_reinforce_cannot_hide_content_mutation():
    with pytest.raises(ValidationError):
        ItemOperation(
            operation=ItemOperationKind.REINFORCE,
            existing_item_id=make_id("mi"),
            content="новое содержание",
            evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
            source_span=SPAN,
        )


def test_valid_correct_contract():
    operation = ItemOperation(
        operation=ItemOperationKind.CORRECT,
        local_id="item:replacement",
        existing_item_id=make_id("mi"),
        kind=ModelItemKind.EXPERIENCE,
        content="уточнённый материал",
        evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
        source_span=SPAN,
    )

    assert operation.operation == ItemOperationKind.CORRECT


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"existing_item_id": make_id("mi"), "local_item_id": "item:new"},
    ),
)
def test_item_reference_requires_xor(kwargs):
    with pytest.raises(ValidationError):
        ItemReference(**kwargs)


def test_valid_relation_between_two_new_items():
    relation = RelationOperation(
        operation=RelationOperationKind.ADD,
        local_id="relation:causal-link",
        source_items=(ItemReference(local_item_id="item:first"),),
        target_items=(ItemReference(local_item_id="item:second"),),
        meaning="пользователь связывает эти переживания",
        evidence_origin=EvidenceOrigin.CURRENT_USER_INTERPRETATION,
        source_span=SPAN,
    )
    result = _turn(
        patch=StatePatch(
            item_operations=(_add("item:first"), _add("item:second")),
            relation_operations=(relation,),
        )
    )

    assert result.state_patch.relation_operations[0].local_id == "relation:causal-link"


def test_duplicate_item_local_ids_are_rejected():
    with pytest.raises(ValidationError):
        _turn(patch=StatePatch(item_operations=(_add(), _add())))


def test_ambiguous_resolution_allows_multiple_targets():
    resolution = TargetResolution(
        previous_response_target_ids=(make_id("rt"), make_id("rt")),
        outcome=ReconciliationOutcome.AMBIGUOUS,
    )

    assert len(resolution.previous_response_target_ids) == 2


def test_non_ambiguous_resolution_rejects_multiple_targets():
    with pytest.raises(ValidationError):
        TargetResolution(
            previous_response_target_ids=(make_id("rt"), make_id("rt")),
            outcome=ReconciliationOutcome.SUPPORTED,
        )


def test_duplicate_resolution_targets_are_rejected_across_entries():
    target_id = make_id("rt")
    with pytest.raises(ValidationError):
        _turn(
            reconciliation=(
                TargetResolution(
                    previous_response_target_ids=(target_id,),
                    outcome=ReconciliationOutcome.ANSWERED,
                ),
                TargetResolution(
                    previous_response_target_ids=(target_id,),
                    outcome=ReconciliationOutcome.REFUSED,
                ),
            )
        )


def test_action_target_cannot_point_to_missing_content():
    with pytest.raises(ValidationError):
        SystemAction(
            contents=(),
            response_targets=(
                ActionTarget(
                    local_id="target:missing",
                    active_content_local_id="content:missing",
                    subject=ActionSubjectReference(
                        kind=TargetSubjectKind.ACTIVE_CONTENT,
                        active_content_local_id="content:missing",
                    ),
                    interaction=TargetInteractionKind.OPEN_RESPONSE,
                ),
            ),
        )


def test_action_target_rejects_persistent_active_content_id():
    with pytest.raises(ValidationError, match="content"):
        ActionTarget(
            local_id="target:question",
            active_content_local_id=make_id("aci"),
            subject=ActionSubjectReference(
                kind=TargetSubjectKind.ACTIVE_CONTENT,
                active_content_local_id="content:question",
            ),
            interaction=TargetInteractionKind.OPEN_RESPONSE,
        )


def test_action_subject_rejects_persistent_active_content_id():
    with pytest.raises(ValidationError, match="content"):
        ActionSubjectReference(
            kind=TargetSubjectKind.ACTIVE_CONTENT,
            active_content_local_id=make_id("aci"),
        )


@pytest.mark.parametrize(
    ("owner_id", "subject_id"),
    (
        ("content:missing", "content:reflection"),
        ("content:question", "content:missing"),
    ),
)
def test_action_owner_and_subject_must_belong_to_current_system_action(
    owner_id, subject_id
):
    contents = (
        ActionContent(
            local_id="content:reflection",
            kind=ActiveContentKind.SYSTEM_REFLECTION,
            semantic_content="вечерний голод мешает устойчивости",
        ),
        ActionContent(
            local_id="content:question",
            kind=ActiveContentKind.SYSTEM_QUESTION,
            semantic_content="насколько это похоже на опыт пользователя",
        ),
    )

    with pytest.raises(ValidationError, match="ActionContent"):
        SystemAction(
            contents=contents,
            response_targets=(
                ActionTarget(
                    local_id="target:question",
                    active_content_local_id=owner_id,
                    subject=ActionSubjectReference(
                        kind=TargetSubjectKind.ACTIVE_CONTENT,
                        active_content_local_id=subject_id,
                    ),
                    interaction=TargetInteractionKind.EVALUATION,
                ),
            ),
        )


def test_action_owner_and_active_content_subject_may_be_distinct():
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
    result = _turn(
        action=action,
        segments=(
            ReplySegment(
                local_id="segment:question",
                text="Насколько это похоже на ваш опыт?",
                realizes_action_content_ids=(question.local_id,),
            ),
        ),
    )

    assert result.system_action.response_targets[0].active_content_local_id == question.local_id
    assert (
        result.system_action.response_targets[0].subject.active_content_local_id
        == reflection.local_id
    )


def test_reply_segment_cannot_point_to_missing_content():
    with pytest.raises(ValidationError):
        _turn(
            action=SystemAction(),
            segments=(
                ReplySegment(
                    local_id="segment:reply",
                    text="Ответ",
                    realizes_action_content_ids=("content:missing",),
                ),
            ),
        )


def test_targeted_content_requires_realizing_segment():
    with pytest.raises(ValidationError):
        _turn(
            segments=(ReplySegment(local_id="segment:reply", text="Ответ"),),
        )


def test_empty_system_action_is_valid():
    result = _turn(
        action=SystemAction(),
        segments=(ReplySegment(local_id="segment:reply", text="Фактический ответ."),),
    )

    assert result.system_action.response_targets == ()


def test_unknown_fields_are_forbidden():
    with pytest.raises(ValidationError):
        SourceSpan(char_start=0, char_end=1, unsupported=True)
