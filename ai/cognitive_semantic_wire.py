"""Production semantic wire contract and lossless adapter for CognitiveTurnResult.

Handles name only declarations within one response; they are never persistent IDs.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from ai.cognitive_core import (
    CognitiveCoreError,
    _normalize_strict_schema,
    _resolve_source_quotes,
)
from ai.cognitive_turn import (
    CognitiveTurnResult,
    DecisionIntent,
    EvidenceOrigin,
    ItemOperationKind,
    ReconciliationOutcome,
    RelationOperationKind,
    _validate_persistent_id,
)
from ai.discovery_data_model import (
    ActiveContentKind,
    ActiveConversationState,
    HumanModel,
    ModelItemKind,
    TargetInteractionKind,
)


class SemanticWireError(ValueError):
    """The semantic wire payload cannot be mapped without guessing."""


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Handle = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$", max_length=32),
]
Quote = Annotated[str, StringConstraints(min_length=1)]
Meaning = Annotated[str, StringConstraints(min_length=1)]
ItemId = Annotated[str, AfterValidator(lambda value: _validate_persistent_id(value, "mi"))]
RelationId = Annotated[str, AfterValidator(lambda value: _validate_persistent_id(value, "rel"))]
TargetId = Annotated[str, AfterValidator(lambda value: _validate_persistent_id(value, "rt"))]
ContentId = Annotated[str, AfterValidator(lambda value: _validate_persistent_id(value, "aci"))]


class _ExistingItem(_Wire):
    existing_id: ItemId


class _LocalItem(_Wire):
    handle: Handle


ItemRef = Union[_ExistingItem, _LocalItem]


class _ExistingRelation(_Wire):
    existing_id: RelationId


class _LocalRelation(_Wire):
    handle: Handle


RelationRef = Union[_ExistingRelation, _LocalRelation]


class _AddItem(_Wire):
    operation: Literal[ItemOperationKind.ADD]
    handle: Handle
    kind: ModelItemKind
    content: Meaning
    evidence_origin: EvidenceOrigin
    source_quote: Quote


class _ReinforceItem(_Wire):
    operation: Literal[ItemOperationKind.REINFORCE]
    existing_id: ItemId
    evidence_origin: EvidenceOrigin
    source_quote: Quote


class _CorrectItem(_Wire):
    operation: Literal[ItemOperationKind.CORRECT]
    handle: Handle
    existing_id: ItemId
    kind: ModelItemKind
    content: Meaning
    evidence_origin: EvidenceOrigin
    source_quote: Quote


ItemChange = Union[_AddItem, _ReinforceItem, _CorrectItem]


class _AddRelation(_Wire):
    operation: Literal[RelationOperationKind.ADD]
    handle: Handle
    source_items: tuple[ItemRef, ...] = Field(min_length=1)
    target_items: tuple[ItemRef, ...] = Field(min_length=1)
    meaning: Meaning
    evidence_origin: EvidenceOrigin
    source_quote: Quote


class _ReinforceRelation(_Wire):
    operation: Literal[RelationOperationKind.REINFORCE]
    existing_id: RelationId
    evidence_origin: EvidenceOrigin
    source_quote: Quote


class _CorrectRelation(_Wire):
    operation: Literal[RelationOperationKind.CORRECT]
    handle: Handle
    existing_id: RelationId
    source_items: tuple[ItemRef, ...] = Field(min_length=1)
    target_items: tuple[ItemRef, ...] = Field(min_length=1)
    meaning: Meaning
    evidence_origin: EvidenceOrigin
    source_quote: Quote


RelationChange = Union[_AddRelation, _ReinforceRelation, _CorrectRelation]


class _ItemMaterialization(_Wire):
    subject_kind: Literal["item"]
    resolution_target_id: TargetId
    previous_active_content_id: ContentId
    kind: ModelItemKind
    content: Meaning


class _RelationMaterialization(_Wire):
    subject_kind: Literal["relation"]
    resolution_target_id: TargetId
    previous_active_content_id: ContentId
    source_items: tuple[ItemRef, ...] = Field(min_length=1)
    target_items: tuple[ItemRef, ...] = Field(min_length=1)
    meaning: Meaning


ProposalMaterialization = Union[_ItemMaterialization, _RelationMaterialization]


class _StatePatch(_Wire):
    item_changes: tuple[ItemChange, ...]
    relation_changes: tuple[RelationChange, ...]
    proposal_materializations: tuple[ProposalMaterialization, ...]


class _Resolution(_Wire):
    previous_target_ids: tuple[TargetId, ...] = Field(min_length=1)
    outcome: ReconciliationOutcome
    source_quote: Optional[Quote]


class _Content(_Wire):
    handle: Handle
    kind: ActiveContentKind
    semantic_content: Meaning
    model_item_refs: tuple[ItemRef, ...]
    relation_refs: tuple[RelationRef, ...]


class _ReflectionContent(_Content):
    kind: Literal[ActiveContentKind.SYSTEM_REFLECTION]


class _RecognitionContent(_Content):
    kind: Literal[ActiveContentKind.RECOGNITION_OPTION]


class _TransitionContent(_Content):
    kind: Literal[ActiveContentKind.SYSTEM_TRANSITION]


class _ContentSubject(_Wire):
    content_handle: Handle


class _ItemSubject(_Wire):
    item: ItemRef


class _RelationSubject(_Wire):
    relation: RelationRef


Subject = Union[_ContentSubject, _ItemSubject, _RelationSubject]


class _Target(_Wire):
    owner_content: Handle
    subject: Subject
    interaction: TargetInteractionKind


class _SelectedBase(_Wire):
    primary_content: Optional[_Content]
    additional_contents: tuple[_Content, ...]
    response_targets: tuple[_Target, ...]


class _HumanDiscovery(_SelectedBase):
    decision_intent: Literal[DecisionIntent.HUMAN_DISCOVERY]
    response_targets: tuple[_Target, ...] = Field(min_length=1)


class _MechanismDiscovery(_SelectedBase):
    decision_intent: Literal[DecisionIntent.MECHANISM_DISCOVERY]
    response_targets: tuple[_Target, ...] = Field(min_length=1)


class _Reflection(_SelectedBase):
    decision_intent: Literal[DecisionIntent.REFLECTION]
    primary_content: _ReflectionContent


class _Recognition(_SelectedBase):
    decision_intent: Literal[DecisionIntent.RECOGNITION]
    primary_content: _RecognitionContent


class _Transition(_SelectedBase):
    decision_intent: Literal[DecisionIntent.TRANSITION]
    primary_content: _TransitionContent


class _RespectPause(_SelectedBase):
    decision_intent: Literal[DecisionIntent.RESPECT_PAUSE_OR_REFUSAL]


class _StopExploration(_SelectedBase):
    decision_intent: Literal[DecisionIntent.STOP_EXPLORATION]
    response_targets: tuple[_Target, ...] = Field(max_length=0)


class _AnswerQuestion(_SelectedBase):
    decision_intent: Literal[DecisionIntent.ANSWER_USER_QUESTION]


SelectedAction = Union[
    _HumanDiscovery,
    _MechanismDiscovery,
    _Reflection,
    _Recognition,
    _Transition,
    _RespectPause,
    _StopExploration,
    _AnswerQuestion,
]


class _ReplySegment(_Wire):
    text: Meaning
    realizes: tuple[Handle, ...]


class SemanticTurn(_Wire):
    state_patch: _StatePatch
    reconciliation: tuple[_Resolution, ...]
    selected_action: SelectedAction
    reply_segments: tuple[_ReplySegment, ...] = Field(min_length=1)


def _strict_schema() -> dict[str, Any]:
    """Strict production schema for the semantic API boundary."""
    return _normalize_strict_schema(SemanticTurn.model_json_schema())


SEMANTIC_TURN_JSON_SCHEMA = _strict_schema()


def adapt_semantic_turn(
    payload: SemanticTurn | dict[str, Any],
    *,
    current_user_text: str,
    human_model: HumanModel,
    previous_acs: ActiveConversationState | None,
) -> CognitiveTurnResult:
    """Map explicit semantic choices to technical local IDs, without inference."""
    try:
        turn = SemanticTurn.model_validate(payload)
    except ValidationError as error:
        raise SemanticWireError("Invalid semantic wire payload.") from error

    declarations: dict[str, tuple[str, str]] = {}

    def declare(handle: str, kind: str, technical_id: str) -> None:
        if handle in declarations:
            raise SemanticWireError(f"Duplicate semantic handle: {handle}")
        declarations[handle] = (kind, technical_id)

    item_number = 0
    for change in turn.state_patch.item_changes:
        if change.operation != ItemOperationKind.REINFORCE:
            item_number += 1
            declare(change.handle, "item", f"item:{item_number}")

    relation_number = 0
    for change in turn.state_patch.relation_changes:
        if change.operation != RelationOperationKind.REINFORCE:
            relation_number += 1
            declare(change.handle, "relation", f"relation:{relation_number}")

    selected = turn.selected_action
    contents = (
        (() if selected.primary_content is None else (selected.primary_content,))
        + selected.additional_contents
    )
    for index, content in enumerate(contents, start=1):
        declare(content.handle, "content", f"content:{index}")

    def local(handle: str, expected_kind: str) -> str:
        declaration = declarations.get(handle)
        if declaration is None:
            raise SemanticWireError(f"Unknown semantic handle: {handle}")
        if declaration[0] != expected_kind:
            raise SemanticWireError(
                f"Wrong-type semantic handle: {handle}; expected {expected_kind}"
            )
        return declaration[1]

    def existing(value: str, kind: str) -> str:
        collection = human_model.items if kind == "item" else human_model.relations
        if value not in collection:
            raise SemanticWireError(f"Unknown existing {kind} reference: {value}")
        return value

    def item_ref(ref: ItemRef) -> dict[str, str | None]:
        if isinstance(ref, _ExistingItem):
            return {"existing_item_id": existing(ref.existing_id, "item"), "local_item_id": None}
        return {"existing_item_id": None, "local_item_id": local(ref.handle, "item")}

    def relation_ref(ref: RelationRef) -> dict[str, str | None]:
        if isinstance(ref, _ExistingRelation):
            return {
                "existing_relation_id": existing(ref.existing_id, "relation"),
                "local_relation_id": None,
            }
        return {
            "existing_relation_id": None,
            "local_relation_id": local(ref.handle, "relation"),
        }

    item_operations: list[dict[str, Any]] = []
    for change in turn.state_patch.item_changes:
        operation: dict[str, Any] = {
            "operation": change.operation.value,
            "local_id": None,
            "existing_item_id": None,
            "kind": None,
            "content": None,
            "evidence_origin": change.evidence_origin.value,
            "source_quote": change.source_quote,
        }
        if change.operation != ItemOperationKind.REINFORCE:
            operation.update(
                local_id=local(change.handle, "item"),
                kind=change.kind.value,
                content=change.content,
            )
        if change.operation != ItemOperationKind.ADD:
            operation["existing_item_id"] = existing(change.existing_id, "item")
        item_operations.append(operation)

    relation_operations: list[dict[str, Any]] = []
    for change in turn.state_patch.relation_changes:
        operation = {
            "operation": change.operation.value,
            "local_id": None,
            "existing_relation_id": None,
            "source_items": [],
            "target_items": [],
            "meaning": None,
            "evidence_origin": change.evidence_origin.value,
            "source_quote": change.source_quote,
        }
        if change.operation != RelationOperationKind.REINFORCE:
            operation.update(
                local_id=local(change.handle, "relation"),
                source_items=[item_ref(ref) for ref in change.source_items],
                target_items=[item_ref(ref) for ref in change.target_items],
                meaning=change.meaning,
            )
        if change.operation != RelationOperationKind.ADD:
            operation["existing_relation_id"] = existing(change.existing_id, "relation")
        relation_operations.append(operation)

    if previous_acs is None and (
        turn.reconciliation or turn.state_patch.proposal_materializations
    ):
        raise SemanticWireError("Previous ACS is required for reconciliation/materialization.")

    resolutions: list[dict[str, Any]] = []
    resolved: dict[str, ReconciliationOutcome] = {}
    for resolution in turn.reconciliation:
        count = len(resolution.previous_target_ids)
        if (resolution.outcome == ReconciliationOutcome.AMBIGUOUS) != (count > 1):
            raise SemanticWireError("AMBIGUOUS requires multiple targets; other outcomes require one.")
        for target_id in resolution.previous_target_ids:
            if target_id in resolved:
                raise SemanticWireError(f"Previous target resolved twice: {target_id}")
            if target_id not in previous_acs.response_targets:
                raise SemanticWireError(f"Unknown previous response target: {target_id}")
            resolved[target_id] = resolution.outcome
        resolutions.append(
            {
                "previous_response_target_ids": list(resolution.previous_target_ids),
                "outcome": resolution.outcome.value,
                "source_quote": resolution.source_quote,
            }
        )

    materializations: list[dict[str, Any]] = []
    for materialization in turn.state_patch.proposal_materializations:
        target_id = materialization.resolution_target_id
        if resolved.get(target_id) not in (
            ReconciliationOutcome.SUPPORTED,
            ReconciliationOutcome.REJECTED,
            ReconciliationOutcome.PARTIALLY_SUPPORTED,
            ReconciliationOutcome.UNCERTAIN,
        ):
            raise SemanticWireError("Materialization requires a persistable resolution.")
        target = previous_acs.response_targets[target_id]
        if target.active_content_id != materialization.previous_active_content_id:
            raise SemanticWireError("Materialization target/content mismatch.")
        row: dict[str, Any] = {
            "resolution_target_id": target_id,
            "previous_active_content_id": materialization.previous_active_content_id,
            "subject_kind": materialization.subject_kind,
            "kind": None,
            "content": None,
            "source_items": [],
            "target_items": [],
            "meaning": None,
        }
        if isinstance(materialization, _ItemMaterialization):
            row.update(kind=materialization.kind.value, content=materialization.content)
        else:
            row.update(
                source_items=[item_ref(ref) for ref in materialization.source_items],
                target_items=[item_ref(ref) for ref in materialization.target_items],
                meaning=materialization.meaning,
            )
        materializations.append(row)

    def action_content(content: _Content) -> dict[str, Any]:
        return {
            "local_id": local(content.handle, "content"),
            "kind": content.kind.value,
            "semantic_content": content.semantic_content,
            "model_item_refs": [item_ref(ref) for ref in content.model_item_refs],
            "relation_refs": [relation_ref(ref) for ref in content.relation_refs],
        }

    targets: list[dict[str, Any]] = []
    for index, target in enumerate(selected.response_targets, start=1):
        subject = target.subject
        subject_row: dict[str, Any] = {
            "kind": None,
            "active_content_local_id": None,
            "item_reference": None,
            "relation_reference": None,
        }
        if isinstance(subject, _ContentSubject):
            subject_row.update(
                kind="active_content",
                active_content_local_id=local(subject.content_handle, "content"),
            )
        elif isinstance(subject, _ItemSubject):
            subject_row.update(kind="model_item", item_reference=item_ref(subject.item))
        else:
            subject_row.update(
                kind="relation", relation_reference=relation_ref(subject.relation)
            )
        targets.append(
            {
                "local_id": f"target:{index}",
                "active_content_local_id": local(target.owner_content, "content"),
                "subject": subject_row,
                "interaction": target.interaction.value,
            }
        )

    segments = [
        {
            "local_id": f"segment:{index}",
            "text": segment.text,
            "realizes_action_content_ids": [local(handle, "content") for handle in segment.realizes],
        }
        for index, segment in enumerate(turn.reply_segments, start=1)
    ]
    internal = {
        "decision_intent": selected.decision_intent.value,
        "state_patch": {
            "item_operations": item_operations,
            "relation_operations": relation_operations,
            "proposal_materializations": materializations,
        },
        "reconciliation": resolutions,
        "system_action": {
            "contents": [action_content(content) for content in contents],
            "response_targets": targets,
        },
        "reply_segments": segments,
    }
    try:
        converted = _resolve_source_quotes(internal, current_user_text)
        return CognitiveTurnResult.model_validate(converted)
    except (CognitiveCoreError, ValidationError) as error:
        raise SemanticWireError("Semantic output violates CognitiveTurnResult.") from error
