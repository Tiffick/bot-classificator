"""Deterministic application of an ephemeral Cognitive Core proposal.

This module deliberately has no LLM, memory-store, Telegram or DialogEngine
dependencies.  It transforms immutable Discovery contract objects or rejects
the entire proposed semantic transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from ai.cognitive_turn import (
    CognitiveTurnResult,
    EvidenceOrigin,
    ItemOperationKind,
    ReconciliationOutcome,
    RelationOperationKind,
)
from ai.discovery_data_model import (
    ActiveContentItem,
    ActiveConversationState,
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    ItemStatus,
    ModelItem,
    Provenance,
    Relation,
    ResponseTarget,
    SourceReference,
    TargetReference,
    TargetSubjectKind,
    make_id,
    validate_discovery_memory,
)


class DiscoveryStateApplyError(ValueError):
    """The complete proposed turn is invalid and must not be partially applied."""


@dataclass(frozen=True)
class AppliedTurn:
    updated_human_model: HumanModel
    new_acs: Optional[ActiveConversationState]
    updated_history: DialogueHistory
    item_id_mapping: dict[str, str]
    relation_id_mapping: dict[str, str]
    active_content_id_mapping: dict[str, str]
    response_target_id_mapping: dict[str, str]


_PROVENANCE_BY_EVIDENCE = {
    EvidenceOrigin.CURRENT_USER_MATERIAL: Provenance.USER_PROVIDED,
    EvidenceOrigin.CURRENT_USER_INTERPRETATION: Provenance.USER_INTERPRETATION,
    EvidenceOrigin.CURRENT_OTHER_PERSON_REPORT: Provenance.OTHER_PERSON,
}

_MATERIALIZATION_STATUSES = {
    ReconciliationOutcome.SUPPORTED: ItemStatus.SUPPORTED,
    ReconciliationOutcome.REJECTED: ItemStatus.REJECTED,
    ReconciliationOutcome.PARTIALLY_SUPPORTED: ItemStatus.PARTIALLY_SUPPORTED,
    ReconciliationOutcome.UNCERTAIN: ItemStatus.UNCERTAIN,
}


def _current_source(user_message: DialogueMessage, source_span) -> SourceReference:
    return SourceReference(
        message_id=user_message.id,
        char_start=source_span.char_start,
        char_end=source_span.char_end,
    )


def _validate_span(source_span, user_message: DialogueMessage) -> None:
    if source_span is None or source_span.char_end > len(user_message.text):
        raise DiscoveryStateApplyError("Source span must be inside the current user message.")


def _derive_provenance(evidence_origin: EvidenceOrigin, kind) -> Provenance:
    provenance = _PROVENANCE_BY_EVIDENCE[evidence_origin]
    if provenance == Provenance.OTHER_PERSON and kind.value != "other_person_view":
        raise DiscoveryStateApplyError(
            "CURRENT_OTHER_PERSON_REPORT requires OTHER_PERSON_VIEW material."
        )
    if kind.value == "other_person_view" and provenance != Provenance.OTHER_PERSON:
        raise DiscoveryStateApplyError(
            "OTHER_PERSON_VIEW material requires CURRENT_OTHER_PERSON_REPORT."
        )
    return provenance


def _updated_item(item: ModelItem, *, status: ItemStatus, source_ref=None, turn: int) -> ModelItem:
    source_refs = item.source_refs
    if source_ref is not None and source_ref not in source_refs:
        source_refs = (*source_refs, source_ref)
    return ModelItem(
        id=item.id,
        kind=item.kind,
        content=item.content,
        provenance=item.provenance,
        status=status,
        source_refs=source_refs,
        supersedes_item_ids=item.supersedes_item_ids,
        created_turn=item.created_turn,
        updated_turn=turn,
    )


def _updated_relation(relation: Relation, *, status: ItemStatus, source_ref=None, turn: int) -> Relation:
    source_refs = relation.source_refs
    if source_ref is not None and source_ref not in source_refs:
        source_refs = (*source_refs, source_ref)
    return Relation(
        id=relation.id,
        meaning=relation.meaning,
        source_item_ids=relation.source_item_ids,
        target_item_ids=relation.target_item_ids,
        provenance=relation.provenance,
        status=status,
        source_refs=source_refs,
        supersedes_relation_ids=relation.supersedes_relation_ids,
        created_turn=relation.created_turn,
        updated_turn=turn,
    )


def _resolve_item_reference(reference, item_ids: dict[str, str], items: dict[str, ModelItem]) -> str:
    if reference.existing_item_id is not None:
        if reference.existing_item_id not in items:
            raise DiscoveryStateApplyError("Item reference does not exist in HumanModel.")
        return reference.existing_item_id
    try:
        return item_ids[reference.local_item_id]
    except KeyError as error:
        raise DiscoveryStateApplyError("Local item reference was not reserved.") from error


def _resolve_relation_reference(reference, relation_ids: dict[str, str], relations: dict[str, Relation]) -> str:
    if reference.existing_relation_id is not None:
        if reference.existing_relation_id not in relations:
            raise DiscoveryStateApplyError("Relation reference does not exist in HumanModel.")
        return reference.existing_relation_id
    try:
        return relation_ids[reference.local_relation_id]
    except KeyError as error:
        raise DiscoveryStateApplyError("Local relation reference was not reserved.") from error


def _validate_message_pair(history, user_message, system_message) -> None:
    if user_message.role != DialogueRole.USER:
        raise DiscoveryStateApplyError("user_message must have USER role.")
    if system_message.role != DialogueRole.SYSTEM:
        raise DiscoveryStateApplyError("system_message must have SYSTEM role.")
    expected_user_sequence = history.messages[-1].sequence + 1 if history.messages else 1
    if user_message.sequence != expected_user_sequence:
        raise DiscoveryStateApplyError("USER message sequence is not next in DialogueHistory.")
    if system_message.sequence != user_message.sequence + 1:
        raise DiscoveryStateApplyError("SYSTEM message sequence must immediately follow USER message.")


def _validate_reconciliation(previous_acs, turn_result: CognitiveTurnResult, user_message) -> dict[str, object]:
    if previous_acs is None:
        if turn_result.reconciliation or turn_result.state_patch.proposal_materializations:
            raise DiscoveryStateApplyError("Reconciliation and proposal materialization require previous ACS.")
        return {}

    targets = previous_acs.response_targets
    resolutions: dict[str, object] = {}
    interaction_by_outcome = {
        ReconciliationOutcome.ANSWERED: {"open_response", "clarification"},
        ReconciliationOutcome.SUPPORTED: {"evaluation", "clarification"},
        ReconciliationOutcome.REJECTED: {"evaluation", "clarification"},
        ReconciliationOutcome.PARTIALLY_SUPPORTED: {"evaluation", "clarification"},
        ReconciliationOutcome.UNCERTAIN: {"evaluation", "clarification"},
        ReconciliationOutcome.CONSENTED: {"consent"},
        ReconciliationOutcome.DECLINED: {"consent"},
    }
    for resolution in turn_result.reconciliation:
        if resolution.source_span is not None:
            _validate_span(resolution.source_span, user_message)
        for target_id in resolution.previous_response_target_ids:
            if target_id not in targets:
                raise DiscoveryStateApplyError("Reconciliation references an unknown previous target.")
            target = targets[target_id]
            if target_id in resolutions:
                raise DiscoveryStateApplyError("Previous target has more than one resolution.")
            if resolution.outcome in interaction_by_outcome:
                if target.interaction.value not in interaction_by_outcome[resolution.outcome]:
                    raise DiscoveryStateApplyError("Reconciliation outcome is incompatible with target interaction.")
            resolutions[target_id] = resolution
    return resolutions


def _validate_operations(turn_result: CognitiveTurnResult, human_model, user_message) -> None:
    for operation in turn_result.state_patch.item_operations:
        _validate_span(operation.source_span, user_message)
        if operation.operation in (ItemOperationKind.REINFORCE, ItemOperationKind.CORRECT):
            if operation.existing_item_id not in human_model.items:
                raise DiscoveryStateApplyError("Item operation references an unknown existing item.")
        if operation.operation != ItemOperationKind.REINFORCE:
            _derive_provenance(operation.evidence_origin, operation.kind)
    for operation in turn_result.state_patch.relation_operations:
        _validate_span(operation.source_span, user_message)
        if operation.operation in (RelationOperationKind.REINFORCE, RelationOperationKind.CORRECT):
            if operation.existing_relation_id not in human_model.relations:
                raise DiscoveryStateApplyError("Relation operation references an unknown existing relation.")
        if operation.operation != RelationOperationKind.REINFORCE:
            _PROVENANCE_BY_EVIDENCE[operation.evidence_origin]


def _reserve_ids(turn_result: CognitiveTurnResult) -> tuple[dict[str, str], dict[str, str]]:
    item_ids: dict[str, str] = {}
    relation_ids: dict[str, str] = {}
    for operation in turn_result.state_patch.item_operations:
        if operation.operation in (ItemOperationKind.ADD, ItemOperationKind.CORRECT):
            item_ids[operation.local_id] = make_id("mi")
    for operation in turn_result.state_patch.relation_operations:
        if operation.operation in (RelationOperationKind.ADD, RelationOperationKind.CORRECT):
            relation_ids[operation.local_id] = make_id("rel")
    return item_ids, relation_ids


def _apply_items(turn_result, items, item_ids, user_message, turn) -> None:
    for operation in turn_result.state_patch.item_operations:
        source_ref = _current_source(user_message, operation.source_span)
        if operation.operation == ItemOperationKind.ADD:
            items[item_ids[operation.local_id]] = ModelItem(
                id=item_ids[operation.local_id],
                kind=operation.kind,
                content=operation.content,
                provenance=_derive_provenance(operation.evidence_origin, operation.kind),
                status=ItemStatus.RECORDED,
                source_refs=(source_ref,),
                created_turn=turn,
                updated_turn=turn,
            )
        elif operation.operation == ItemOperationKind.REINFORCE:
            existing = items[operation.existing_item_id]
            items[existing.id] = _updated_item(
                existing, status=existing.status, source_ref=source_ref, turn=turn
            )
        else:
            existing = items[operation.existing_item_id]
            items[existing.id] = _updated_item(
                existing, status=ItemStatus.CORRECTED, turn=turn
            )
            items[item_ids[operation.local_id]] = ModelItem(
                id=item_ids[operation.local_id],
                kind=operation.kind,
                content=operation.content,
                provenance=_derive_provenance(operation.evidence_origin, operation.kind),
                status=ItemStatus.RECORDED,
                source_refs=(source_ref,),
                supersedes_item_ids=(existing.id,),
                created_turn=turn,
                updated_turn=turn,
            )


def _apply_relations(turn_result, items, relations, item_ids, relation_ids, user_message, turn) -> None:
    for operation in turn_result.state_patch.relation_operations:
        source_ref = _current_source(user_message, operation.source_span)
        if operation.operation == RelationOperationKind.REINFORCE:
            existing = relations[operation.existing_relation_id]
            relations[existing.id] = _updated_relation(
                existing, status=existing.status, source_ref=source_ref, turn=turn
            )
            continue
        source_ids = tuple(
            _resolve_item_reference(reference, item_ids, items)
            for reference in operation.source_items
        )
        target_ids = tuple(
            _resolve_item_reference(reference, item_ids, items)
            for reference in operation.target_items
        )
        if operation.operation == RelationOperationKind.CORRECT:
            existing = relations[operation.existing_relation_id]
            relations[existing.id] = _updated_relation(
                existing, status=ItemStatus.CORRECTED, turn=turn
            )
            supersedes = (existing.id,)
        else:
            supersedes = ()
        relations[relation_ids[operation.local_id]] = Relation(
            id=relation_ids[operation.local_id],
            meaning=operation.meaning,
            source_item_ids=source_ids,
            target_item_ids=target_ids,
            provenance=_PROVENANCE_BY_EVIDENCE[operation.evidence_origin],
            status=ItemStatus.RECORDED,
            source_refs=(source_ref,),
            supersedes_relation_ids=supersedes,
            created_turn=turn,
            updated_turn=turn,
        )


def _apply_materializations(
    turn_result,
    resolutions,
    previous_acs,
    history,
    user_message,
    items,
    relations,
    item_ids,
    turn,
) -> None:
    if not turn_result.state_patch.proposal_materializations:
        return
    history_by_id = {message.id: message for message in history.messages}
    for materialization in turn_result.state_patch.proposal_materializations:
        resolution = resolutions.get(materialization.resolution_target_id)
        if resolution is None:
            raise DiscoveryStateApplyError("Proposal materialization requires a target resolution.")
        if resolution.outcome not in _MATERIALIZATION_STATUSES:
            raise DiscoveryStateApplyError("Proposal materialization outcome is not persistable.")
        target = previous_acs.response_targets[materialization.resolution_target_id]
        if target.active_content_id != materialization.previous_active_content_id:
            raise DiscoveryStateApplyError("Materialization target does not belong to active content.")
        active_content = previous_acs.active_content[materialization.previous_active_content_id]
        previous_system = history_by_id.get(active_content.source_message_id)
        if previous_system is None or previous_system.role != DialogueRole.SYSTEM:
            raise DiscoveryStateApplyError("Materialization source system message is unavailable.")
        source_refs = [SourceReference(message_id=previous_system.id)]
        if resolution.source_span is not None:
            source_refs.append(_current_source(user_message, resolution.source_span))
        status = _MATERIALIZATION_STATUSES[resolution.outcome]
        if materialization.subject_kind == "item":
            item_id = make_id("mi")
            items[item_id] = ModelItem(
                id=item_id,
                kind=materialization.kind,
                content=materialization.content,
                provenance=Provenance.SYSTEM_PROPOSED,
                status=status,
                source_refs=tuple(source_refs),
                created_turn=turn,
                updated_turn=turn,
            )
        else:
            relation_id = make_id("rel")
            source_ids = tuple(
                _resolve_item_reference(reference, item_ids, items)
                for reference in materialization.source_items
            )
            target_ids = tuple(
                _resolve_item_reference(reference, item_ids, items)
                for reference in materialization.target_items
            )
            relations[relation_id] = Relation(
                id=relation_id,
                meaning=materialization.meaning,
                source_item_ids=source_ids,
                target_item_ids=target_ids,
                provenance=Provenance.SYSTEM_PROPOSED,
                status=status,
                source_refs=tuple(source_refs),
                created_turn=turn,
                updated_turn=turn,
            )


def _build_acs(turn_result, system_message, items, relations, item_ids, relation_ids):
    if not turn_result.system_action.response_targets:
        return None, {}, {}
    targeted_content_ids = {
        target.active_content_local_id
        for target in turn_result.system_action.response_targets
    }
    content_id_mapping: dict[str, str] = {
        local_id: make_id("aci") for local_id in targeted_content_ids
    }
    active_content = {}
    content_by_local_id = {
        content.local_id: content for content in turn_result.system_action.contents
    }
    for local_id in targeted_content_ids:
        content = content_by_local_id[local_id]
        active_content[content_id_mapping[local_id]] = ActiveContentItem(
            id=content_id_mapping[local_id],
            kind=content.kind,
            content=content.semantic_content,
            model_item_ids=tuple(
                _resolve_item_reference(reference, item_ids, items)
                for reference in content.model_item_refs
            ),
            relation_ids=tuple(
                _resolve_relation_reference(reference, relation_ids, relations)
                for reference in content.relation_refs
            ),
            source_message_id=system_message.id,
        )
    target_id_mapping: dict[str, str] = {}
    response_targets = {}
    for target in turn_result.system_action.response_targets:
        persistent_target_id = make_id("rt")
        target_id_mapping[target.local_id] = persistent_target_id
        subject = target.subject
        if subject.kind == TargetSubjectKind.ACTIVE_CONTENT:
            subject_id = content_id_mapping[subject.active_content_local_id]
        elif subject.kind == TargetSubjectKind.MODEL_ITEM:
            subject_id = _resolve_item_reference(subject.item_reference, item_ids, items)
        else:
            subject_id = _resolve_relation_reference(subject.relation_reference, relation_ids, relations)
        response_targets[persistent_target_id] = ResponseTarget(
            id=persistent_target_id,
            active_content_id=content_id_mapping[target.active_content_local_id],
            subject=TargetReference(kind=subject.kind, id=subject_id),
            interaction=target.interaction,
        )
    return (
        ActiveConversationState(
            created_turn=(system_message.sequence + 1) // 2,
            source_system_message_id=system_message.id,
            active_content=active_content,
            response_targets=response_targets,
        ),
        content_id_mapping,
        target_id_mapping,
    )


def apply_cognitive_turn(
    human_model: HumanModel,
    previous_acs: Optional[ActiveConversationState],
    history: DialogueHistory,
    user_message: DialogueMessage,
    system_message: DialogueMessage,
    turn_result: CognitiveTurnResult,
) -> AppliedTurn:
    """Return a fully validated candidate turn without mutating any input."""
    try:
        _validate_message_pair(history, user_message, system_message)
        validate_discovery_memory(human_model, previous_acs, history)
        reply = "".join(segment.text for segment in turn_result.reply_segments)
        if system_message.text != reply:
            raise DiscoveryStateApplyError("SYSTEM message text must equal joined reply segments.")
        resolutions = _validate_reconciliation(previous_acs, turn_result, user_message)
        _validate_operations(turn_result, human_model, user_message)
        item_ids, relation_ids = _reserve_ids(turn_result)
        items = dict(human_model.items)
        relations = dict(human_model.relations)
        turn = (user_message.sequence + 1) // 2
        _apply_items(turn_result, items, item_ids, user_message, turn)
        _apply_relations(
            turn_result, items, relations, item_ids, relation_ids, user_message, turn
        )
        _apply_materializations(
            turn_result,
            resolutions,
            previous_acs,
            history,
            user_message,
            items,
            relations,
            item_ids,
            turn,
        )
        updated_human_model = human_model.with_updates(items=items, relations=relations)
        new_acs, content_ids, target_ids = _build_acs(
            turn_result,
            system_message,
            items,
            relations,
            item_ids,
            relation_ids,
        )
        updated_history = DialogueHistory(
            messages=(*history.messages, user_message, system_message)
        )
        validate_discovery_memory(updated_human_model, new_acs, updated_history)
        return AppliedTurn(
            updated_human_model=updated_human_model,
            new_acs=new_acs,
            updated_history=updated_history,
            item_id_mapping=item_ids,
            relation_id_mapping=relation_ids,
            active_content_id_mapping=content_ids,
            response_target_id_mapping=target_ids,
        )
    except (DiscoveryStateApplyError, ValidationError) as error:
        if isinstance(error, DiscoveryStateApplyError):
            raise
        raise DiscoveryStateApplyError(str(error)) from error
