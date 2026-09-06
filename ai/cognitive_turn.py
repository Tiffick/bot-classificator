"""Ephemeral contract for one proposed Cognitive Core turn.

The models in this module are intentionally not persistent Discovery memory and
do not create persistent identifiers.  They describe a proposal that must be
validated and applied by :mod:`ai.discovery_state_applier`.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai.discovery_data_model import (
    ActiveContentKind,
    ModelItemKind,
    TargetInteractionKind,
    TargetSubjectKind,
)


class CognitiveTurnContractError(ValueError):
    """Raised when an ephemeral cognitive proposal violates its contract."""


class EphemeralModel(BaseModel):
    """Strict, immutable base for non-persistent Cognitive Core proposals."""

    model_config = ConfigDict(extra="forbid", frozen=True)


_PERSISTENT_PREFIXES = ("mi_", "rel_", "acs_", "aci_", "rt_", "msg_")
_LOCAL_ID_PATTERN = re.compile(r"^[a-z]+:[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_local_id(value: str, namespace: str) -> str:
    if not value.startswith(f"{namespace}:") or not _LOCAL_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Local ID must use the '{namespace}:…' namespace.")
    if value.startswith(_PERSISTENT_PREFIXES):
        raise ValueError("Local IDs must not use persistent ID prefixes.")
    return value


def _validate_persistent_id(value: str, prefix: str) -> str:
    expected_prefix = f"{prefix}_"
    if not value.startswith(expected_prefix):
        raise ValueError(f"Persistent ID must start with '{expected_prefix}'.")
    try:
        UUID(hex=value.removeprefix(expected_prefix))
    except (AttributeError, ValueError) as error:
        raise ValueError("Persistent ID must contain a UUID.") from error
    return value


def _validate_unique(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate IDs.")
    return values


class EvidenceOrigin(str, Enum):
    CURRENT_USER_MATERIAL = "current_user_material"
    CURRENT_USER_INTERPRETATION = "current_user_interpretation"
    CURRENT_OTHER_PERSON_REPORT = "current_other_person_report"


class SourceSpan(EphemeralModel):
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SourceSpan":
        if self.char_end <= self.char_start:
            raise ValueError("SourceSpan must cover at least one character.")
        return self


class ItemOperationKind(str, Enum):
    ADD = "add"
    REINFORCE = "reinforce"
    CORRECT = "correct"


class ItemOperation(EphemeralModel):
    operation: ItemOperationKind
    local_id: Optional[str] = None
    existing_item_id: Optional[str] = None
    kind: Optional[ModelItemKind] = None
    content: Optional[str] = None
    evidence_origin: Optional[EvidenceOrigin] = None
    source_span: Optional[SourceSpan] = None

    @field_validator("local_id")
    @classmethod
    def validate_local_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_local_id(value, "item")

    @field_validator("existing_item_id")
    @classmethod
    def validate_existing_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_persistent_id(value, "mi")

    @model_validator(mode="after")
    def validate_operation_fields(self) -> "ItemOperation":
        has_new_content = any(
            value is not None
            for value in (self.kind, self.content)
        )
        has_evidence = self.evidence_origin is not None and self.source_span is not None
        if self.operation == ItemOperationKind.ADD:
            if self.local_id is None or self.existing_item_id is not None:
                raise ValueError("ADD requires local_id and forbids existing_item_id.")
            if not has_new_content or not has_evidence:
                raise ValueError("ADD requires kind, content, evidence_origin and source_span.")
        elif self.operation == ItemOperationKind.REINFORCE:
            if self.existing_item_id is None or self.local_id is not None:
                raise ValueError("REINFORCE requires existing_item_id and forbids local_id.")
            if has_new_content:
                raise ValueError("REINFORCE must not change kind or content.")
            if not has_evidence:
                raise ValueError("REINFORCE requires evidence_origin and source_span.")
        else:
            if self.local_id is None or self.existing_item_id is None:
                raise ValueError("CORRECT requires local_id and existing_item_id.")
            if not has_new_content or not has_evidence:
                raise ValueError("CORRECT requires kind, content, evidence_origin and source_span.")
        return self


class ItemReference(EphemeralModel):
    existing_item_id: Optional[str] = None
    local_item_id: Optional[str] = None

    @field_validator("existing_item_id")
    @classmethod
    def validate_existing_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_persistent_id(value, "mi")

    @field_validator("local_item_id")
    @classmethod
    def validate_local_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_local_id(value, "item")

    @model_validator(mode="after")
    def validate_exactly_one_reference(self) -> "ItemReference":
        if (self.existing_item_id is None) == (self.local_item_id is None):
            raise ValueError("ItemReference requires exactly one existing or local item ID.")
        return self


class RelationReference(EphemeralModel):
    existing_relation_id: Optional[str] = None
    local_relation_id: Optional[str] = None

    @field_validator("existing_relation_id")
    @classmethod
    def validate_existing_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_persistent_id(value, "rel")

    @field_validator("local_relation_id")
    @classmethod
    def validate_local_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_local_id(value, "relation")

    @model_validator(mode="after")
    def validate_exactly_one_reference(self) -> "RelationReference":
        if (self.existing_relation_id is None) == (self.local_relation_id is None):
            raise ValueError("RelationReference requires exactly one existing or local relation ID.")
        return self


class RelationOperationKind(str, Enum):
    ADD = "add"
    REINFORCE = "reinforce"
    CORRECT = "correct"


class RelationOperation(EphemeralModel):
    operation: RelationOperationKind
    local_id: Optional[str] = None
    existing_relation_id: Optional[str] = None
    source_items: tuple[ItemReference, ...] = ()
    target_items: tuple[ItemReference, ...] = ()
    meaning: Optional[str] = None
    evidence_origin: Optional[EvidenceOrigin] = None
    source_span: Optional[SourceSpan] = None

    @field_validator("local_id")
    @classmethod
    def validate_local_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_local_id(value, "relation")

    @field_validator("existing_relation_id")
    @classmethod
    def validate_existing_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_persistent_id(value, "rel")

    @model_validator(mode="after")
    def validate_operation_fields(self) -> "RelationOperation":
        has_relation_content = bool(self.source_items and self.target_items and self.meaning)
        has_evidence = self.evidence_origin is not None and self.source_span is not None
        if self.operation == RelationOperationKind.ADD:
            if self.local_id is None or self.existing_relation_id is not None:
                raise ValueError("ADD requires local_id and forbids existing_relation_id.")
            if not has_relation_content or not has_evidence:
                raise ValueError("ADD requires endpoints, meaning, evidence_origin and source_span.")
        elif self.operation == RelationOperationKind.REINFORCE:
            if self.existing_relation_id is None or self.local_id is not None:
                raise ValueError("REINFORCE requires existing_relation_id and forbids local_id.")
            if self.source_items or self.target_items or self.meaning is not None:
                raise ValueError("REINFORCE must not change endpoints or meaning.")
            if not has_evidence:
                raise ValueError("REINFORCE requires evidence_origin and source_span.")
        else:
            if self.local_id is None or self.existing_relation_id is None:
                raise ValueError("CORRECT requires local_id and existing_relation_id.")
            if not has_relation_content or not has_evidence:
                raise ValueError("CORRECT requires endpoints, meaning, evidence_origin and source_span.")
        return self


class ReconciliationOutcome(str, Enum):
    ANSWERED = "answered"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNCERTAIN = "uncertain"
    AMBIGUOUS = "ambiguous"
    CONSENTED = "consented"
    DECLINED = "declined"
    REFUSED = "refused"


class TargetResolution(EphemeralModel):
    previous_response_target_ids: tuple[str, ...] = Field(min_length=1)
    outcome: ReconciliationOutcome
    source_span: Optional[SourceSpan] = None

    @field_validator("previous_response_target_ids")
    @classmethod
    def validate_target_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(_validate_persistent_id(value, "rt") for value in values)
        return _validate_unique(validated, "previous_response_target_ids")

    @model_validator(mode="after")
    def validate_target_count(self) -> "TargetResolution":
        count = len(self.previous_response_target_ids)
        if self.outcome == ReconciliationOutcome.AMBIGUOUS:
            if count < 2:
                raise ValueError("AMBIGUOUS requires multiple candidate response targets.")
        elif count != 1:
            raise ValueError("Only AMBIGUOUS may reference multiple response targets.")
        return self


class ProposalMaterialization(EphemeralModel):
    resolution_target_id: str
    previous_active_content_id: str
    subject_kind: Literal["item", "relation"]
    kind: Optional[ModelItemKind] = None
    content: Optional[str] = None
    source_items: tuple[ItemReference, ...] = ()
    target_items: tuple[ItemReference, ...] = ()
    meaning: Optional[str] = None

    @field_validator("resolution_target_id")
    @classmethod
    def validate_target_id(cls, value: str) -> str:
        return _validate_persistent_id(value, "rt")

    @field_validator("previous_active_content_id")
    @classmethod
    def validate_content_id(cls, value: str) -> str:
        return _validate_persistent_id(value, "aci")

    @model_validator(mode="after")
    def validate_subject_fields(self) -> "ProposalMaterialization":
        if self.subject_kind == "item":
            if self.kind is None or not self.content:
                raise ValueError("Item materialization requires kind and content.")
            if self.source_items or self.target_items or self.meaning is not None:
                raise ValueError("Item materialization must not define relation fields.")
        else:
            if self.kind is not None or self.content is not None:
                raise ValueError("Relation materialization must not define item fields.")
            if not self.source_items or not self.target_items or not self.meaning:
                raise ValueError("Relation materialization requires endpoints and meaning.")
        return self


class ActionSubjectReference(EphemeralModel):
    kind: TargetSubjectKind
    active_content_local_id: Optional[str] = None
    item_reference: Optional[ItemReference] = None
    relation_reference: Optional[RelationReference] = None

    @field_validator("active_content_local_id")
    @classmethod
    def validate_content_id(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _validate_local_id(value, "content")

    @model_validator(mode="after")
    def validate_subject(self) -> "ActionSubjectReference":
        supplied = sum(
            value is not None
            for value in (
                self.active_content_local_id,
                self.item_reference,
                self.relation_reference,
            )
        )
        if supplied != 1:
            raise ValueError("ActionSubjectReference requires exactly one reference.")
        expected_field = {
            TargetSubjectKind.ACTIVE_CONTENT: self.active_content_local_id,
            TargetSubjectKind.MODEL_ITEM: self.item_reference,
            TargetSubjectKind.RELATION: self.relation_reference,
        }[self.kind]
        if expected_field is None:
            raise ValueError("ActionSubjectReference kind does not match its reference.")
        return self


class ActionContent(EphemeralModel):
    local_id: str
    kind: ActiveContentKind
    semantic_content: str = Field(min_length=1)
    model_item_refs: tuple[ItemReference, ...] = ()
    relation_refs: tuple[RelationReference, ...] = ()

    @field_validator("local_id")
    @classmethod
    def validate_local_id(cls, value: str) -> str:
        return _validate_local_id(value, "content")


class ActionTarget(EphemeralModel):
    local_id: str
    active_content_local_id: str
    subject: ActionSubjectReference
    interaction: TargetInteractionKind

    @field_validator("local_id")
    @classmethod
    def validate_local_id(cls, value: str) -> str:
        return _validate_local_id(value, "target")

    @field_validator("active_content_local_id")
    @classmethod
    def validate_content_id(cls, value: str) -> str:
        return _validate_local_id(value, "content")


class SystemAction(EphemeralModel):
    contents: tuple[ActionContent, ...] = ()
    response_targets: tuple[ActionTarget, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> "SystemAction":
        content_ids = tuple(content.local_id for content in self.contents)
        target_ids = tuple(target.local_id for target in self.response_targets)
        _validate_unique(content_ids, "SystemAction content IDs")
        _validate_unique(target_ids, "SystemAction target IDs")
        known_contents = set(content_ids)
        for target in self.response_targets:
            if target.active_content_local_id not in known_contents:
                raise ValueError("ActionTarget must reference an ActionContent in SystemAction.")
            if (
                target.subject.kind == TargetSubjectKind.ACTIVE_CONTENT
                and target.subject.active_content_local_id not in known_contents
            ):
                raise ValueError("Action target subject must reference an ActionContent in SystemAction.")
        return self


class ReplySegment(EphemeralModel):
    local_id: str
    text: str = Field(min_length=1)
    realizes_action_content_ids: tuple[str, ...] = ()

    @field_validator("local_id")
    @classmethod
    def validate_local_id(cls, value: str) -> str:
        return _validate_local_id(value, "segment")

    @field_validator("realizes_action_content_ids")
    @classmethod
    def validate_content_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(_validate_local_id(value, "content") for value in values)
        return _validate_unique(validated, "realizes_action_content_ids")


class StatePatch(EphemeralModel):
    item_operations: tuple[ItemOperation, ...] = ()
    relation_operations: tuple[RelationOperation, ...] = ()
    proposal_materializations: tuple[ProposalMaterialization, ...] = ()


class CognitiveTurnResult(EphemeralModel):
    state_patch: StatePatch = Field(default_factory=StatePatch)
    reconciliation: tuple[TargetResolution, ...] = ()
    system_action: SystemAction = Field(default_factory=SystemAction)
    reply_segments: tuple[ReplySegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cross_references(self) -> "CognitiveTurnResult":
        item_local_ids = tuple(
            operation.local_id
            for operation in self.state_patch.item_operations
            if operation.local_id is not None
        )
        relation_local_ids = tuple(
            operation.local_id
            for operation in self.state_patch.relation_operations
            if operation.local_id is not None
        )
        segment_ids = tuple(segment.local_id for segment in self.reply_segments)
        _validate_unique(item_local_ids, "StatePatch item local IDs")
        _validate_unique(relation_local_ids, "StatePatch relation local IDs")
        _validate_unique(segment_ids, "ReplySegment local IDs")

        known_item_ids = set(item_local_ids)
        known_relation_ids = set(relation_local_ids)
        for operation in self.state_patch.relation_operations:
            for reference in operation.source_items + operation.target_items:
                if (
                    reference.local_item_id is not None
                    and reference.local_item_id not in known_item_ids
                ):
                    raise ValueError("Relation operation references an unknown local item.")
        for materialization in self.state_patch.proposal_materializations:
            for reference in materialization.source_items + materialization.target_items:
                if (
                    reference.local_item_id is not None
                    and reference.local_item_id not in known_item_ids
                ):
                    raise ValueError("Proposal materialization references an unknown local item.")
        for content in self.system_action.contents:
            for reference in content.model_item_refs:
                if (
                    reference.local_item_id is not None
                    and reference.local_item_id not in known_item_ids
                ):
                    raise ValueError("ActionContent references an unknown local item.")
            for reference in content.relation_refs:
                if (
                    reference.local_relation_id is not None
                    and reference.local_relation_id not in known_relation_ids
                ):
                    raise ValueError("ActionContent references an unknown local relation.")
        for target in self.system_action.response_targets:
            subject = target.subject
            if (
                subject.item_reference is not None
                and subject.item_reference.local_item_id is not None
                and subject.item_reference.local_item_id not in known_item_ids
            ):
                raise ValueError("ActionTarget references an unknown local item.")
            if (
                subject.relation_reference is not None
                and subject.relation_reference.local_relation_id is not None
                and subject.relation_reference.local_relation_id not in known_relation_ids
            ):
                raise ValueError("ActionTarget references an unknown local relation.")

        known_contents = {content.local_id for content in self.system_action.contents}
        realized_contents: set[str] = set()
        for segment in self.reply_segments:
            for content_id in segment.realizes_action_content_ids:
                if content_id not in known_contents:
                    raise ValueError("ReplySegment references an unknown ActionContent.")
                realized_contents.add(content_id)
        targeted_contents = {
            target.active_content_local_id
            for target in self.system_action.response_targets
        }
        if not targeted_contents.issubset(realized_contents):
            raise ValueError("Every targetable ActionContent must be realized by a ReplySegment.")

        resolved_targets: list[str] = []
        for resolution in self.reconciliation:
            resolved_targets.extend(resolution.previous_response_target_ids)
        _validate_unique(tuple(resolved_targets), "Reconciliation response target IDs")
        return self
