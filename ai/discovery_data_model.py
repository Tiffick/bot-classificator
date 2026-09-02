"""Isolated, persistent data contract for future Discovery lifecycle work.

This module intentionally has no imports from the active V2 runtime.  It defines
only validated data structures; Integration, Decision and persistence behaviour
remain outside its scope.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for persisted Discovery objects."""
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    """Create a stable, serialisable identifier with a semantic object prefix."""
    return f"{prefix}_{uuid4().hex}"


def _validate_prefixed_uuid(value: str, prefix: str) -> str:
    expected_prefix = f"{prefix}_"
    if not value.startswith(expected_prefix):
        raise ValueError(f"ID must start with '{expected_prefix}'.")
    try:
        UUID(hex=value.removeprefix(expected_prefix))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"ID after '{expected_prefix}' must be a UUID.") from error
    return value


class DiscoveryModel(BaseModel):
    """Strict base class for the Discovery persistence contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelItemKind(str, Enum):
    STATED_PROBLEM = "stated_problem"
    EXPERIENCE = "experience"
    LIFE_CHANGE = "life_change"
    BEFORE = "before"
    NOW = "now"
    EMOTION = "emotion"
    PERSONAL_MEANING = "personal_meaning"
    DESIRED_CHANGE = "desired_change"
    PREVIOUS_ATTEMPT = "previous_attempt"
    BARRIER = "barrier"
    USER_INTERPRETATION = "user_interpretation"
    OTHER_PERSON_VIEW = "other_person_view"
    OTHER_RELEVANT_USER_MATERIAL = "other_relevant_user_material"


class Provenance(str, Enum):
    USER_PROVIDED = "user_provided"
    USER_INTERPRETATION = "user_interpretation"
    SYSTEM_PROPOSED = "system_proposed"
    OTHER_PERSON = "other_person"


class ItemStatus(str, Enum):
    """Epistemic state, separate from the origin of material."""

    RECORDED = "recorded"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    CORRECTED = "corrected"


class DialogueRole(str, Enum):
    USER = "user"
    SYSTEM = "system"


class SourceReference(DiscoveryModel):
    message_id: str
    char_start: Optional[int] = Field(default=None, ge=0)
    char_end: Optional[int] = Field(default=None, ge=0)

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "msg")

    @model_validator(mode="after")
    def validate_offsets(self) -> "SourceReference":
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must not be earlier than char_start.")
        return self


class DialogueMessage(DiscoveryModel):
    id: str = Field(default_factory=lambda: make_id("msg"))
    role: DialogueRole
    text: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "msg")


class DialogueHistory(DiscoveryModel):
    messages: list[DialogueMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_message_sequences(self) -> "DialogueHistory":
        sequences = [message.sequence for message in self.messages]
        if len(sequences) != len(set(sequences)):
            raise ValueError("Dialogue message sequences must be unique.")
        return self


class ModelItem(DiscoveryModel):
    id: str = Field(default_factory=lambda: make_id("mi"))
    kind: ModelItemKind
    content: str = Field(min_length=1)
    provenance: Provenance
    status: ItemStatus
    source_refs: list[SourceReference] = Field(min_length=1)
    supersedes_item_ids: list[str] = Field(default_factory=list)
    created_turn: int = Field(ge=1)
    updated_turn: int = Field(ge=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "mi")

    @field_validator("supersedes_item_ids")
    @classmethod
    def validate_superseded_ids(cls, values: list[str]) -> list[str]:
        return [_validate_prefixed_uuid(value, "mi") for value in values]

    @model_validator(mode="after")
    def validate_turn_order(self) -> "ModelItem":
        if self.updated_turn < self.created_turn:
            raise ValueError("updated_turn must not be earlier than created_turn.")
        if self.id in self.supersedes_item_ids:
            raise ValueError("A ModelItem cannot supersede itself.")
        return self


class Relation(DiscoveryModel):
    id: str = Field(default_factory=lambda: make_id("rel"))
    meaning: str = Field(min_length=1)
    source_item_ids: list[str] = Field(min_length=1)
    target_item_ids: list[str] = Field(min_length=1)
    provenance: Provenance
    status: ItemStatus
    source_refs: list[SourceReference] = Field(min_length=1)
    supersedes_relation_ids: list[str] = Field(default_factory=list)
    created_turn: int = Field(ge=1)
    updated_turn: int = Field(ge=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "rel")

    @field_validator("source_item_ids", "target_item_ids")
    @classmethod
    def validate_item_ids(cls, values: list[str]) -> list[str]:
        return [_validate_prefixed_uuid(value, "mi") for value in values]

    @field_validator("supersedes_relation_ids")
    @classmethod
    def validate_superseded_ids(cls, values: list[str]) -> list[str]:
        return [_validate_prefixed_uuid(value, "rel") for value in values]

    @model_validator(mode="after")
    def validate_turn_order(self) -> "Relation":
        if self.updated_turn < self.created_turn:
            raise ValueError("updated_turn must not be earlier than created_turn.")
        if self.id in self.supersedes_relation_ids:
            raise ValueError("A Relation cannot supersede itself.")
        return self


class HumanModel(DiscoveryModel):
    """Accumulated user-specific understanding, not a questionnaire profile."""

    schema_version: Literal[1] = 1
    items: dict[str, ModelItem] = Field(default_factory=dict)
    relations: dict[str, Relation] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_references(self) -> "HumanModel":
        for key, item in self.items.items():
            if key != item.id:
                raise ValueError("HumanModel item key must equal ModelItem.id.")
            for superseded_id in item.supersedes_item_ids:
                if superseded_id not in self.items:
                    raise ValueError("ModelItem supersedes_item_ids must reference HumanModel.items.")

        for key, relation in self.relations.items():
            if key != relation.id:
                raise ValueError("HumanModel relation key must equal Relation.id.")
            for item_id in relation.source_item_ids + relation.target_item_ids:
                if item_id not in self.items:
                    raise ValueError("Relation item references must exist in HumanModel.items.")
            for superseded_id in relation.supersedes_relation_ids:
                if superseded_id not in self.relations:
                    raise ValueError(
                        "Relation supersedes_relation_ids must reference HumanModel.relations."
                    )
        return self


class ActiveContentKind(str, Enum):
    SYSTEM_QUESTION = "system_question"
    SYSTEM_PROPOSAL = "system_proposal"
    SYSTEM_REFLECTION = "system_reflection"
    RECOGNITION_OPTION = "recognition_option"
    SYSTEM_TRANSITION = "system_transition"
    SYSTEM_STATEMENT = "system_statement"


class ActiveContentItem(DiscoveryModel):
    id: str = Field(default_factory=lambda: make_id("aci"))
    kind: ActiveContentKind
    content: str = Field(min_length=1)
    model_item_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    source_message_id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "aci")

    @field_validator("model_item_ids")
    @classmethod
    def validate_model_item_ids(cls, values: list[str]) -> list[str]:
        return [_validate_prefixed_uuid(value, "mi") for value in values]

    @field_validator("relation_ids")
    @classmethod
    def validate_relation_ids(cls, values: list[str]) -> list[str]:
        return [_validate_prefixed_uuid(value, "rel") for value in values]

    @field_validator("source_message_id")
    @classmethod
    def validate_message_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "msg")


class TargetSubjectKind(str, Enum):
    ACTIVE_CONTENT = "active_content"
    MODEL_ITEM = "model_item"
    RELATION = "relation"


class TargetInteractionKind(str, Enum):
    OPEN_RESPONSE = "open_response"
    CLARIFICATION = "clarification"
    EVALUATION = "evaluation"
    CONSENT = "consent"


class TargetReference(DiscoveryModel):
    kind: TargetSubjectKind
    id: str

    @model_validator(mode="after")
    def validate_id_prefix(self) -> "TargetReference":
        prefixes = {
            TargetSubjectKind.ACTIVE_CONTENT: "aci",
            TargetSubjectKind.MODEL_ITEM: "mi",
            TargetSubjectKind.RELATION: "rel",
        }
        _validate_prefixed_uuid(self.id, prefixes[self.kind])
        return self


class ResponseTarget(DiscoveryModel):
    """A semantic referent for the next user message; not a result log."""

    id: str = Field(default_factory=lambda: make_id("rt"))
    active_content_id: str
    subject: TargetReference
    interaction: TargetInteractionKind

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "rt")

    @field_validator("active_content_id")
    @classmethod
    def validate_active_content_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "aci")


class ActiveConversationState(DiscoveryModel):
    """Persistent semantic bookmark for exactly the preceding system action."""

    id: str = Field(default_factory=lambda: make_id("acs"))
    created_turn: int = Field(ge=1)
    source_system_message_id: str
    active_content: dict[str, ActiveContentItem] = Field(min_length=1)
    response_targets: dict[str, ResponseTarget] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "acs")

    @field_validator("source_system_message_id")
    @classmethod
    def validate_message_id(cls, value: str) -> str:
        return _validate_prefixed_uuid(value, "msg")

    @model_validator(mode="after")
    def validate_internal_references(self) -> "ActiveConversationState":
        for key, content in self.active_content.items():
            if key != content.id:
                raise ValueError(
                    "ActiveConversationState content key must equal ActiveContentItem.id."
                )

        for key, target in self.response_targets.items():
            if key != target.id:
                raise ValueError(
                    "ActiveConversationState target key must equal ResponseTarget.id."
                )
            if target.active_content_id not in self.active_content:
                raise ValueError(
                    "ResponseTarget.active_content_id must reference active_content."
                )
            if (
                target.subject.kind == TargetSubjectKind.ACTIVE_CONTENT
                and target.subject.id not in self.active_content
            ):
                raise ValueError(
                    "ACTIVE_CONTENT TargetReference must reference active_content."
                )
        return self


def validate_acs_human_model_references(
    active_state: ActiveConversationState,
    human_model: HumanModel,
) -> None:
    """Validate cross-object ACS references without a global registry."""
    for content in active_state.active_content.values():
        for item_id in content.model_item_ids:
            if item_id not in human_model.items:
                raise ValueError("ActiveContentItem model_item_ids must exist in HumanModel.items.")
        for relation_id in content.relation_ids:
            if relation_id not in human_model.relations:
                raise ValueError(
                    "ActiveContentItem relation_ids must exist in HumanModel.relations."
                )

    for target in active_state.response_targets.values():
        if (
            target.subject.kind == TargetSubjectKind.MODEL_ITEM
            and target.subject.id not in human_model.items
        ):
            raise ValueError("MODEL_ITEM TargetReference must exist in HumanModel.items.")
        if (
            target.subject.kind == TargetSubjectKind.RELATION
            and target.subject.id not in human_model.relations
        ):
            raise ValueError("RELATION TargetReference must exist in HumanModel.relations.")
