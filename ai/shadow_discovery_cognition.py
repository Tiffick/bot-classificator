"""Isolated Stage 2A cognition for independently stated user material.

The module deliberately does not import or alter the legacy V2 engines.  The
LLM produces an ephemeral semantic proposal; Python is the only component that
creates or updates persistent Discovery Data Model objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai.discovery_data_model import (
    DialogueHistory,
    HumanModel,
    ItemStatus,
    ModelItem,
    ModelItemKind,
    Provenance,
    Relation,
    SourceReference,
)


class ShadowCognitionError(RuntimeError):
    """Raised when an ephemeral shadow proposal cannot be safely applied."""


class EphemeralModel(BaseModel):
    """Strict, immutable base for one-turn, non-persistent cognition data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MaterialOperation(str, Enum):
    NEW = "new"
    REINFORCE = "reinforce"
    CORRECT = "correct"
    UNRESOLVED = "unresolved"


class SourceSpan(EphemeralModel):
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SourceSpan":
        if self.char_end <= self.char_start:
            raise ValueError("source spans must cover at least one character.")
        return self


class PerceptionMaterial(EphemeralModel):
    local_id: str = Field(min_length=1)
    normalized_content: str = Field(min_length=1)
    kind: ModelItemKind
    provenance: Provenance
    source_span: SourceSpan
    operation: MaterialOperation
    existing_item_id: Optional[str] = None


class PerceptionReference(EphemeralModel):
    local_id: Optional[str] = None
    existing_item_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_exactly_one_reference(self) -> "PerceptionReference":
        if (self.local_id is None) == (self.existing_item_id is None):
            raise ValueError("A relation reference must name exactly one material.")
        return self


class PerceptionRelationCandidate(EphemeralModel):
    source_references: tuple[PerceptionReference, ...] = Field(min_length=1)
    target_references: tuple[PerceptionReference, ...] = Field(min_length=1)
    normalized_meaning: str = Field(min_length=1)
    provenance: Provenance
    source_span: SourceSpan


class InteractionSignals(EphemeralModel):
    direct_question: bool = False
    refusal_or_pause: bool = False
    topic_shift: bool = False
    ambiguity: bool = False


class PerceptionResult(EphemeralModel):
    materials: tuple[PerceptionMaterial, ...] = Field(default_factory=tuple)
    relation_candidates: tuple[PerceptionRelationCandidate, ...] = Field(
        default_factory=tuple
    )
    interaction_signals: InteractionSignals = Field(default_factory=InteractionSignals)


@dataclass(frozen=True)
class IntegrationResult:
    """Ephemeral record of a Python-applied immutable Human Model update."""

    human_model: HumanModel
    added_item_ids: tuple[str, ...] = ()
    reinforced_item_ids: tuple[str, ...] = ()
    corrected_item_ids: tuple[str, ...] = ()
    added_relation_ids: tuple[str, ...] = ()


class ShadowPerceptionEngine:
    """Ask an LLM for a strict, non-persistent Stage 2A semantic proposal."""

    def __init__(
        self,
        client=None,
        model: str = "gpt-5-mini",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.client = client
        self.model = model
        self.timeout_seconds = timeout_seconds

    def perceive(
        self,
        user_message,
        previous_human_model: HumanModel,
        previous_active_state,
        dialogue_history: DialogueHistory,
    ) -> PerceptionResult:
        if self.client is None:
            raise ShadowCognitionError("No LLM client is available for shadow perception.")
        if previous_active_state is not None:
            raise ShadowCognitionError("Stage 2A only supports ACS=None.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(
                user_message.text,
                previous_human_model,
                dialogue_history,
            ),
            response_format={"type": "json_object"},
            timeout=self.timeout_seconds,
        )
        try:
            content = (response.choices[0].message.content or "").strip()
            return PerceptionResult.model_validate(json.loads(content))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ShadowCognitionError("Invalid shadow perception payload.") from error

    @staticmethod
    def _messages(
        user_text: str,
        human_model: HumanModel,
        dialogue_history: DialogueHistory,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Ты выполняешь Stage 2A shadow Perception для Discovery. "
                    "Верни только JSON. Извлекай только самостоятельный материал, "
                    "который пользователь прямо внёс в текущем сообщении. Не создавай "
                    "фактов из Human Experience, не ставь диагнозы, не предлагай "
                    "SYSTEM_PROPOSED и не трактуй короткие «да/нет/может быть» как "
                    "ответ на предыдущую систему. existing_item_id можно указать только "
                    "для явного semantic reinforcement или correction существующего "
                    "пользовательского материала. Relations предлагай только если "
                    "пользователь сам выразил связь. source_span должен указывать "
                    "границы текущего сообщения. "
                    "Формат: {materials: [{local_id, normalized_content, kind, "
                    "provenance, source_span:{char_start,char_end}, operation, "
                    "existing_item_id|null}], relation_candidates: [{source_references, "
                    "target_references, normalized_meaning, provenance, source_span}], "
                    "interaction_signals: {direct_question, refusal_or_pause, "
                    "topic_shift, ambiguity}}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_user_message": user_text,
                        "previous_human_model": human_model.model_dump(mode="json"),
                        "dialogue_history": dialogue_history.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
            },
        ]


class ShadowIntegrationEngine:
    """Apply a PerceptionResult through Data Model invariants only."""

    _ALLOWED_PROVENANCE = {
        Provenance.USER_PROVIDED,
        Provenance.USER_INTERPRETATION,
        Provenance.OTHER_PERSON,
    }

    def integrate(
        self,
        perception: PerceptionResult,
        previous_human_model: HumanModel,
        current_user_message,
        current_turn: int,
    ) -> IntegrationResult:
        if current_turn < 1:
            raise ShadowCognitionError("current_turn must be positive.")

        items = dict(previous_human_model.items)
        relations = dict(previous_human_model.relations)
        local_item_ids: dict[str, str] = {}
        added_item_ids: list[str] = []
        reinforced_item_ids: list[str] = []
        corrected_item_ids: list[str] = []

        seen_local_ids: set[str] = set()
        for material in perception.materials:
            if material.local_id in seen_local_ids:
                raise ShadowCognitionError("Perception contains duplicate local material IDs.")
            seen_local_ids.add(material.local_id)
            self._validate_material(material, items, current_user_message.text)
            if material.operation == MaterialOperation.UNRESOLVED:
                continue

            source_ref = self._source_reference(
                current_user_message.id, material.source_span
            )
            if material.operation == MaterialOperation.NEW:
                item = ModelItem(
                    kind=material.kind,
                    content=material.normalized_content,
                    provenance=material.provenance,
                    status=ItemStatus.RECORDED,
                    source_refs=(source_ref,),
                    created_turn=current_turn,
                    updated_turn=current_turn,
                )
                items[item.id] = item
                local_item_ids[material.local_id] = item.id
                added_item_ids.append(item.id)
                continue

            existing_item = items[material.existing_item_id]
            if material.operation == MaterialOperation.REINFORCE:
                items[existing_item.id] = self._reinforced_item(
                    existing_item, source_ref, current_turn
                )
                local_item_ids[material.local_id] = existing_item.id
                reinforced_item_ids.append(existing_item.id)
                continue

            corrected_old_item = self._corrected_item(existing_item, current_turn)
            items[existing_item.id] = corrected_old_item
            replacement_item = ModelItem(
                kind=material.kind,
                content=material.normalized_content,
                provenance=material.provenance,
                status=ItemStatus.RECORDED,
                source_refs=(source_ref,),
                supersedes_item_ids=(existing_item.id,),
                created_turn=current_turn,
                updated_turn=current_turn,
            )
            items[replacement_item.id] = replacement_item
            local_item_ids[material.local_id] = replacement_item.id
            corrected_item_ids.extend((existing_item.id, replacement_item.id))

        added_relation_ids: list[str] = []
        for candidate in perception.relation_candidates:
            self._validate_relation_candidate(
                candidate, items, local_item_ids, current_user_message.text
            )
            relation = Relation(
                meaning=candidate.normalized_meaning,
                source_item_ids=tuple(
                    self._resolve_reference(reference, local_item_ids)
                    for reference in candidate.source_references
                ),
                target_item_ids=tuple(
                    self._resolve_reference(reference, local_item_ids)
                    for reference in candidate.target_references
                ),
                provenance=candidate.provenance,
                status=ItemStatus.RECORDED,
                source_refs=(
                    self._source_reference(
                        current_user_message.id, candidate.source_span
                    ),
                ),
                created_turn=current_turn,
                updated_turn=current_turn,
            )
            relations[relation.id] = relation
            added_relation_ids.append(relation.id)

        return IntegrationResult(
            human_model=previous_human_model.with_updates(
                items=items,
                relations=relations,
            ),
            added_item_ids=tuple(added_item_ids),
            reinforced_item_ids=tuple(reinforced_item_ids),
            corrected_item_ids=tuple(corrected_item_ids),
            added_relation_ids=tuple(added_relation_ids),
        )

    def _validate_material(self, material, items, message_text: str) -> None:
        if material.provenance not in self._ALLOWED_PROVENANCE:
            raise ShadowCognitionError("Stage 2A cannot persist SYSTEM_PROPOSED material.")
        if (
            material.provenance == Provenance.OTHER_PERSON
            and material.kind != ModelItemKind.OTHER_PERSON_VIEW
        ):
            raise ShadowCognitionError(
                "OTHER_PERSON provenance requires OTHER_PERSON_VIEW material."
            )
        if (
            material.kind == ModelItemKind.OTHER_PERSON_VIEW
            and material.provenance != Provenance.OTHER_PERSON
        ):
            raise ShadowCognitionError(
                "OTHER_PERSON_VIEW material requires OTHER_PERSON provenance."
            )
        self._validate_span(material.source_span, message_text)
        if material.operation in (MaterialOperation.REINFORCE, MaterialOperation.CORRECT):
            if material.existing_item_id not in items:
                raise ShadowCognitionError("Perception references an unknown ModelItem.")
            if items[material.existing_item_id].provenance not in self._ALLOWED_PROVENANCE:
                raise ShadowCognitionError(
                    "Stage 2A cannot reconcile SYSTEM_PROPOSED material."
                )
        elif material.existing_item_id is not None:
            raise ShadowCognitionError("Only reinforce/correct may reference an existing ModelItem.")

    def _validate_relation_candidate(
        self, candidate, items, local_item_ids, message_text: str
    ) -> None:
        if candidate.provenance not in self._ALLOWED_PROVENANCE:
            raise ShadowCognitionError("Stage 2A cannot persist SYSTEM_PROPOSED relations.")
        self._validate_span(candidate.source_span, message_text)
        for reference in candidate.source_references + candidate.target_references:
            if reference.local_id is not None and reference.local_id not in local_item_ids:
                raise ShadowCognitionError("Relation references an unresolved local material.")
            if (
                reference.existing_item_id is not None
                and reference.existing_item_id not in items
            ):
                raise ShadowCognitionError("Relation references an unknown ModelItem.")

    @staticmethod
    def _validate_span(source_span: SourceSpan, message_text: str) -> None:
        if source_span.char_end > len(message_text):
            raise ShadowCognitionError("Perception source span exceeds the user message.")

    @staticmethod
    def _source_reference(message_id: str, source_span: SourceSpan) -> SourceReference:
        return SourceReference(
            message_id=message_id,
            char_start=source_span.char_start,
            char_end=source_span.char_end,
        )

    @staticmethod
    def _resolve_reference(
        reference: PerceptionReference, local_item_ids: dict[str, str]
    ) -> str:
        return (
            local_item_ids[reference.local_id]
            if reference.local_id is not None
            else reference.existing_item_id
        )

    @staticmethod
    def _reinforced_item(
        item: ModelItem, source_ref: SourceReference, current_turn: int
    ) -> ModelItem:
        source_refs = (
            item.source_refs
            if source_ref in item.source_refs
            else (*item.source_refs, source_ref)
        )
        return ModelItem(
            id=item.id,
            kind=item.kind,
            content=item.content,
            provenance=item.provenance,
            status=item.status,
            source_refs=source_refs,
            supersedes_item_ids=item.supersedes_item_ids,
            created_turn=item.created_turn,
            updated_turn=current_turn,
        )

    @staticmethod
    def _corrected_item(item: ModelItem, current_turn: int) -> ModelItem:
        return ModelItem(
            id=item.id,
            kind=item.kind,
            content=item.content,
            provenance=item.provenance,
            status=ItemStatus.CORRECTED,
            source_refs=item.source_refs,
            supersedes_item_ids=item.supersedes_item_ids,
            created_turn=item.created_turn,
            updated_turn=current_turn,
        )
