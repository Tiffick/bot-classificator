"""Isolated one-call Cognitive Core adapter.

The adapter has no runtime, memory, Telegram or State Applier dependency.  It
only presents the existing Discovery state to an injected LLM client and
strictly parses the proposed :class:`CognitiveTurnResult`.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Any, Optional

from pydantic import ValidationError

from ai.cognitive_turn import CognitiveTurnResult
from ai.discovery_data_model import (
    ActiveConversationState,
    DialogueHistory,
    DialogueMessage,
    HumanModel,
)


class CognitiveCoreError(RuntimeError):
    """A single Cognitive Core attempt failed before producing a valid proposal."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


_PROMPT_PATH = Path(__file__).with_name("prompts") / "cognitive_core_system_prompt.txt"


def _normalize_strict_schema(source_schema: dict[str, Any]) -> dict[str, Any]:
    """Return a strict Structured Outputs presentation without mutating its source."""
    normalized = deepcopy(source_schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(normalized)
    return normalized


COGNITIVE_TURN_JSON_SCHEMA: dict[str, Any] = _normalize_strict_schema(
    CognitiveTurnResult.model_json_schema()
)


class CognitiveCore:
    """Make one structured LLM proposal for one Cognitive Discovery turn."""

    def __init__(
        self,
        client,
        model: str = "gpt-5-mini",
    ) -> None:
        self.client = client
        self.model = model
        self.last_diagnostics: dict[str, Any] = {}

    def propose(
        self,
        current_user_message: DialogueMessage,
        human_model: HumanModel,
        previous_acs: Optional[ActiveConversationState],
        dialogue_history: DialogueHistory,
    ) -> CognitiveTurnResult:
        """Return one validated proposal, or explicitly fail after one LLM attempt."""
        started_at = monotonic()
        try:
            messages = self._messages(
                current_user_message,
                human_model,
                previous_acs,
                dialogue_history,
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cognitive_turn_result",
                        "strict": True,
                        "schema": COGNITIVE_TURN_JSON_SCHEMA,
                    },
                },
            )
        except CognitiveCoreError as error:
            self._record_failure(error.category, started_at)
            raise
        except Exception as error:
            self._record_failure("api_failure", started_at)
            raise CognitiveCoreError("api_failure", "Cognitive Core API call failed.") from error

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            self._record_failure("structured_output_failure", started_at)
            raise CognitiveCoreError(
                "structured_output_failure",
                "Cognitive Core response has no structured content.",
            ) from error
        if not isinstance(content, str) or not content.strip():
            self._record_failure("structured_output_failure", started_at)
            raise CognitiveCoreError(
                "structured_output_failure",
                "Cognitive Core response has empty structured content.",
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            self._record_failure("parsing_failure", started_at)
            raise CognitiveCoreError(
                "parsing_failure", "Cognitive Core response is not valid JSON."
            ) from error
        try:
            result = CognitiveTurnResult.model_validate(payload)
        except ValidationError as error:
            self._record_failure("validation_failure", started_at)
            raise CognitiveCoreError(
                "validation_failure",
                "Cognitive Core response violates CognitiveTurnResult.",
            ) from error

        self.last_diagnostics = {
            "model": self.model,
            "elapsed_seconds": monotonic() - started_at,
            "success": True,
            "failure_category": None,
        }
        return result

    def _record_failure(self, category: str, started_at: float) -> None:
        self.last_diagnostics = {
            "model": self.model,
            "elapsed_seconds": monotonic() - started_at,
            "success": False,
            "failure_category": category,
        }

    @classmethod
    def _messages(
        cls,
        current_user_message: DialogueMessage,
        human_model: HumanModel,
        previous_acs: Optional[ActiveConversationState],
        dialogue_history: DialogueHistory,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    cls._input_view(
                        current_user_message,
                        human_model,
                        previous_acs,
                        dialogue_history,
                    ),
                    ensure_ascii=False,
                ),
            },
        ]

    @staticmethod
    def _system_prompt() -> str:
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except OSError as error:
            raise CognitiveCoreError(
                "prompt_failure", "Cognitive Core system prompt is unavailable."
            ) from error

    @classmethod
    def _input_view(
        cls,
        current_user_message: DialogueMessage,
        human_model: HumanModel,
        previous_acs: Optional[ActiveConversationState],
        dialogue_history: DialogueHistory,
    ) -> dict[str, Any]:
        return {
            "current_user_message": cls._message_view(current_user_message),
            "human_model": {
                "items": [
                    {
                        "id": item.id,
                        "kind": item.kind.value,
                        "content": item.content,
                        "provenance": item.provenance.value,
                        "status": item.status.value,
                        "supersedes_item_ids": list(item.supersedes_item_ids),
                        "source_message_ids": cls._source_message_ids(item.source_refs),
                    }
                    for item in human_model.items.values()
                ],
                "relations": [
                    {
                        "id": relation.id,
                        "meaning": relation.meaning,
                        "source_item_ids": list(relation.source_item_ids),
                        "target_item_ids": list(relation.target_item_ids),
                        "provenance": relation.provenance.value,
                        "status": relation.status.value,
                        "supersedes_relation_ids": list(
                            relation.supersedes_relation_ids
                        ),
                        "source_message_ids": cls._source_message_ids(
                            relation.source_refs
                        ),
                    }
                    for relation in human_model.relations.values()
                ],
            },
            "previous_active_conversation_state": cls._acs_view(previous_acs),
            "dialogue_history": [
                cls._message_view(message)
                for message in cls._history_closure(
                    human_model, previous_acs, dialogue_history
                )
            ],
            "human_experience": {"enabled": False},
        }

    @staticmethod
    def _message_view(message: DialogueMessage) -> dict[str, Any]:
        return {"id": message.id, "sequence": message.sequence, "text": message.text}

    @staticmethod
    def _source_message_ids(source_refs) -> list[str]:
        return list(dict.fromkeys(source_ref.message_id for source_ref in source_refs))

    @classmethod
    def _acs_view(cls, previous_acs: Optional[ActiveConversationState]):
        if previous_acs is None:
            return None
        return {
            "active_content": [
                {
                    "id": content.id,
                    "kind": content.kind.value,
                    "content": content.content,
                    "model_item_ids": list(content.model_item_ids),
                    "relation_ids": list(content.relation_ids),
                }
                for content in previous_acs.active_content.values()
            ],
            "response_targets": [
                {
                    "id": target.id,
                    "active_content_id": target.active_content_id,
                    "subject": {
                        "kind": target.subject.kind.value,
                        "id": target.subject.id,
                    },
                    "interaction": target.interaction.value,
                }
                for target in previous_acs.response_targets.values()
            ],
        }

    @classmethod
    def _history_closure(
        cls,
        human_model: HumanModel,
        previous_acs: Optional[ActiveConversationState],
        dialogue_history: DialogueHistory,
    ) -> tuple[DialogueMessage, ...]:
        by_id = {message.id: message for message in dialogue_history.messages}
        required_ids = {message.id for message in dialogue_history.messages[-8:]}
        for item in human_model.items.values():
            required_ids.update(cls._source_message_ids(item.source_refs))
        for relation in human_model.relations.values():
            required_ids.update(cls._source_message_ids(relation.source_refs))
        if previous_acs is not None:
            required_ids.add(previous_acs.source_system_message_id)
            required_ids.update(
                content.source_message_id
                for content in previous_acs.active_content.values()
            )
        missing_ids = required_ids.difference(by_id)
        if missing_ids:
            raise CognitiveCoreError(
                "history_reference_failure",
                "Cognitive Core input references messages absent from DialogueHistory.",
            )
        return tuple(sorted((by_id[message_id] for message_id in required_ids), key=lambda message: message.sequence))
