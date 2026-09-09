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


def _api_facing_schema() -> dict[str, Any]:
    """Derive the wire representation; keep the internal contract unchanged."""
    schema = CognitiveTurnResult.model_json_schema()
    for name in ("ItemOperation", "RelationOperation", "TargetResolution"):
        properties = schema["$defs"][name]["properties"]
        del properties["source_span"]
        quote_schema = {"type": "string", "minLength": 1}
        properties["source_quote"] = (
            {"anyOf": [quote_schema, {"type": "null"}]}
            if name == "TargetResolution" else quote_schema
        )
    del schema["$defs"]["SourceSpan"]

    content_reference_pattern = r"^content:[A-Za-z0-9][A-Za-z0-9._-]*$"
    action_target_reference = schema["$defs"]["ActionTarget"]["properties"][
        "active_content_local_id"
    ]
    action_target_reference.update(
        pattern=content_reference_pattern,
        description=(
            "Local ID of the ActionContent in the current SystemAction that carries "
            "this target; persistent aci_ IDs are invalid."
        ),
    )
    subject_reference = schema["$defs"]["ActionSubjectReference"]["properties"][
        "active_content_local_id"
    ]
    subject_reference["description"] = (
        "When kind is active_content, the semantic subject must be an ActionContent "
        "in the current SystemAction; persistent aci_ IDs are invalid."
    )
    for branch in subject_reference["anyOf"]:
        if branch.get("type") == "string":
            branch["pattern"] = content_reference_pattern

    return _normalize_strict_schema(schema)


COGNITIVE_TURN_JSON_SCHEMA: dict[str, Any] = _api_facing_schema()


def _resolve_source_quotes(payload: Any, message_text: str) -> dict[str, Any]:
    """Resolve exact, unique evidence on a copy before internal validation."""
    def fail(path: str, reason: str) -> None:
        raise CognitiveCoreError("validation_failure", f"{path}: {reason}")

    if not isinstance(payload, dict):
        fail("$", "expected an object")
    converted = deepcopy(payload)
    patch = converted.get("state_patch", {})
    if not isinstance(patch, dict):
        fail("state_patch", "expected an object")
    for path, entries, required in (
        ("state_patch.item_operations", patch.get("item_operations", []), True),
        ("state_patch.relation_operations", patch.get("relation_operations", []), True),
        ("reconciliation", converted.get("reconciliation", []), False),
    ):
        if not isinstance(entries, list):
            fail(path, "expected an array")
        for index, entry in enumerate(entries):
            entry_path = f"{path}[{index}]"
            if not isinstance(entry, dict):
                fail(entry_path, "expected an object")
            if "source_span" in entry:
                fail(f"{entry_path}.source_span", "numeric evidence is not accepted from the API")
            quote_path = f"{entry_path}.source_quote"
            if "source_quote" not in entry and required:
                fail(quote_path, "required quote is missing")
            quote = entry.pop("source_quote", None)
            if quote is None and not required:
                entry["source_span"] = None
                continue
            if not isinstance(quote, str) or not quote:
                fail(quote_path, "expected a non-empty string")
            matches = []
            position = message_text.find(quote)
            while position != -1:
                matches.append(position)
                position = message_text.find(quote, position + 1)
            if len(matches) != 1:
                fail(quote_path, f"quote={quote!r}; exact matches={len(matches)}; expected exactly one")
            entry["source_span"] = {
                "char_start": matches[0],
                "char_end": matches[0] + len(quote),
            }
    return converted


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
            converted_payload = _resolve_source_quotes(payload, current_user_message.text)
            result = CognitiveTurnResult.model_validate(converted_payload)
        except CognitiveCoreError as error:
            self._record_failure(error.category, started_at)
            raise
        except ValidationError as error:
            self._record_failure("validation_failure", started_at)
            raise CognitiveCoreError(
                "validation_failure",
                "Cognitive Core response violates CognitiveTurnResult.",
            ) from error

        message_length = len(current_user_message.text)
        for path, entries in (
            ("state_patch.item_operations", result.state_patch.item_operations),
            ("state_patch.relation_operations", result.state_patch.relation_operations),
            ("reconciliation", result.reconciliation),
        ):
            for index, entry in enumerate(entries):
                span = entry.source_span
                if span is not None and not (
                    0 <= span.char_start < span.char_end <= message_length
                ):
                    self._record_failure("validation_failure", started_at)
                    raise CognitiveCoreError(
                        "validation_failure",
                        f"{path}[{index}].source_span [{span.char_start}, "
                        f"{span.char_end}) is outside the current USER message "
                        f"(length={message_length}).",
                    )

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
