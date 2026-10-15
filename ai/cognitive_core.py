"""Isolated one-call Cognitive Core adapter.

The adapter has no runtime, memory, Telegram or State Applier dependency.  It
only presents the existing Discovery state to an injected LLM client and
adapts one strict semantic response to :class:`CognitiveTurnResult`.
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
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
    DialogueRole,
    HumanModel,
)
from ai.human_experience import (
    HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION,
    HumanExperienceSelection,
    human_experience_enabled,
    select_human_experience,
)
from ai.knowledge_retrieval import (
    PROFESSIONAL_KNOWLEDGE_SYSTEM_INSTRUCTION,
    KnowledgeRetrievalError,
    KnowledgeRetrievalSelection,
    knowledge_retrieval_enabled,
    retrieve_professional_context,
)


class CognitiveCoreError(RuntimeError):
    """A single Cognitive Core attempt failed before producing a valid proposal."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


_PROMPT_PATH = Path(__file__).with_name("prompts") / "cognitive_core_system_prompt.txt"
_LOGGER = logging.getLogger(__name__)


def _previous_target_handle_map(
    previous_acs: Optional[ActiveConversationState],
) -> dict[str, str]:
    """Map input-scoped target handles to persistent IDs deterministically."""
    if previous_acs is None:
        return {}
    return {
        f"previous_target_{index}": target_id
        for index, target_id in enumerate(previous_acs.response_targets, start=1)
    }


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
            quote = entry.get("source_quote")
            if quote is None and not required:
                entry.pop("source_quote", None)
                entry["source_span"] = None
                continue
            if not isinstance(quote, str) or not quote:
                fail(quote_path, "expected a non-empty string")
            exact_matches = []
            position = message_text.find(quote)
            while position != -1:
                exact_matches.append(position)
                position = message_text.find(quote, position + 1)

            if len(exact_matches) == 1:
                start = exact_matches[0]
                end = start + len(quote)
            elif exact_matches:
                fail(
                    quote_path,
                    f"quote={quote!r}; exact matches={len(exact_matches)}; "
                    "expected exactly one",
                )
            else:
                case_insensitive_matches = list(
                    re.finditer(re.escape(quote), message_text, flags=re.IGNORECASE)
                )
                if len(case_insensitive_matches) != 1:
                    fail(
                        quote_path,
                        f"quote={quote!r}; exact matches=0; "
                        "case-insensitive matches="
                        f"{len(case_insensitive_matches)}; expected exactly one",
                    )
                start, end = case_insensitive_matches[0].span()
                # Canonicalize only the copied API payload. Downstream evidence
                # therefore points at the exact original USER substring.
                entry["source_quote"] = message_text[start:end]

            entry.pop("source_quote")
            entry["source_span"] = {
                "char_start": start,
                "char_end": end,
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
        # Imported here because the semantic adapter reuses the quote resolver
        # and error type defined in this module. This is the only production wire.
        from ai.cognitive_semantic_wire import (
            SemanticTurn,
            SemanticWireError,
            adapt_semantic_turn,
            semantic_turn_json_schema,
        )

        started_at = monotonic()
        schema = semantic_turn_json_schema(previous_acs)
        assembly_diagnostics: dict[str, Any] = {}
        self.last_diagnostics = {
            "requested_model": self.model,
            "success": False,
            "stage": "request_assembly",
            "llm_call_count": 0,
        }
        try:
            assembly_started_at = monotonic()
            messages = self._messages(
                current_user_message,
                human_model,
                previous_acs,
                dialogue_history,
                diagnostics=assembly_diagnostics,
            )
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_turn",
                    "strict": True,
                    "schema": schema,
                },
            }
            api_request = {
                "model": self.model,
                "messages": messages,
                "response_format": response_format,
            }
            self.last_diagnostics.update(
                {
                    "stage": "api_request",
                    "context_assembly": assembly_diagnostics,
                    "full_api_request": api_request,
                    "client_configuration": {
                        "max_retries": getattr(self.client, "max_retries", None),
                    },
                    "semantic_schema_sha256": hashlib.sha256(
                        json.dumps(
                            schema,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "stage_latency_ms": {
                        "context_assembly": round(
                            (monotonic() - assembly_started_at) * 1000, 3
                        )
                    },
                }
            )
            self.last_diagnostics["llm_call_count"] = 1
            api_started_at = monotonic()
            response = self.client.chat.completions.create(**api_request)
            api_latency_ms = round((monotonic() - api_started_at) * 1000, 3)
            choice = response.choices[0] if getattr(response, "choices", None) else None
            self.last_diagnostics.update(
                {
                    "stage": "structured_output",
                    "actual_model": getattr(response, "model", None),
                    "usage": self._usage_view(getattr(response, "usage", None)),
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "raw_api_response": self._raw_response_view(response),
                }
            )
            self.last_diagnostics["stage_latency_ms"]["api"] = api_latency_ms
        except CognitiveCoreError as error:
            self._record_failure(error.category, started_at)
            raise
        except Exception as error:
            self.last_diagnostics["api_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
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
        self.last_diagnostics["raw_structured_output"] = content

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            self._record_failure("parsing_failure", started_at)
            raise CognitiveCoreError(
                "parsing_failure", "Cognitive Core response is not valid JSON."
            ) from error
        self.last_diagnostics.update({"stage": "semantic_validation", "raw_semantic_turn": payload})
        semantic_validation_started_at = monotonic()
        try:
            semantic_turn = SemanticTurn.model_validate(payload)
        except ValidationError as error:
            self.last_diagnostics["semantic_turn_validation_error"] = str(error)
            self._record_failure("validation_failure", started_at)
            raise CognitiveCoreError(
                "validation_failure",
                "Cognitive Core response violates SemanticTurn.",
            ) from error
        self.last_diagnostics["parsed_semantic_turn"] = semantic_turn.model_dump(mode="json")
        self.last_diagnostics["stage_latency_ms"]["semantic_turn_validation"] = round(
            (monotonic() - semantic_validation_started_at) * 1000, 3
        )
        adapter_started_at = monotonic()
        try:
            result = adapt_semantic_turn(
                semantic_turn,
                current_user_text=current_user_message.text,
                human_model=human_model,
                previous_acs=previous_acs,
            )
        except CognitiveCoreError as error:
            self._record_failure(error.category, started_at)
            raise
        except ValidationError as error:
            self.last_diagnostics["cognitive_turn_validation_error"] = str(error)
            self._record_failure("validation_failure", started_at)
            raise CognitiveCoreError(
                "validation_failure",
                "Cognitive Core response violates CognitiveTurnResult.",
            ) from error
        except SemanticWireError as error:
            self.last_diagnostics["adapter_error"] = str(error)
            self._record_failure("validation_failure", started_at)
            raise CognitiveCoreError(
                "validation_failure",
                "Cognitive Core response violates semantic wire contract.",
            ) from error
        self.last_diagnostics["stage_latency_ms"]["adapter"] = round(
            (monotonic() - adapter_started_at) * 1000, 3
        )

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

        self.last_diagnostics.update(
            {
                "stage": "completed",
                "model": self.model,
                "elapsed_seconds": monotonic() - started_at,
                "success": True,
                "failure_category": None,
                "cognitive_turn_result": result.model_dump(mode="json"),
            }
        )
        return result

    def _record_failure(self, category: str, started_at: float) -> None:
        self.last_diagnostics.update(
            {
                "model": self.model,
                "elapsed_seconds": monotonic() - started_at,
                "success": False,
                "failure_category": category,
            }
        )

    @staticmethod
    def _usage_view(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "cached_input_tokens": getattr(prompt_details, "cached_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    @staticmethod
    def _raw_response_view(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump(mode="json")
            except Exception:
                pass
        choices = []
        for choice in getattr(response, "choices", ()) or ():
            message = getattr(choice, "message", None)
            choices.append(
                {
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "message": {
                        "role": getattr(message, "role", None),
                        "content": getattr(message, "content", None),
                        "refusal": getattr(message, "refusal", None),
                    },
                }
            )
        return {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "choices": choices,
            "usage": CognitiveCore._usage_view(getattr(response, "usage", None)),
        }

    @classmethod
    def _messages(
        cls,
        current_user_message: DialogueMessage,
        human_model: HumanModel,
        previous_acs: Optional[ActiveConversationState],
        dialogue_history: DialogueHistory,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        human_experience = None
        knowledge_selection = None
        base_system_prompt = cls._system_prompt()
        system_prompt = base_system_prompt
        human_experience_is_enabled = human_experience_enabled()
        if knowledge_retrieval_enabled():
            try:
                knowledge_selection = retrieve_professional_context(
                    current_user_message.text,
                    human_model,
                    previous_acs,
                    dialogue_history,
                    include_human_experience=human_experience_is_enabled,
                )
            except KnowledgeRetrievalError as error:
                _LOGGER.warning(
                    "Professional knowledge retrieval failed; Cognitive Core "
                    "is using the previous production context path.",
                    exc_info=True,
                )
                if diagnostics is not None:
                    diagnostics["knowledge_retrieval"] = {
                        "enabled": True,
                        "status": "fallback",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
            else:
                system_prompt = (
                    f"{system_prompt.rstrip()}\n"
                    f"{PROFESSIONAL_KNOWLEDGE_SYSTEM_INSTRUCTION}\n"
                )
                if human_experience_is_enabled:
                    system_prompt = (
                        f"{system_prompt.rstrip()}\n"
                        f"{HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION}\n"
                    )
        if knowledge_selection is None and human_experience_is_enabled:
            human_experience = select_human_experience(
                current_user_message.text,
                tuple(
                    message.text
                    for message in dialogue_history.messages
                    if message.role == DialogueRole.USER
                ),
            )
            system_prompt = (
                f"{system_prompt.rstrip()}\n{HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION}\n"
            )
        if diagnostics is not None:
            diagnostics.update(
                {
                    "knowledge_retrieval_enabled": knowledge_retrieval_enabled(),
                    "human_experience_enabled": human_experience_is_enabled,
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "base_prompt_sha256": hashlib.sha256(
                        base_system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "base_system_prompt": base_system_prompt,
                    "assembled_system_message": system_prompt,
                    "system_instruction_components": {
                        "professional_knowledge": (
                            PROFESSIONAL_KNOWLEDGE_SYSTEM_INSTRUCTION
                            if knowledge_selection is not None
                            else None
                        ),
                        "human_experience": (
                            HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION
                            if human_experience_is_enabled
                            else None
                        ),
                    },
                    "knowledge_retrieval": cls._retrieval_diagnostics(
                        knowledge_selection,
                        fallback_human_experience=human_experience,
                        existing=diagnostics.get("knowledge_retrieval"),
                    ),
                    "context_summary": {
                        "human_model_items": len(human_model.items),
                        "human_model_relations": len(human_model.relations),
                        "previous_active_content": (
                            len(previous_acs.active_content) if previous_acs else 0
                        ),
                        "previous_response_targets": (
                            len(previous_acs.response_targets) if previous_acs else 0
                        ),
                        "dialogue_messages": len(dialogue_history.messages),
                    },
                }
            )
        input_view = cls._input_view(
            current_user_message,
            human_model,
            previous_acs,
            dialogue_history,
            human_experience,
            knowledge_selection,
        )
        if diagnostics is not None:
            diagnostics["input_context"] = input_view
        user_context_message = json.dumps(input_view, ensure_ascii=False)
        if diagnostics is not None:
            diagnostics["context_sizes_characters"] = {
                "base_system_prompt": len(base_system_prompt),
                "assembled_system_message": len(system_prompt),
                "user_json_context": len(user_context_message),
                "total_message_content": len(system_prompt) + len(user_context_message),
            }
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_context_message,
            },
        ]

    @staticmethod
    def _retrieval_diagnostics(
        selection: KnowledgeRetrievalSelection | None,
        *,
        fallback_human_experience: HumanExperienceSelection | None,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if selection is None:
            result = dict(existing or {"enabled": False, "status": "disabled"})
            if fallback_human_experience is not None:
                result["legacy_human_experience_card_ids"] = [
                    card.card_id for card in fallback_human_experience.cards
                ]
            return result
        professional_scores = dict(getattr(selection, "professional_scores", ()))
        human_scores = dict(getattr(selection, "human_experience_scores", ()))
        selected_professional_ids = {
            card.id for card in getattr(selection, "professional_cards", ())
        }
        selected_human_ids = {
            card.id for card in getattr(selection, "human_experience_cards", ())
        }
        return {
            "enabled": True,
            "status": (
                "selected"
                if getattr(selection, "professional_cards", ())
                else "abstained"
            ),
            "embedding_model": getattr(selection, "embedding_model_id", None),
            "index_corpus_sha256": getattr(selection, "index_corpus_sha256", None),
            "index_schema_version": getattr(selection, "index_schema_version", None),
            "semantic_view_version": getattr(selection, "semantic_view_version", None),
            "query_view_kinds": list(getattr(selection, "query_view_kinds", ())),
            "query_views": [
                {"kind": kind, "text": text, "weight": weight}
                for kind, text, weight in getattr(selection, "query_views", ())
            ],
            "latency_ms": getattr(selection, "retrieval_latency_ms", None),
            "similarity_floors": {
                "professional": getattr(
                    selection, "professional_similarity_floor", None
                ),
                "human_experience": getattr(
                    selection, "human_experience_similarity_floor", None
                ),
            },
            "professional_cards": [
                {
                    "card_id": card.id,
                    "title": card.title,
                    "similarity": professional_scores.get(card.id),
                }
                for card in getattr(selection, "professional_cards", ())
            ],
            "human_experience_cards": [
                {
                    "card_id": card.id,
                    "title": card.title,
                    "similarity": human_scores.get(card.id),
                }
                for card in getattr(selection, "human_experience_cards", ())
            ],
            "rejected_professional_candidates": [
                {"card_id": card_id, "similarity": score}
                for card_id, score in getattr(
                    selection, "professional_candidate_scores", ()
                )
                if card_id not in selected_professional_ids
            ],
            "rejected_human_experience_candidates": [
                {"card_id": card_id, "similarity": score}
                for card_id, score in getattr(
                    selection, "human_experience_candidate_scores", ()
                )
                if card_id not in selected_human_ids
            ],
            "safety_cards": [
                {"card_id": card.id, "title": card.title}
                for card in getattr(selection, "safety_cards", ())
            ],
        }

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
        human_experience: HumanExperienceSelection | None = None,
        knowledge_selection: KnowledgeRetrievalSelection | None = None,
    ) -> dict[str, Any]:
        input_view = {
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
            "human_experience": (
                knowledge_selection.human_experience_input_view(
                    enabled=human_experience_enabled()
                )
                if knowledge_selection is not None
                else (
                    human_experience.input_view()
                    if human_experience is not None
                    else {"enabled": False}
                )
            ),
        }
        if knowledge_selection is not None:
            input_view["professional_knowledge"] = (
                knowledge_selection.professional_input_view()
            )
        return input_view

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
        target_handles = _previous_target_handle_map(previous_acs)
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
                    "handle": handle,
                    "active_content_id": target.active_content_id,
                    "subject": {
                        "kind": target.subject.kind.value,
                        "id": target.subject.id,
                    },
                    "interaction": target.interaction.value,
                }
                for handle, target_id in target_handles.items()
                for target in (previous_acs.response_targets[target_id],)
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
