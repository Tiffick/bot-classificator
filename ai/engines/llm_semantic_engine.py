"""LLM semantic extraction for Conversation Lab shadow comparisons only."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ai.engines.semantic_engine import SemanticContext


TOPICS = (
    "weight",
    "energy",
    "health",
    "sleep",
    "mobility",
    "breathing",
    "appearance",
    "clothes",
)
INTENTS = ("unknown", "information_statement", "question", "goal_statement", "pause_or_end")
GOALS = (
    "lose_weight",
    "restore_energy",
    "improve_wellbeing",
    "move_easier",
    "improve_appearance",
    "restore_confidence",
    "restore_control",
)
PREFERENCES = (
    "avoid_strict_diet",
    "avoid_calorie_counting",
    "avoid_gym",
    "prefer_simple_path",
    "want_fast_result",
    "ready_gradual",
)
CONSTRAINTS = (
    "lack_time",
    "fatigue_after_work",
    "shift_schedule",
    "family_or_children",
    "financial_limit",
    "stress",
    "lack_support",
    "hunger",
    "habits",
    "distrust",
    "routine_difficulty",
    "avoid_counting_or_restriction",
)
EMOTIONAL_SIGNALS = (
    "self_dissatisfaction",
    "shame",
    "low_confidence",
    "loss_of_control",
    "frustration",
    "situation_fatigue",
    "despair",
    "anxiety",
    "closedness",
    "motivation",
)
EVENTS = (
    "daily_effect:difficulty_cooking",
    "daily_effect:reduced_energy",
    "daily_effect:limited_movement",
    "daily_effect:shortness_of_breath",
    "daily_effect:clothing_difficulty",
    "daily_effect:mirror_or_photo_avoidance",
    "daily_effect:social_discomfort",
    "attempt:self_directed",
    "attempt:professional",
    "attempt:diet",
    "attempt:sport",
    "attempt:calorie_counting",
    "attempt:medication_or_supplements",
    "attempt:nutrition_program",
    "attempt_result:helped",
    "attempt_result:not_helped",
    "attempt_result:temporary_help",
    "attempt_result:stopped",
    "attempt_result:relapse",
    "attempt_result:could_not_follow",
    "dynamics:gradual",
    "dynamics:sudden",
    "dynamics:worsening",
    "dynamics:stable",
    "dynamics:returning",
    "dynamics:linked_to_event",
)
FACT_FIELDS = {
    "age": "integer",
    "current_weight": "number",
    "height": "integer",
    "duration": "string",
}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


SEMANTIC_JSON_SCHEMA: dict[str, Any] = {
    "name": "semantic_context_shadow",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "topics",
            "intent",
            "goal",
            "facts",
            "preferences",
            "constraints",
            "emotional_signals",
            "events",
            "questions",
            "confidence",
        ],
        "properties": {
            "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
            "intent": {"type": "string", "enum": list(INTENTS)},
            "goal": _nullable({"type": "string", "enum": list(GOALS)}),
            "facts": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FACT_FIELDS),
                "properties": {
                    "age": _nullable({"type": "integer"}),
                    "current_weight": _nullable({"type": "number"}),
                    "height": _nullable({"type": "integer"}),
                    "duration": _nullable({"type": "string"}),
                },
            },
            "preferences": {"type": "array", "items": {"type": "string", "enum": list(PREFERENCES)}},
            "constraints": {"type": "array", "items": {"type": "string", "enum": list(CONSTRAINTS)}},
            "emotional_signals": {"type": "array", "items": {"type": "string", "enum": list(EMOTIONAL_SIGNALS)}},
            "events": {"type": "array", "items": {"type": "string", "enum": list(EVENTS)}},
            "questions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
}


class LLMSemanticEngine:
    """Extracts only explicit message meaning into the existing SemanticContext.

    This class is intentionally not imported by DialogEngine.  Conversation Lab
    can invoke it beside the deterministic engine and record the comparison.
    """

    def __init__(
        self, client=None, model: str = "gpt-5-mini", timeout_seconds: float = 15.0
    ) -> None:
        load_dotenv()
        self.client = client or OpenAI()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.last_diagnostics: dict[str, Any] = {}

    def analyze(self, user_text: str) -> SemanticContext:
        if not user_text or not user_text.strip():
            self.last_diagnostics = {
                "success": True,
                "fallback_reason": None,
                "rejected_fields": [],
                "unconfirmed_interpretations": [],
            }
            return SemanticContext()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(user_text),
                response_format={"type": "json_schema", "json_schema": SEMANTIC_JSON_SCHEMA},
                timeout=self.timeout_seconds,
            )
            payload = json.loads((response.choices[0].message.content or "").strip())
            context = self._context_from_payload(payload, user_text)
            rejected_fields, interpretations = self._remove_unconfirmed_values(
                context, user_text
            )
            self.last_diagnostics = {
                "success": True,
                "fallback_reason": None,
                "rejected_fields": rejected_fields,
                "unconfirmed_interpretations": interpretations,
            }
            return context
        except Exception as error:
            self.last_diagnostics = {
                "success": False,
                "fallback_reason": f"{type(error).__name__}: {error}",
                "rejected_fields": [],
                "unconfirmed_interpretations": [],
            }
            return SemanticContext()

    @staticmethod
    def _remove_unconfirmed_values(
        context: SemanticContext, user_text: str
    ) -> tuple[list[str], list[str]]:
        """Keep inferred values diagnostic-only; never forward them as facts."""
        normalized_text = user_text.lower()
        rejected_fields: list[str] = []
        interpretations: list[str] = []
        if "fatigue_after_work" in context.constraints and "работ" not in normalized_text:
            context.constraints.remove("fatigue_after_work")
            rejected_fields.append("constraints.fatigue_after_work")
            interpretations.append(
                "fatigue_after_work inferred work although the message did not name work"
            )
        if "closedness" in context.emotional_signals and "закры" not in normalized_text:
            context.emotional_signals.remove("closedness")
            rejected_fields.append("emotional_signals.closedness")
            interpretations.append(
                "closedness interpreted a communication boundary as an emotional trait"
            )
        context.emotional_tone = (
            context.emotional_signals[0] if context.emotional_signals else "unknown"
        )
        return rejected_fields, interpretations

    @staticmethod
    def _messages(user_text: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Ты выполняешь только извлечение явных смыслов одного сообщения "
                    "пользователя. Не рассуждай, не выбирай вопрос, не давай совет, "
                    "не ставь диагнозы и не выводи причины. Заполняй только значения, "
                    "которые прямо выражены в тексте. Отрицание имеет приоритет: "
                    "«не готова» не является motivation. Используй только коды из схемы. "
                    "Диагнозы, включая depression, anxiety_disorder и eating_disorder, "
                    "не извлекай: таких полей в схеме нет."
                ),
            },
            {"role": "user", "content": user_text},
        ]

    @staticmethod
    def _is_string_list(value: Any, allowed: tuple[str, ...]) -> bool:
        return isinstance(value, list) and all(
            isinstance(item, str) and item in allowed for item in value
        )

    @classmethod
    def _context_from_payload(cls, payload: Any, user_text: str) -> SemanticContext:
        required = {
            "topics",
            "intent",
            "goal",
            "facts",
            "preferences",
            "constraints",
            "emotional_signals",
            "events",
            "questions",
            "confidence",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("LLM semantic payload has an invalid top-level schema.")
        if not cls._is_string_list(payload["topics"], TOPICS):
            raise ValueError("Invalid topics.")
        if payload["intent"] not in INTENTS:
            raise ValueError("Invalid intent.")
        if payload["goal"] is not None and payload["goal"] not in GOALS:
            raise ValueError("Invalid goal.")
        if not cls._is_string_list(payload["preferences"], PREFERENCES):
            raise ValueError("Invalid preferences.")
        if not cls._is_string_list(payload["constraints"], CONSTRAINTS):
            raise ValueError("Invalid constraints.")
        if not cls._is_string_list(payload["emotional_signals"], EMOTIONAL_SIGNALS):
            raise ValueError("Invalid emotional signals.")
        if not cls._is_string_list(payload["events"], EVENTS):
            raise ValueError("Invalid events.")
        if not isinstance(payload["questions"], list) or not all(
            isinstance(item, str) for item in payload["questions"]
        ):
            raise ValueError("Invalid questions.")
        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence.")

        raw_facts = payload["facts"]
        if not isinstance(raw_facts, dict) or set(raw_facts) != set(FACT_FIELDS):
            raise ValueError("Invalid facts schema.")
        if raw_facts["age"] is not None and (isinstance(raw_facts["age"], bool) or not isinstance(raw_facts["age"], int)):
            raise ValueError("Invalid age.")
        if raw_facts["current_weight"] is not None and (isinstance(raw_facts["current_weight"], bool) or not isinstance(raw_facts["current_weight"], (int, float))):
            raise ValueError("Invalid current weight.")
        if raw_facts["height"] is not None and (isinstance(raw_facts["height"], bool) or not isinstance(raw_facts["height"], int)):
            raise ValueError("Invalid height.")
        if raw_facts["duration"] is not None and not isinstance(raw_facts["duration"], str):
            raise ValueError("Invalid duration.")
        if not cls._facts_are_grounded(raw_facts, user_text.lower()):
            raise ValueError("LLM returned a fact that is not explicit in the message.")

        facts = {key: value for key, value in raw_facts.items() if value is not None}
        context = SemanticContext(
            topics=payload["topics"],
            intent=payload["intent"],
            goal=payload["goal"],
            questions=payload["questions"],
            events=payload["events"],
            facts=facts,
            preferences=payload["preferences"],
            constraints=payload["constraints"],
            emotional_signals=payload["emotional_signals"],
            confidence=float(confidence),
        )
        context.emotional_tone = context.emotional_signals[0] if context.emotional_signals else "unknown"
        context.new_information = context.facts.copy()
        return context

    @staticmethod
    def _facts_are_grounded(raw_facts: dict[str, Any], normalized_text: str) -> bool:
        age = raw_facts["age"]
        if age is not None and not re.search(
            rf"\bмне\s+{age}\s*(?:лет|года|год)\b", normalized_text
        ):
            return False
        weight = raw_facts["current_weight"]
        weight_pattern = re.escape(str(weight)).replace(r"\.", "[.,]")
        if weight is not None and not re.search(
            rf"\b(?:вешу|вес)\s*{weight_pattern}\s*(?:кг|килограмм)",
            normalized_text,
        ):
            return False
        height = raw_facts["height"]
        if height is not None and not re.search(
            rf"\bрост\s*{height}\s*(?:см|сантиметр)", normalized_text
        ):
            return False
        duration = raw_facts["duration"]
        return duration is None or duration.lower() in normalized_text


def compare_semantic_contexts(
    deterministic: SemanticContext, llm: SemanticContext
) -> dict[str, Any]:
    """Return a comparison artifact; it never chooses a winner or mutates inputs."""
    current = asdict(deterministic) if is_dataclass(deterministic) else deterministic
    candidate = asdict(llm) if is_dataclass(llm) else llm
    list_fields = (
        "topics",
        "questions",
        "events",
        "preferences",
        "constraints",
        "emotional_signals",
    )
    return {
        "only_llm": {
            **{
                field: [value for value in candidate[field] if value not in current[field]]
                for field in list_fields
            },
            "facts": {
                key: value
                for key, value in candidate["facts"].items()
                if current["facts"].get(key) != value
            },
            "goal": candidate["goal"] if candidate["goal"] != current["goal"] else None,
            "intent": candidate["intent"] if candidate["intent"] != current["intent"] else None,
        },
        "only_deterministic": {
            **{
                field: [value for value in current[field] if value not in candidate[field]]
                for field in list_fields
            },
            "facts": {
                key: value
                for key, value in current["facts"].items()
                if candidate["facts"].get(key) != value
            },
            "goal": current["goal"] if current["goal"] != candidate["goal"] else None,
            "intent": current["intent"] if current["intent"] != candidate["intent"] else None,
        },
    }
