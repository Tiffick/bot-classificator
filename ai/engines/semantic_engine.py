"""Semantic Engine: объективное описание текущего сообщения пользователя."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SemanticContext:
    """Результат семантического анализа одного сообщения."""

    topics: list[str] = field(default_factory=list)
    intent: str = "unknown"
    goal: Optional[str] = None
    questions: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    preferences: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    emotional_signals: list[str] = field(default_factory=list)
    confidence: float = 0.0

    # Временная совместимость с текущим HumanModelEngine.
    known: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    emotional_tone: str = "unknown"
    new_information: dict = field(default_factory=dict)


class SemanticEngine:
    """Преобразует текст текущего сообщения в SemanticContext."""

    def analyze(self, user_text: str) -> SemanticContext:
        text = user_text.strip()
        normalized_text = text.lower()

        context = SemanticContext()

        topic_keywords = {
            "weight": ("вес", "вешу", "похуд", "лишн", "живот", "толст"),
            "energy": ("устал", "энерги", "сил", "разбит"),
            "health": ("здоров", "самочувств", "болит"),
            "appearance": ("внешн", "фото", "зеркал", "одежд"),
            "sleep": ("сон", "сплю", "бессон"),
            "mobility": ("двиг", "ход", "лестниц", "подвиж"),
            "previous_attempts": ("пробовал", "пытал", "срыв", "диет"),
        }

        context.topics = [
            topic
            for topic, keywords in topic_keywords.items()
            if any(keyword in normalized_text for keyword in keywords)
        ]

        context.questions = [
            question.strip()
            for question in re.findall(r"[^?]+\?", text)
            if question.strip()
        ]

        age_match = re.search(r"\bмне\s+(\d{1,3})\s*(?:лет|года|год)\b", normalized_text)
        if age_match:
            context.facts["age"] = int(age_match.group(1))

        weight_match = re.search(
            r"\b(?:вешу|вес)\s*(\d{2,3}(?:[.,]\d+)?)\s*(?:кг|килограмм)",
            normalized_text,
        )
        if weight_match:
            context.facts["current_weight"] = float(
                weight_match.group(1).replace(",", ".")
            )

        height_match = re.search(
            r"\b(?:рост)\s*(\d{2,3})\s*(?:см|сантиметр)",
            normalized_text,
        )
        if height_match:
            context.facts["height"] = int(height_match.group(1))

        duration_match = re.search(
            r"\b(?:уже|примерно)\s+(\d+)\s*(?:лет|года|год|месяц(?:ев|а)?)\b",
            normalized_text,
        )
        if duration_match:
            context.facts["duration"] = duration_match.group(0)

        if any(phrase in normalized_text for phrase in ("хочу похудеть", "сбросить вес")):
            context.goal = "lose_weight"

        if context.questions:
            context.intent = "question"
        elif context.goal:
            context.intent = "goal_statement"
        elif text:
            context.intent = "information_statement"

        context.preferences = [
            phrase
            for phrase in ("не люблю", "предпочитаю", "мне нравится")
            if phrase in normalized_text
        ]
        context.constraints = [
            phrase
            for phrase in ("не могу", "не получается", "нет времени", "нельзя")
            if phrase in normalized_text
        ]

        emotional_keywords = {
            "frustration": ("надоело", "не получается", "срыв", "устал"),
            "anxiety": ("боюсь", "страшно", "тревож"),
            "motivation": ("хочу", "готов", "начать"),
        }
        context.emotional_signals = [
            signal
            for signal, keywords in emotional_keywords.items()
            if any(keyword in normalized_text for keyword in keywords)
        ]
        if context.emotional_signals:
            context.emotional_tone = context.emotional_signals[0]

        context.new_information = context.facts.copy()
        context.confidence = 1.0 if context.facts else 0.5 if text else 0.0

        return context
