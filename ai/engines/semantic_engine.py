"""
Semantic Engine

Ответственность:
Подготавливать семантический контекст для анализа сообщения.

На текущем этапе модуль только определяет,
что уже известно о пользователе,
а что ещё неизвестно.
"""

from dataclasses import dataclass, field


@dataclass
class SemanticContext:
    known: list[str]
    unknown: list[str]
    facts: dict = field(default_factory=dict)
    emotional_tone: str = "unknown"
    intent: str = "unknown"
    new_information: dict = field(default_factory=dict)


class SemanticEngine:

    def analyze(self, user_text: str) -> SemanticContext:

        return SemanticContext(
            known=[],
            unknown=[],
        )
