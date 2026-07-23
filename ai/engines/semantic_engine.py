"""
Semantic Engine

Ответственность:
Подготавливать семантический контекст для анализа сообщения.

На текущем этапе модуль только определяет,
что уже известно о пользователе,
а что ещё неизвестно.
"""

from dataclasses import dataclass


@dataclass
class SemanticContext:
    known: list[str]
    unknown: list[str]


class SemanticEngine:

    def analyze(self, profile: dict) -> SemanticContext:

        slots = [
            "age",
            "current_weight",
            "target_weight",
            "duration",
            "main_problem",
            "previous_attempts",
            "failure_reason",
        ]

        known = []
        unknown = []

        for slot in slots:

            value = profile.get(slot)

            if value:
                known.append(f"{slot}: {value}")
            else:
                unknown.append(slot)

        return SemanticContext(
            known=known,
            unknown=unknown,
        )