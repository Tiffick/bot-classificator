"""
Human Model Engine

Ответственность:
Преобразовывать факты о пользователе в модель понимания человека.

На текущем этапе модуль является архитектурной заготовкой.

Логика будет переноситься сюда постепенно.
"""

from ai.engines.human_model import HumanModel
from ai.engines.semantic_engine import SemanticEngine


class HumanModelEngine:
    """
    Архитектурная заготовка.

    Пока ничего не делает.
    """

    def build(self, profile: dict) -> HumanModel:
        """
        Строит первичную модель человека
        на основе накопленного профиля.
        """

        model = HumanModel()

        model.facts = profile.copy()

        semantic = SemanticEngine().analyze(profile)
        model.known = semantic.known

        model.unknown = semantic.unknown

        return model

    def apply_update(self, profile: dict, update: dict) -> dict:
        """
        Применяет безопасное обновление профиля.
        """

        safe_update = {}

        immutable_fields = {
            "name",
            "gender"
        }

        for key, value in update.items():

            if value is None:
                continue

            if key in immutable_fields and profile.get(key) is not None:
                continue

            safe_update[key] = value

        return safe_update

    def is_discovery_complete(self, profile: dict) -> bool:
        """
        Проверяет, завершён ли этап знакомства.
        """

        required = [
            "main_problem",
            "duration"
        ]

        optional = [
            "age",
            "current_weight",
            "target_weight",
            "previous_attempts",
            "failure_reason"
        ]

        for field in required:
            if not profile.get(field):
                return False

        filled = sum(
            1 for field in optional
            if profile.get(field)
        )

        return filled >= 2

        