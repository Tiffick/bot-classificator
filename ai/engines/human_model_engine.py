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