"""
Human Model Engine

Ответственность:
Преобразовывать факты о пользователе в модель понимания человека.

На текущем этапе модуль является архитектурной заготовкой.

Логика будет переноситься сюда постепенно.
"""

from ai.engines.human_model import HumanModel


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

        return model