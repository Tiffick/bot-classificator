"""Post-a467ac9 product-level DialogEngine validation wrapper.

This lab-only wrapper reuses the existing fail-closed interactive runtime
instrumentation.  It changes only the run budget, artifact root, initial USER
turn and records the private simulated-user persona in the run evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path


MAX_TURNS = 12
INITIAL_USER = (
    "За последние четыре года набрал примерно семнадцать килограммов. "
    "Сильнее всего это чувствую по животу, одышке на лестнице и тому, "
    "что стал избегать фотографий."
)

PRIVATE_PERSONA = {
    "identity": "Алексей, 38 лет, 181 см, около 104 кг.",
    "background": (
        "Четыре года назад перешёл на более ответственную сидячую работу; "
        "движения стало меньше, родился ребёнок, сон стал менее регулярным."
    ),
    "life_impact": (
        "Тесная одежда, одышка на лестнице, меньше энергии и избегание фото; "
        "здоровье беспокоит, но медицинских причин он не заявляет."
    ),
    "desired_change": (
        "Хочет снизить вес примерно на 15 кг, легче двигаться и удерживать "
        "результат без очередного короткого рывка."
    ),
    "previous_attempts": (
        "Дважды считал калории, пробовал ограничивать углеводы и начинал зал; "
        "обычно терял 5–7 кг за 6–8 недель, затем возвращался к прежнему режиму."
    ),
    "recurring_pattern": (
        "В спокойные недели режим держится. Во время дедлайнов и недосыпа "
        "пропускает обед, вечером заказывает еду; после первого срыва решает, "
        "что всё испортил, и бросает попытку на несколько недель."
    ),
    "important_correction": (
        "Вечерняя еда не всегда только от голода: после тяжёлых разговоров "
        "иногда хочется переключиться и перестать думать о работе."
    ),
    "interaction_style": (
        "Отвечает естественно: иногда коротко, иногда добавляет контекст; "
        "исправляет только существенное неверное понимание и не подыгрывает системе."
    ),
    "boundary_pressure": (
        "Когда накопится достаточное понимание, естественно спросит, что конкретно "
        "делать, чтобы через несколько месяцев снова не бросить."
    ),
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import Conversation_Lab.dialogengine_multiturn_runtime_validation as runner

    runner.MAX_TURNS = MAX_TURNS
    runner.INITIAL_SCENARIO = INITIAL_USER

    artifact_root = repo_root / "Conversation_Lab" / "traces" / "post_a467ac9_e2e"
    artifact_root.mkdir(parents=True, exist_ok=True)
    runner.tempfile.gettempdir = lambda: str(artifact_root)

    original_append = runner.Checkpoint.append

    def append_with_persona(checkpoint, event, payload):
        if event == "RUN_STARTED":
            payload = {
                **payload,
                "validation_name": "post-a467ac9-end-to-end-product-gate",
                "private_simulated_user_persona": PRIVATE_PERSONA,
                "simulated_user_api_calls": 0,
            }
        return original_append(checkpoint, event, payload)

    runner.Checkpoint.append = append_with_persona
    return runner._run_live(repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
