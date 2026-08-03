from ai.engines.human_model_engine import HumanModelEngine
from ai.engines.semantic_engine import SemanticEngine
from memory.user_memory import (
    get_human_model,
    get_user_memory,
    get_user_profile,
    reset_user_profile,
    set_human_model,
    update_user_profile,
)


def test_memory_separates_facts_history_and_human_model():
    user_id = 101
    reset_user_profile(user_id)
    update_user_profile(user_id, {"age": 30, "history": [{"role": "user"}]})

    memory = get_user_memory(user_id)

    assert memory["facts"]["age"] == 30
    assert memory["history"] == [{"role": "user"}]
    assert memory["human_model"] is None
    assert get_user_profile(user_id)["age"] == 30


def test_second_cycle_receives_previous_human_model():
    user_id = 102
    reset_user_profile(user_id)
    engine = HumanModelEngine()

    first_model = engine.build(
        SemanticEngine().analyze("Мне 30 лет."),
        None,
        get_user_memory(user_id),
    )
    set_human_model(user_id, first_model)

    second_model = engine.build(
        SemanticEngine().analyze("Я уже 5 лет пытаюсь похудеть."),
        get_human_model(user_id),
        get_user_memory(user_id),
    )

    assert second_model.facts["age"] == 30
    assert second_model.facts["duration"] == "уже 5 лет"


def test_dialog_engine_persists_human_model_between_cycles(fake_openai):
    user_id = 103
    reset_user_profile(user_id)

    first_profile = get_user_profile(user_id)
    asyncio.run(
        fake_openai.run_dialog_engine("Мне 30 лет.", first_profile, user_id)
    )

    second_profile = get_user_profile(user_id)
    asyncio.run(
        fake_openai.run_dialog_engine(
            "Я уже 5 лет пытаюсь похудеть.",
            second_profile,
            user_id,
        )
    )

    model = get_human_model(user_id)

    assert model.facts["age"] == 30
    assert model.facts["duration"] == "уже 5 лет"
import asyncio
