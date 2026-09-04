import asyncio

import pytest

from ai.discovery_data_model import validate_discovery_memory
from memory.user_memory import (
    get_human_model,
    get_shadow_discovery_memory,
    get_user_memory,
    get_user_profile,
    reset_user_profile,
    user_profiles,
)


def test_first_turn_creates_valid_shadow_discovery_memory(fake_openai):
    user_id = 401
    user_text = "У меня появился живот"
    reset_user_profile(user_id)

    result = asyncio.run(
        fake_openai.run_dialog_engine(user_text, get_user_profile(user_id), user_id)
    )

    shadow_memory = get_shadow_discovery_memory(user_id)

    assert shadow_memory.human_model.items == {}
    assert shadow_memory.human_model.relations == {}
    assert shadow_memory.active_conversation_state is None
    assert [message.sequence for message in shadow_memory.dialogue_history.messages] == [1, 2]
    assert shadow_memory.dialogue_history.messages[0].role.value == "user"
    assert shadow_memory.dialogue_history.messages[0].text == user_text
    assert shadow_memory.dialogue_history.messages[1].role.value == "system"
    assert shadow_memory.dialogue_history.messages[1].text == result["reply"]
    validate_discovery_memory(
        shadow_memory.human_model,
        shadow_memory.active_conversation_state,
        shadow_memory.dialogue_history,
    )


def test_second_adjacent_turn_preserves_shadow_history(fake_openai):
    user_id = 402
    reset_user_profile(user_id)

    first_result = asyncio.run(
        fake_openai.run_dialog_engine("У меня появился живот", get_user_profile(user_id), user_id)
    )
    second_result = asyncio.run(
        fake_openai.run_dialog_engine("Да", get_user_profile(user_id), user_id)
    )

    shadow_memory = get_shadow_discovery_memory(user_id)
    messages = shadow_memory.dialogue_history.messages

    assert [message.sequence for message in messages] == [1, 2, 3, 4]
    assert [message.text for message in messages] == [
        "У меня появился живот",
        first_result["reply"],
        "Да",
        second_result["reply"],
    ]
    assert shadow_memory.human_model.items == {}
    assert shadow_memory.human_model.relations == {}
    assert shadow_memory.active_conversation_state is None


def test_shadow_discovery_memory_is_isolated_per_user(fake_openai):
    first_user_id = 403
    second_user_id = 404
    reset_user_profile(first_user_id)
    reset_user_profile(second_user_id)

    asyncio.run(
        fake_openai.run_dialog_engine("Первый пользователь", get_user_profile(first_user_id), first_user_id)
    )
    asyncio.run(
        fake_openai.run_dialog_engine("Второй пользователь", get_user_profile(second_user_id), second_user_id)
    )

    first_memory = get_shadow_discovery_memory(first_user_id)
    second_memory = get_shadow_discovery_memory(second_user_id)

    assert first_memory is not second_memory
    assert [message.sequence for message in first_memory.dialogue_history.messages] == [1, 2]
    assert [message.sequence for message in second_memory.dialogue_history.messages] == [1, 2]
    assert first_memory.dialogue_history.messages[0].text == "Первый пользователь"
    assert second_memory.dialogue_history.messages[0].text == "Второй пользователь"
    assert first_memory.human_model.items == second_memory.human_model.items == {}
    assert first_memory.human_model.relations == second_memory.human_model.relations == {}


def test_failed_turn_does_not_append_an_unpaired_shadow_message(monkeypatch):
    import ai.dialog_engine as dialog_engine

    user_id = 405
    reset_user_profile(user_id)
    previous_memory = get_shadow_discovery_memory(user_id)

    class FailedResponse:
        def generate(self, *args):
            raise RuntimeError("response failed")

    monkeypatch.setattr(dialog_engine, "ResponseEngine", FailedResponse)

    with pytest.raises(RuntimeError, match="response failed"):
        asyncio.run(
            dialog_engine.run_dialog_engine("Незавершённый ход", get_user_profile(user_id), user_id)
        )

    assert get_shadow_discovery_memory(user_id) == previous_memory
    assert get_shadow_discovery_memory(user_id).dialogue_history.messages == ()


def test_no_user_id_does_not_create_shadow_memory_and_keeps_return_contract(fake_openai):
    known_user_ids = set(user_profiles)
    profile = {"history": [], "last_question": None, "discovery_complete": False}

    result = asyncio.run(fake_openai.run_dialog_engine("Мне 30 лет", profile))

    assert set(user_profiles) == known_user_ids
    assert set(result) == {"reply", "update"}
    assert result["update"]["age"] == 30


def test_legacy_state_continues_to_update_without_receiving_shadow_memory(fake_openai, monkeypatch):
    import ai.dialog_engine as dialog_engine

    user_id = 406
    reset_user_profile(user_id)
    received_memories = []
    original_human_model_engine = dialog_engine.HumanModelEngine

    class HumanModelEngineWithoutShadow:
        def __init__(self):
            self._delegate = original_human_model_engine()

        def build(self, semantic_context, previous_human_model, memory):
            received_memories.append(memory)
            assert "shadow_discovery_memory" not in memory
            return self._delegate.build(semantic_context, previous_human_model, memory)

        def apply_update(self, profile, update):
            return self._delegate.apply_update(profile, update)

        def is_discovery_complete(self, model_or_profile):
            return self._delegate.is_discovery_complete(model_or_profile)

    monkeypatch.setattr(dialog_engine, "HumanModelEngine", HumanModelEngineWithoutShadow)

    result = asyncio.run(
        fake_openai.run_dialog_engine("Мне 30 лет", get_user_profile(user_id), user_id)
    )

    memory = get_user_memory(user_id)
    assert result["reply"]
    assert received_memories
    assert memory["history"][-1]["content"] == result["reply"]
    assert get_human_model(user_id).facts["age"] == 30
