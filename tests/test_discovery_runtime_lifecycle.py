import asyncio

from ai.discovery_data_model import validate_discovery_memory
from memory.user_memory import (
    get_shadow_discovery_memory,
    get_user_profile,
    reset_user_profile,
    user_profiles,
)


def test_first_turn_persists_valid_discovery_memory(fake_openai):
    user_id = 401
    user_text = "У меня появился живот"
    reset_user_profile(user_id)

    result = asyncio.run(
        fake_openai.run_dialog_engine(user_text, get_user_profile(user_id), user_id)
    )
    memory = get_shadow_discovery_memory(user_id)

    assert [message.sequence for message in memory.dialogue_history.messages] == [1, 2]
    assert [message.text for message in memory.dialogue_history.messages] == [
        user_text,
        result["reply"],
    ]
    assert memory.active_conversation_state is not None
    validate_discovery_memory(
        memory.human_model,
        memory.active_conversation_state,
        memory.dialogue_history,
    )


def test_second_turn_preserves_discovery_history_and_replaces_acs(fake_openai):
    user_id = 402
    reset_user_profile(user_id)

    first_result = asyncio.run(
        fake_openai.run_dialog_engine(
            "У меня появился живот", get_user_profile(user_id), user_id
        )
    )
    first_acs = get_shadow_discovery_memory(user_id).active_conversation_state
    second_result = asyncio.run(
        fake_openai.run_dialog_engine("Да", get_user_profile(user_id), user_id)
    )
    memory = get_shadow_discovery_memory(user_id)

    assert [message.sequence for message in memory.dialogue_history.messages] == [1, 2, 3, 4]
    assert [message.text for message in memory.dialogue_history.messages] == [
        "У меня появился живот",
        first_result["reply"],
        "Да",
        second_result["reply"],
    ]
    assert memory.active_conversation_state is not None
    assert memory.active_conversation_state != first_acs


def test_discovery_memory_is_isolated_per_user(fake_openai):
    first_user_id = 403
    second_user_id = 404
    reset_user_profile(first_user_id)
    reset_user_profile(second_user_id)

    asyncio.run(
        fake_openai.run_dialog_engine(
            "Первый пользователь", get_user_profile(first_user_id), first_user_id
        )
    )
    asyncio.run(
        fake_openai.run_dialog_engine(
            "Второй пользователь", get_user_profile(second_user_id), second_user_id
        )
    )
    first_memory = get_shadow_discovery_memory(first_user_id)
    second_memory = get_shadow_discovery_memory(second_user_id)

    assert first_memory is not second_memory
    assert first_memory.dialogue_history.messages[0].text == "Первый пользователь"
    assert second_memory.dialogue_history.messages[0].text == "Второй пользователь"
    assert first_memory.dialogue_history.messages[0].id != second_memory.dialogue_history.messages[0].id


def test_no_user_id_does_not_create_persistent_memory(fake_openai):
    known_user_ids = set(user_profiles)
    profile = {"history": [], "last_question": None, "discovery_complete": False}

    result = asyncio.run(fake_openai.run_dialog_engine("Мне 30 лет", profile))

    assert set(user_profiles) == known_user_ids
    assert set(result) == {"reply", "update"}
    assert result["update"]["discovery_complete"] is False
    assert [entry["role"] for entry in result["update"]["history"]] == [
        "user",
        "assistant",
    ]
