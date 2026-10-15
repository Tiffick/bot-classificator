import asyncio
import json

import ai.dialog_engine as dialog_engine
import bot.handlers as handlers
from ai.cognitive_diagnostics import (
    DIAGNOSTICS_FLAG,
    DIAGNOSTICS_PATH,
    conversation_identifier,
    diagnostics_enabled,
    start_diagnostic_session,
    write_turn_diagnostic,
)
from ai.knowledge_retrieval import KNOWLEDGE_RETRIEVAL_FLAG
from memory.user_memory import get_user_profile, reset_user_profile, update_user_profile
from test_cognitive_core import FakeClient, _semantic_baseline


def _document(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_diagnostics_default_on_and_explicit_false_disables(monkeypatch):
    monkeypatch.delenv(DIAGNOSTICS_FLAG, raising=False)
    assert diagnostics_enabled() is True
    monkeypatch.setenv(DIAGNOSTICS_FLAG, "false")
    assert diagnostics_enabled() is False


def test_start_replaces_prior_test_and_turns_accumulate_atomically(monkeypatch, tmp_path):
    path = tmp_path / "latest.json"
    monkeypatch.setenv(DIAGNOSTICS_FLAG, "true")
    monkeypatch.setenv(DIAGNOSTICS_PATH, str(path))
    user_id = 402960599

    start_diagnostic_session(user_id, start_exchange={"user_command": "/start"})
    write_turn_diagnostic(
        {
            "conversation_id": conversation_identifier(user_id),
            "turn_id": "turn-1",
            "status": "completed",
        }
    )
    write_turn_diagnostic(
        {
            "conversation_id": conversation_identifier(user_id),
            "turn_id": "turn-2",
            "status": "completed",
        }
    )
    assert [turn["turn_id"] for turn in _document(path)["turns"]] == [
        "turn-1",
        "turn-2",
    ]
    assert not path.with_suffix(".json.tmp").exists()

    start_diagnostic_session(user_id, start_exchange={"user_command": "/start"})
    document = _document(path)
    assert document["turns"] == []
    assert document["session"]["start_exchange"]["user_command"] == "/start"


def test_different_users_are_never_mixed_in_single_test_file(monkeypatch, tmp_path):
    path = tmp_path / "latest.json"
    monkeypatch.setenv(DIAGNOSTICS_FLAG, "true")
    monkeypatch.setenv(DIAGNOSTICS_PATH, str(path))
    first = conversation_identifier(1)
    second = conversation_identifier(2)

    write_turn_diagnostic(
        {"conversation_id": first, "turn_id": "first", "status": "completed"}
    )
    write_turn_diagnostic(
        {"conversation_id": second, "turn_id": "second", "status": "completed"}
    )

    document = _document(path)
    assert document["session"]["conversation_id"] == second
    assert [turn["turn_id"] for turn in document["turns"]] == ["second"]


def test_telegram_start_is_the_explicit_clean_test_boundary(monkeypatch, tmp_path):
    path = tmp_path / "latest.json"
    monkeypatch.setenv(DIAGNOSTICS_FLAG, "true")
    monkeypatch.setenv(DIAGNOSTICS_PATH, str(path))
    monkeypatch.setattr(handlers, "set_last_question", lambda *args: None)
    monkeypatch.setattr(handlers, "reset_user_profile", lambda *args: None)

    class Message:
        from_user = type("User", (), {"id": 91})()
        chat = type("Chat", (), {"id": 91})()
        message_id = 7

        async def answer(self, text):
            return type("Sent", (), {"message_id": 8})()

    asyncio.run(handlers.start_handler(Message()))

    document = _document(path)
    assert document["session"]["start_trigger"] == "telegram_start"
    assert document["session"]["start_exchange"]["user_command"] == "/start"
    assert document["session"]["start_exchange"]["telegram_delivery"] == "delivered"
    assert document["turns"] == []


def test_diagnostic_writer_failure_never_breaks_the_turn(monkeypatch, tmp_path):
    blocked = tmp_path / "occupied"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(DIAGNOSTICS_FLAG, "true")
    monkeypatch.setenv(DIAGNOSTICS_PATH, str(blocked / "latest.json"))

    assert start_diagnostic_session(1) is None
    assert write_turn_diagnostic(
        {
            "conversation_id": conversation_identifier(1),
            "turn_id": "turn",
        }
    ) is None


def test_real_dialog_engine_trace_has_exact_request_response_and_state_without_extra_call(
    monkeypatch, tmp_path
):
    path = tmp_path / "latest.json"
    user_id = 880099
    reset_user_profile(user_id)
    client = FakeClient(json.dumps(_semantic_baseline(), ensure_ascii=False))
    monkeypatch.setenv(DIAGNOSTICS_FLAG, "true")
    monkeypatch.setenv(DIAGNOSTICS_PATH, str(path))
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "false")
    monkeypatch.setattr(dialog_engine, "OpenAI", lambda **kwargs: client)

    result = asyncio.run(
        dialog_engine.run_dialog_engine("Хочу лучше понять свой набор веса.", {}, user_id)
    )

    assert len(client.completions.calls) == 1
    turn = _document(path)["turns"][0]
    core = turn["cognitive_core"]
    assert turn["status"] == "completed"
    assert turn["actual_reply"] == result["reply"]
    assert core["full_api_request"] == client.completions.calls[0]
    assert core["full_api_request"]["messages"][0]["content"]
    assert core["full_api_request"]["response_format"]["json_schema"]["schema"]
    assert core["raw_api_response"]["choices"][0]["message"]["content"]
    assert core["raw_semantic_turn"] == core["parsed_semantic_turn"]
    assert core["cognitive_turn_result"]
    assert core["llm_call_count"] == 1
    assert turn["discovery_memory_before"]["human_model"]["items"] == {}
    assert len(turn["discovery_memory_after"]["dialogue_history"]["messages"]) == 2
    assert len(turn["state_diff"]["dialogue_messages_added"]) == 2
    assert turn["validation"] == {
        "semantic_turn": "passed",
        "adapter": "passed",
        "cognitive_turn_result": "passed",
        "state_applier": "passed",
        "persistence": "passed",
    }
    assert turn["reply_matches_persisted_system_text"] is True


def test_disabled_diagnostics_do_not_change_dialog_result_or_create_file(
    monkeypatch, tmp_path
):
    path = tmp_path / "latest.json"
    user_id = 880100
    reset_user_profile(user_id)
    client = FakeClient(json.dumps(_semantic_baseline(), ensure_ascii=False))
    monkeypatch.setenv(DIAGNOSTICS_FLAG, "false")
    monkeypatch.setenv(DIAGNOSTICS_PATH, str(path))
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "false")
    monkeypatch.setattr(dialog_engine, "OpenAI", lambda **kwargs: client)

    result = asyncio.run(
        dialog_engine.run_dialog_engine(
            "Хочу понять ситуацию.", get_user_profile(user_id), user_id
        )
    )
    update_user_profile(user_id, result["update"])

    assert result["reply"]
    assert len(client.completions.calls) == 1
    assert not path.exists()
