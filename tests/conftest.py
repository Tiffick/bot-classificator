from types import SimpleNamespace

import pytest


@pytest.fixture
def user_profile():
    return {
        "history": [],
        "last_question": None,
        "discovery_complete": False,
    }


@pytest.fixture
def fake_openai(monkeypatch):
    import ai.dialog_engine as dialog_engine

    class FakeCompletions:
        def create(self, **kwargs):
            content = '{"update": {"age": 30}, "reply": "Test reply"}'
            message = SimpleNamespace(content=content)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

    class FakeOpenAI:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(dialog_engine, "OpenAI", FakeOpenAI)
    return dialog_engine
