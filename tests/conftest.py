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
    import ai.engines.response_engine as response_engine
    from ai.engines.semantic_engine import SemanticEngine

    class FakeCompletions:
        def create(self, **kwargs):
            prompt = kwargs["messages"][-1]["content"]
            content = (
                '{"is_valid": true, "reason": ""}'
                if "ПРОВЕРЬ ОТВЕТ" in prompt
                else "Понимаю. Как давно это тебя беспокоит?"
            )
            message = SimpleNamespace(content=content)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

    class FakeOpenAI:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    class FakeLLMSemanticEngine:
        def __init__(self):
            self.last_diagnostics = {"success": True}

        def analyze(self, user_text):
            return SemanticEngine().analyze(user_text)

    monkeypatch.setattr(response_engine, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(dialog_engine, "LLMSemanticEngine", FakeLLMSemanticEngine)
    return dialog_engine
