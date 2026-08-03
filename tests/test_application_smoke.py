import asyncio
import importlib


def test_main_imports(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "123456:smoke-test-token")
    asyncio.set_event_loop(asyncio.new_event_loop())

    module = importlib.import_module("main")

    assert module.dp is not None


def test_current_dialog_pipeline(fake_openai, user_profile):
    result = asyncio.run(
        fake_openai.run_dialog_engine("Мне 30 лет", user_profile)
    )

    assert result["reply"] == "Test reply"
    assert result["update"]["age"] == 30
