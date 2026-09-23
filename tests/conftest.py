import pytest

from ai.cognitive_turn import (
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    DecisionIntent,
    ReplySegment,
    SystemAction,
)
from ai.discovery_data_model import (
    ActiveContentKind,
    TargetInteractionKind,
    TargetSubjectKind,
)


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

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.options = kwargs

    class FakeCognitiveCore:
        calls = []

        def __init__(self, client):
            self.client = client

        def propose(self, current, human_model, previous_acs, history):
            self.calls.append((current, human_model, previous_acs, history))
            content = ActionContent(
                local_id="content:question",
                kind=ActiveContentKind.SYSTEM_QUESTION,
                semantic_content="уточнить значимый пользовательский материал",
            )
            target = ActionTarget(
                local_id="target:question",
                active_content_local_id=content.local_id,
                subject=ActionSubjectReference(
                    kind=TargetSubjectKind.ACTIVE_CONTENT,
                    active_content_local_id=content.local_id,
                ),
                interaction=TargetInteractionKind.OPEN_RESPONSE,
            )
            return CognitiveTurnResult(
                decision_intent=DecisionIntent.HUMAN_DISCOVERY,
                system_action=SystemAction(
                    contents=(content,),
                    response_targets=(target,),
                ),
                reply_segments=(
                    ReplySegment(
                        local_id="segment:reply",
                        text="Понимаю. Как давно это тебя беспокоит?",
                        realizes_action_content_ids=(content.local_id,),
                    ),
                ),
            )

    FakeCognitiveCore.calls = []
    monkeypatch.setattr(dialog_engine, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(dialog_engine, "CognitiveCore", FakeCognitiveCore)
    return dialog_engine
