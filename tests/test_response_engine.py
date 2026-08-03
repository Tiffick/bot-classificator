from copy import deepcopy
from types import SimpleNamespace

from ai.engines.decision_engine import DecisionContext
from ai.engines.emotional_engine import EmotionalContext
from ai.engines.human_model import HumanModel
from ai.engines.impact_engine import ImpactContext
from ai.engines.reasoning_engine import ReasoningContext
from ai.engines.response_engine import ResponseEngine
from ai.engines.semantic_engine import SemanticContext


class FakeCompletions:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.contents)))]
        )


class FakeClient:
    def __init__(self, contents):
        self.chat = SimpleNamespace(completions=FakeCompletions(contents))


def _contexts():
    return (
        SemanticContext(topics=["weight"], facts={"age": 30}, confidence=1.0),
        HumanModel(facts={"age": 30}, goals=["lose_weight"]),
        ReasoningContext(missing_information=["motivation"]),
        DecisionContext(next_goal="clarify_weight_impact", response_type="question"),
        ImpactContext(main_goal="increase_self_understanding"),
        EmotionalContext(tone="reflective", response_length="medium"),
        {
            "facts": {"age": 30},
            "history": [{"role": "user", "content": "Хочу похудеть"}],
            "current_message": "я толстый",
        },
    )


def test_response_engine_regenerates_after_failed_validation():
    client = FakeClient(
        [
            "Понимаю. Как вес влияет на твою энергию?",
            '{"is_valid": false, "reason": "не соответствует тону"}',
            "Понимаю. Больше беспокоит сам вес или то, как он влияет на энергию?",
            '{"is_valid": true, "reason": ""}',
        ]
    )
    engine = ResponseEngine(client=client)

    result = engine.generate(*_contexts())

    assert result == "Понимаю. Больше беспокоит сам вес или то, как он влияет на энергию?"
    assert len(client.chat.completions.calls) == 4
    generation_prompt = client.chat.completions.calls[0]["messages"][-1]["content"]
    for context_name in (
        "SemanticContext",
        "HumanModel",
        "ReasoningContext",
        "DecisionContext",
        "ImpactContext",
        "EmotionalContext",
        "Memory",
        "Knowledge Base",
    ):
        assert context_name in generation_prompt
    assert '"current_message": "я толстый"' in generation_prompt


def test_response_engine_does_not_mutate_its_inputs():
    contexts = _contexts()
    before = deepcopy(contexts)
    engine = ResponseEngine(
        client=FakeClient(["Как вес влияет на твоё самочувствие?", '{"is_valid": true, "reason": ""}'])
    )

    engine.generate(*contexts)

    assert contexts == before


def test_response_engine_rejects_formal_long_and_multiple_action_reply():
    engine = ResponseEngine(client=FakeClient([]))
    decision_context = DecisionContext(response_type="question")
    formal_reply = (
        "Спасибо, что поделились своими мыслями. Чтобы помочь вам лучше, мне "
        "важно понять, какие цели вы видите для себя сейчас. Что важно: "
        "самочувствие, энергия или внешний вид? Расскажите подробнее?"
    )
    concise_reply = "Понимаю. Как давно вес начал тебя напрягать?"

    assert not engine._has_concise_single_action(formal_reply, decision_context)
    assert engine._has_concise_single_action(concise_reply, decision_context)


def test_response_engine_rejects_question_without_new_information():
    engine = ResponseEngine(client=FakeClient([]))
    decision_context = DecisionContext(
        next_goal="clarify_goals",
        reason="missing_information:goals",
        response_type="question",
    )
    repeated_question = (
        "Понимаю, это важно. Что именно тебя больше всего беспокоит в своём весе сейчас?"
    )
    concrete_question = "Понимаю. Больше беспокоит сам вес или то, как он влияет на энергию?"

    assert not engine._question_requests_new_information(
        repeated_question, "я толстый", decision_context
    )
    assert engine._question_requests_new_information(
        concrete_question, "я толстый", decision_context
    )


def test_response_engine_keeps_decision_selected_question_axis():
    engine = ResponseEngine(client=FakeClient([]))
    impact_decision = DecisionContext(
        next_goal="clarify_weight_impact", response_type="question"
    )
    duration_decision = DecisionContext(
        next_goal="clarify_problem_duration", response_type="question"
    )
    impact_question = "Понимаю. Как вес влияет на твоё самочувствие?"
    duration_question = "Понимаю. Как давно это тебя беспокоит?"

    assert engine._question_requests_new_information(
        impact_question, "я толстый", impact_decision
    )
    assert not engine._question_requests_new_information(
        duration_question, "я толстый", impact_decision
    )
    assert engine._question_requests_new_information(
        duration_question, "я толстый", duration_decision
    )
    assert not engine._question_requests_new_information(
        impact_question, "я толстый", duration_decision
    )


def test_response_engine_fallback_uses_decision_context():
    contexts = list(_contexts())
    contexts[3] = DecisionContext(
        next_goal="clarify_goals",
        reason="missing_information:goals",
        response_type="question",
    )
    engine = ResponseEngine(
        client=FakeClient(
            [
                "Понимаю. Что тебя беспокоит в весе?",
                "Понимаю. Что тебя беспокоит в весе?",
            ]
        )
    )

    result = engine.generate(*contexts)

    assert result == "Понимаю. Что тебе хотелось бы изменить в первую очередь?"


def test_weight_concern_never_returns_a_restatement_question():
    engine = ResponseEngine(client=FakeClient([]))
    decision_context = DecisionContext(
        next_goal="clarify_weight_impact", response_type="question"
    )
    repeated_question = "Понимаю. Что именно тебя беспокоит в весе?"

    assert not engine._question_requests_new_information(
        repeated_question, "я толстый", decision_context
    )
