from ai.engines.semantic_engine import SemanticEngine
from tests.fixtures.semantic_messages import (
    AMBIGUOUS_MESSAGE,
    DURATION_MESSAGE,
    EMPTY_MESSAGE,
    EXPLICIT_FACT_MESSAGE,
    QUESTION_MESSAGE,
)


def test_extracts_explicit_facts_without_interpreting_person():
    context = SemanticEngine().analyze(EXPLICIT_FACT_MESSAGE)

    assert context.facts == {
        "age": 30,
        "current_weight": 92.0,
        "height": 175,
    }
    assert "weight" in context.topics
    assert context.intent == "information_statement"
    assert context.confidence == 1.0
    assert context.known == []
    assert context.unknown == []


def test_records_question_as_message_content():
    context = SemanticEngine().analyze(QUESTION_MESSAGE)

    assert context.intent == "question"
    assert context.questions == [QUESTION_MESSAGE]
    assert "mobility" in context.topics
    assert context.facts == {}


def test_does_not_treat_duration_as_age():
    context = SemanticEngine().analyze(DURATION_MESSAGE)

    assert context.facts["duration"] == "уже 5 лет"
    assert "age" not in context.facts


def test_preserves_uncertainty_for_ambiguous_message():
    context = SemanticEngine().analyze(AMBIGUOUS_MESSAGE)

    assert context.intent == "information_statement"
    assert context.facts == {}
    assert context.confidence == 0.5
    assert context.emotional_signals == []


def test_marks_self_description_as_weight_topic_without_interpretation():
    context = SemanticEngine().analyze("я толстый")

    assert context.topics == ["weight"]
    assert context.facts == {}
    assert context.intent == "information_statement"


def test_returns_empty_context_for_empty_message():
    context = SemanticEngine().analyze(EMPTY_MESSAGE)

    assert context.intent == "unknown"
    assert context.topics == []
    assert context.confidence == 0.0
