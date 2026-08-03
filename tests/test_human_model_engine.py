from ai.engines.human_model import HumanModel
from ai.engines.human_model_engine import HumanModelEngine
from ai.engines.semantic_engine import SemanticEngine


def test_build_updates_model_from_semantic_context_and_memory():
    previous_model = HumanModel(facts={"age": 29}, goals=["lose_weight"])
    memory = {
        "facts": {
            "age": 29,
            "current_weight": 90.0,
            "history": "must not become a fact",
            "energy": False,
        }
    }
    semantic_context = SemanticEngine().analyze(
        "Мне 30 лет, хочу похудеть, но не получается."
    )

    model = HumanModelEngine().build(
        semantic_context,
        previous_model,
        memory,
    )

    assert model.facts["age"] == 30
    assert model.facts["current_weight"] == 90.0
    assert "history" not in model.facts
    assert model.goals == ["lose_weight"]
    assert "не получается" in model.barriers
    assert "frustration" in model.emotional_features
    assert previous_model.facts == {"age": 29}


def test_build_preserves_uncertainty_without_creating_hypotheses():
    context = SemanticEngine().analyze("Не знаю.")

    model = HumanModelEngine().build(context, None, {"facts": {}})

    assert model.hypotheses == {}
    assert model.confidence == {}
    assert model.facts == {}
