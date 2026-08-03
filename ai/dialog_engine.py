from ai.engines.decision_engine import DecisionEngine
from ai.engines.emotional_engine import EmotionalEngine
from ai.engines.human_model_engine import HumanModelEngine
from ai.engines.impact_engine import ImpactEngine
from ai.engines.reasoning_engine import ReasoningEngine
from ai.engines.response_engine import ResponseEngine
from ai.engines.semantic_engine import SemanticEngine
from memory.user_memory import get_human_model, get_user_memory, set_human_model


def append_history(history: list, user_text: str, reply: str) -> None:
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})


async def run_dialog_engine(user_text: str, profile: dict, user_id=None):
    """Orchestrate one full consultation cycle and preserve its result."""
    history = profile.get("history", [])
    human_model_engine = HumanModelEngine()
    semantic_engine = SemanticEngine()
    reasoning_engine = ReasoningEngine()
    decision_engine = DecisionEngine()
    impact_engine = ImpactEngine()
    emotional_engine = EmotionalEngine()
    response_engine = ResponseEngine()

    semantic_context = semantic_engine.analyze(user_text)
    memory = get_user_memory(user_id) if user_id is not None else {"facts": profile}
    previous_human_model = get_human_model(user_id) if user_id is not None else None
    human_model = human_model_engine.build(
        semantic_context, previous_human_model, memory
    )
    reasoning_context = reasoning_engine.reason(
        semantic_context, human_model, memory
    )
    decision_context = decision_engine.decide(
        semantic_context, human_model, reasoning_context, memory
    )
    impact_context = impact_engine.evaluate(
        semantic_context, human_model, reasoning_context, decision_context, memory
    )
    emotional_context = emotional_engine.choose(
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        memory,
    )
    response_memory = {**memory, "current_message": user_text}
    reply = response_engine.generate(
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        emotional_context,
        response_memory,
    )

    safe_update = human_model_engine.apply_update(profile, semantic_context.facts)
    append_history(history, user_text, reply)
    temp_profile = profile.copy()
    temp_profile.update(safe_update)
    safe_update["history"] = history[-20:]
    safe_update["discovery_complete"] = human_model_engine.is_discovery_complete(
        temp_profile
    )

    if user_id is not None:
        set_human_model(user_id, human_model)

    return {"reply": reply, "update": safe_update}
