import logging

from ai.engines.decision_engine import DecisionEngine
from ai.engines.emotional_engine import EmotionalEngine
from ai.engines.human_model_engine import HumanModelEngine
from ai.engines.impact_engine import ImpactEngine
from ai.engines.reasoning_engine import ReasoningEngine
from ai.engines.response_engine import ResponseEngine
from ai.engines.llm_semantic_engine import LLMSemanticEngine
from ai.engines.semantic_engine import SemanticEngine
from ai.discovery_data_model import (
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    validate_discovery_memory,
)
from memory.user_memory import (
    ShadowDiscoveryMemory,
    get_human_model,
    get_shadow_discovery_memory,
    get_user_memory,
    set_human_model,
    set_shadow_discovery_memory,
)


LOGGER = logging.getLogger(__name__)


def append_history(history: list, user_text: str, reply: str) -> None:
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})


def _append_shadow_turn(
    shadow_memory: ShadowDiscoveryMemory,
    user_text: str,
    reply: str,
) -> ShadowDiscoveryMemory:
    """Build a validated immutable shadow record for one completed turn pair."""
    messages = shadow_memory.dialogue_history.messages
    next_sequence = messages[-1].sequence + 1 if messages else 1
    dialogue_history = DialogueHistory(
        messages=(
            *messages,
            DialogueMessage(
                role=DialogueRole.USER,
                text=user_text,
                sequence=next_sequence,
            ),
            DialogueMessage(
                role=DialogueRole.SYSTEM,
                text=reply,
                sequence=next_sequence + 1,
            ),
        )
    )
    updated_memory = ShadowDiscoveryMemory(
        human_model=shadow_memory.human_model,
        active_conversation_state=None,
        dialogue_history=dialogue_history,
    )
    validate_discovery_memory(
        updated_memory.human_model,
        updated_memory.active_conversation_state,
        updated_memory.dialogue_history,
    )
    return updated_memory


async def run_dialog_engine(user_text: str, profile: dict, user_id=None):
    """Orchestrate one full consultation cycle and preserve its result."""
    history = profile.get("history", [])
    shadow_memory = None
    if user_id is not None:
        shadow_memory = get_shadow_discovery_memory(user_id)
        legacy_memory = {
            key: value
            for key, value in get_user_memory(user_id).items()
            if key != "shadow_discovery_memory"
        }
    else:
        legacy_memory = {"facts": profile}
    human_model_engine = HumanModelEngine()
    llm_semantic_engine = LLMSemanticEngine()
    reasoning_engine = ReasoningEngine()
    decision_engine = DecisionEngine()
    impact_engine = ImpactEngine()
    emotional_engine = EmotionalEngine()
    response_engine = ResponseEngine()

    semantic_context = llm_semantic_engine.analyze(user_text)
    if not llm_semantic_engine.last_diagnostics.get("success", False):
        fallback_reason = llm_semantic_engine.last_diagnostics.get("fallback_reason")
        LOGGER.warning(
            "LLM Semantic Engine failed; using deterministic fallback: %s",
            fallback_reason,
        )
        semantic_context = SemanticEngine().analyze(user_text)
    elif llm_semantic_engine.last_diagnostics.get("rejected_fields"):
        LOGGER.info(
            "LLM Semantic Engine discarded unconfirmed values: %s",
            llm_semantic_engine.last_diagnostics["rejected_fields"],
        )
    previous_human_model = get_human_model(user_id) if user_id is not None else None
    human_model = human_model_engine.build(
        semantic_context, previous_human_model, legacy_memory
    )
    reasoning_context = reasoning_engine.reason(
        semantic_context, human_model, legacy_memory
    )
    decision_context = decision_engine.decide(
        semantic_context, human_model, reasoning_context, legacy_memory
    )
    impact_context = impact_engine.evaluate(
        semantic_context, human_model, reasoning_context, decision_context, legacy_memory
    )
    emotional_context = emotional_engine.choose(
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        legacy_memory,
    )
    response_memory = {**legacy_memory, "current_message": user_text}
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
    safe_update["history"] = history[-20:]
    safe_update["discovery_complete"] = human_model_engine.is_discovery_complete(
        human_model
    )

    if user_id is not None:
        set_human_model(user_id, human_model)
        set_shadow_discovery_memory(
            user_id, _append_shadow_turn(shadow_memory, user_text, reply)
        )

    return {"reply": reply, "update": safe_update}
