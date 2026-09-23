from dotenv import load_dotenv
from openai import OpenAI

from ai.cognitive_core import CognitiveCore
from ai.discovery_data_model import (
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    validate_discovery_memory,
)
from ai.discovery_state_applier import apply_cognitive_turn
from memory.user_memory import (
    ShadowDiscoveryMemory,
    get_user_memory,
    set_shadow_discovery_memory,
)


def _empty_discovery_memory() -> ShadowDiscoveryMemory:
    return ShadowDiscoveryMemory(
        human_model=HumanModel(),
        active_conversation_state=None,
        dialogue_history=DialogueHistory(),
    )


def _load_discovery_memory(user_id) -> ShadowDiscoveryMemory:
    if user_id is None:
        return _empty_discovery_memory()
    stored_memory = get_user_memory(user_id)["shadow_discovery_memory"]
    return stored_memory if stored_memory is not None else _empty_discovery_memory()


def _next_message(
    history: DialogueHistory,
    role: DialogueRole,
    text: str,
) -> DialogueMessage:
    next_sequence = history.messages[-1].sequence + 1 if history.messages else 1
    return DialogueMessage(role=role, text=text, sequence=next_sequence)


def _compatibility_update(profile: dict, user_text: str, reply: str) -> dict:
    """Preserve the external update shape without interpreting Discovery semantics."""
    history = list(profile.get("history", ()))
    history.extend(
        (
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        )
    )
    return {
        "history": history[-20:],
        "discovery_complete": profile.get("discovery_complete", False),
    }


async def run_dialog_engine(user_text: str, profile: dict, user_id=None):
    """Run one Cognitive Core turn and atomically persist its Discovery state."""
    discovery_memory = _load_discovery_memory(user_id)
    validate_discovery_memory(
        discovery_memory.human_model,
        discovery_memory.active_conversation_state,
        discovery_memory.dialogue_history,
    )
    user_message = _next_message(
        discovery_memory.dialogue_history,
        DialogueRole.USER,
        user_text,
    )

    load_dotenv()
    turn_result = CognitiveCore(client=OpenAI(max_retries=0)).propose(
        user_message,
        discovery_memory.human_model,
        discovery_memory.active_conversation_state,
        discovery_memory.dialogue_history,
    )
    reply = "".join(segment.text for segment in turn_result.reply_segments)
    system_message = DialogueMessage(
        role=DialogueRole.SYSTEM,
        text=reply,
        sequence=user_message.sequence + 1,
    )
    applied_turn = apply_cognitive_turn(
        discovery_memory.human_model,
        discovery_memory.active_conversation_state,
        discovery_memory.dialogue_history,
        user_message,
        system_message,
        turn_result,
    )

    if user_id is not None:
        set_shadow_discovery_memory(
            user_id,
            ShadowDiscoveryMemory(
                human_model=applied_turn.updated_human_model,
                active_conversation_state=applied_turn.new_acs,
                dialogue_history=applied_turn.updated_history,
            ),
        )

    return {
        "reply": reply,
        "update": _compatibility_update(profile, user_text, reply),
    }
