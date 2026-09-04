from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from ai.discovery_data_model import (
    ActiveConversationState,
    DialogueHistory,
    HumanModel as DiscoveryHumanModel,
)


FACT_DEFAULTS = {
    "name": None,
    "gender": None,
    "age": None,
    "height": None,
    "current_weight": None,
    "target_weight": None,
    "duration": None,
    "main_problem": None,
    "previous_attempts": None,
    "failure_reason": None,
    "appearance": False,
    "clothes": False,
    "breathing": False,
    "energy": False,
    "mobility": False,
    "sleep": False,
    "health": False,
}

user_profiles = {}


@dataclass(frozen=True)
class ShadowDiscoveryMemory:
    """Technical container for isolated Stage 1 Discovery lifecycle state."""

    human_model: DiscoveryHumanModel
    active_conversation_state: Optional[ActiveConversationState]
    dialogue_history: DialogueHistory


def _new_user_memory():
    return {
        "facts": FACT_DEFAULTS.copy(),
        "history": [],
        "human_model": None,
        "shadow_discovery_memory": None,
        "conversation_state": {
            "last_question": None,
            "discovery_complete": False,
        },
    }


def get_user_memory(user_id):
    if user_id not in user_profiles:
        user_profiles[user_id] = _new_user_memory()

    return user_profiles[user_id]


def get_user_profile(user_id):
    memory = get_user_memory(user_id)
    profile = memory["facts"].copy()
    profile["history"] = memory["history"]
    profile.update(memory["conversation_state"])

    return profile


def update_user_profile(user_id, new_data):
    memory = get_user_memory(user_id)

    for key, value in new_data.items():
        if value is None:
            continue

        if key == "history":
            memory["history"] = value
        elif key in memory["conversation_state"]:
            memory["conversation_state"][key] = value
        elif key in memory["facts"]:
            memory["facts"][key] = value

    return get_user_profile(user_id)


def set_last_question(user_id, question):
    get_user_memory(user_id)["conversation_state"]["last_question"] = question


def get_human_model(user_id):
    human_model = get_user_memory(user_id)["human_model"]
    return deepcopy(human_model)


def set_human_model(user_id, human_model):
    get_user_memory(user_id)["human_model"] = deepcopy(human_model)


def get_shadow_discovery_memory(user_id) -> ShadowDiscoveryMemory:
    """Return the user's isolated, immutable Stage 1 Discovery state."""
    memory = get_user_memory(user_id)
    shadow_memory = memory["shadow_discovery_memory"]
    if shadow_memory is None:
        shadow_memory = ShadowDiscoveryMemory(
            human_model=DiscoveryHumanModel(),
            active_conversation_state=None,
            dialogue_history=DialogueHistory(),
        )
        memory["shadow_discovery_memory"] = shadow_memory
    return shadow_memory


def set_shadow_discovery_memory(
    user_id, shadow_memory: ShadowDiscoveryMemory
) -> None:
    """Persist a validated replacement of the user's Stage 1 shadow state."""
    get_user_memory(user_id)["shadow_discovery_memory"] = shadow_memory


def reset_user_profile(user_id):
    user_profiles[user_id] = _new_user_memory()
