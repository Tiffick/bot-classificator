import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from ai.cognitive_core import CognitiveCore
from ai.cognitive_turn import (
    ActionContent,
    CognitiveTurnResult,
    DecisionIntent,
    ReconciliationOutcome,
    ReplySegment,
    SystemAction,
)
from ai.discovery_data_model import (
    ActiveContentKind,
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    TargetInteractionKind,
    validate_discovery_memory,
)
from ai.discovery_state_applier import apply_cognitive_turn
from memory.user_memory import (
    ShadowDiscoveryMemory,
    get_user_memory,
    get_telegram_consultant_handoff,
    set_shadow_discovery_memory,
    set_telegram_consultant_handoff,
)
from ai.telegram_consultant_handoff import (
    CONSULTANT_CHAT_ID,
    CONTACT_REQUEST_TEXT,
    DELIVERY_FAILURE_TEXT,
    FINAL_USER_TEXT,
    INVALID_CONTACT_TEXT,
    TelegramConsultantHandoffState,
    TelegramHandoffStatus,
    build_consultant_summary,
    normalize_phone,
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


def _handoff_update(profile: dict, event: str, reply: str, *, complete: bool) -> dict:
    history = list(profile.get("history", ()))
    history.extend(
        (
            {"role": "user", "content": event},
            {"role": "assistant", "content": reply},
        )
    )
    return {"history": history[-20:], "discovery_complete": complete}


def _consented_transition(previous_acs, turn_result: CognitiveTurnResult) -> bool:
    if previous_acs is None:
        return False
    for resolution in turn_result.reconciliation:
        if resolution.outcome != ReconciliationOutcome.CONSENTED:
            continue
        for target_id in resolution.previous_response_target_ids:
            target = previous_acs.response_targets.get(target_id)
            if target is None or target.interaction != TargetInteractionKind.CONSENT:
                continue
            content = previous_acs.active_content.get(target.active_content_id)
            if content is not None and content.kind == ActiveContentKind.SYSTEM_TRANSITION:
                return True
    return False


def _contact_request_turn(turn_result: CognitiveTurnResult) -> CognitiveTurnResult:
    content = ActionContent(
        local_id="content:contact_request",
        kind=ActiveContentKind.SYSTEM_TRANSITION,
        semantic_content=CONTACT_REQUEST_TEXT,
    )
    return CognitiveTurnResult(
        decision_intent=DecisionIntent.TRANSITION,
        state_patch=turn_result.state_patch,
        reconciliation=turn_result.reconciliation,
        system_action=SystemAction(contents=(content,)),
        reply_segments=(
            ReplySegment(
                local_id="segment:contact_request",
                text=CONTACT_REQUEST_TEXT,
                realizes_action_content_ids=(content.local_id,),
            ),
        ),
    )


async def _run_handoff_step(
    *,
    user_id,
    profile: dict,
    user_text: str,
    contact_phone,
    consultant_sender,
):
    state = get_telegram_consultant_handoff(user_id)
    if state.status == TelegramHandoffStatus.COMPLETED:
        return {
            "reply": FINAL_USER_TEXT,
            "update": _handoff_update(
                profile,
                "Передача консультанту уже завершена.",
                FINAL_USER_TEXT,
                complete=True,
            ),
            "remove_contact_keyboard": True,
        }

    if state.status == TelegramHandoffStatus.AWAITING_CONTACT:
        raw_phone = contact_phone if contact_phone is not None else user_text
        normalized_phone = normalize_phone(raw_phone)
        if normalized_phone is None:
            return {
                "reply": INVALID_CONTACT_TEXT,
                "update": _handoff_update(
                    profile,
                    "Получен некорректный контакт.",
                    INVALID_CONTACT_TEXT,
                    complete=False,
                ),
                "request_contact": True,
            }
        discovery_memory = _load_discovery_memory(user_id)
        summary = build_consultant_summary(discovery_memory.human_model, normalized_phone)
        state = TelegramConsultantHandoffState(
            status=TelegramHandoffStatus.DELIVERY_PENDING,
            raw_phone=raw_phone,
            normalized_phone=normalized_phone,
            consultant_summary=summary,
        )
        set_telegram_consultant_handoff(user_id, state)

    if consultant_sender is None:
        return {
            "reply": DELIVERY_FAILURE_TEXT,
            "update": _handoff_update(
                profile,
                "Контакт сохранён; передача временно недоступна.",
                DELIVERY_FAILURE_TEXT,
                complete=False,
            ),
        }

    try:
        await consultant_sender(CONSULTANT_CHAT_ID, state.consultant_summary)
    except Exception:
        return {
            "reply": DELIVERY_FAILURE_TEXT,
            "update": _handoff_update(
                profile,
                "Контакт сохранён; передача завершилась технической ошибкой.",
                DELIVERY_FAILURE_TEXT,
                complete=False,
            ),
        }

    set_telegram_consultant_handoff(
        user_id,
        TelegramConsultantHandoffState(
            status=TelegramHandoffStatus.COMPLETED,
            raw_phone=state.raw_phone,
            normalized_phone=state.normalized_phone,
            consultant_summary=state.consultant_summary,
        ),
    )
    return {
        "reply": FINAL_USER_TEXT,
        "update": _handoff_update(
            profile,
            "Контакт передан консультанту.",
            FINAL_USER_TEXT,
            complete=True,
        ),
        "remove_contact_keyboard": True,
    }


async def run_dialog_engine(
    user_text: str,
    profile: dict,
    user_id=None,
    *,
    contact_phone=None,
    consultant_sender=None,
):
    """Run one Cognitive Core turn and atomically persist its Discovery state."""
    if user_id is not None and get_telegram_consultant_handoff(user_id) is not None:
        return await _run_handoff_step(
            user_id=user_id,
            profile=profile,
            user_text=user_text,
            contact_phone=contact_phone,
            consultant_sender=consultant_sender,
        )

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
    core = CognitiveCore(client=OpenAI(max_retries=0))
    turn_result = await asyncio.to_thread(
        core.propose,
        user_message,
        discovery_memory.human_model,
        discovery_memory.active_conversation_state,
        discovery_memory.dialogue_history,
    )
    starts_contact_collection = _consented_transition(
        discovery_memory.active_conversation_state, turn_result
    )
    if starts_contact_collection:
        turn_result = _contact_request_turn(turn_result)
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
        if starts_contact_collection:
            set_telegram_consultant_handoff(
                user_id,
                TelegramConsultantHandoffState(
                    status=TelegramHandoffStatus.AWAITING_CONTACT
                ),
            )

    result = {
        "reply": reply,
        "update": _compatibility_update(profile, user_text, reply),
    }
    if starts_contact_collection:
        result["request_contact"] = True
    return result
