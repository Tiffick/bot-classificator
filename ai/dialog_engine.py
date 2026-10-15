import asyncio
from time import monotonic
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from ai.cognitive_core import CognitiveCore
from ai.cognitive_diagnostics import (
    conversation_identifier,
    diagnostics_enabled,
    write_turn_diagnostic,
)
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


def _memory_summary(memory: ShadowDiscoveryMemory) -> dict:
    acs = memory.active_conversation_state
    return {
        "human_model_items": len(memory.human_model.items),
        "human_model_relations": len(memory.human_model.relations),
        "active_content": len(acs.active_content) if acs else 0,
        "response_targets": len(acs.response_targets) if acs else 0,
        "dialogue_messages": len(memory.dialogue_history.messages),
    }


def _memory_view(memory: ShadowDiscoveryMemory) -> dict:
    return {
        "human_model": memory.human_model.model_dump(mode="json"),
        "active_conversation_state": (
            memory.active_conversation_state.model_dump(mode="json")
            if memory.active_conversation_state is not None
            else None
        ),
        "dialogue_history": memory.dialogue_history.model_dump(mode="json"),
    }


def _collection_diff(before: dict, after: dict) -> dict:
    before_ids = set(before)
    after_ids = set(after)
    return {
        "added": {key: after[key] for key in sorted(after_ids - before_ids)},
        "removed": {key: before[key] for key in sorted(before_ids - after_ids)},
        "changed": {
            key: {"before": before[key], "after": after[key]}
            for key in sorted(before_ids & after_ids)
            if before[key] != after[key]
        },
    }


def _state_diff(before: ShadowDiscoveryMemory, after: ShadowDiscoveryMemory) -> dict:
    before_acs = before.active_conversation_state
    after_acs = after.active_conversation_state
    before_message_ids = {message.id for message in before.dialogue_history.messages}
    return {
        "human_model_items": _collection_diff(
            {
                key: value.model_dump(mode="json")
                for key, value in before.human_model.items.items()
            },
            {
                key: value.model_dump(mode="json")
                for key, value in after.human_model.items.items()
            },
        ),
        "human_model_relations": _collection_diff(
            {
                key: value.model_dump(mode="json")
                for key, value in before.human_model.relations.items()
            },
            {
                key: value.model_dump(mode="json")
                for key, value in after.human_model.relations.items()
            },
        ),
        "active_content": _collection_diff(
            {
                key: value.model_dump(mode="json")
                for key, value in (before_acs.active_content.items() if before_acs else ())
            },
            {
                key: value.model_dump(mode="json")
                for key, value in (after_acs.active_content.items() if after_acs else ())
            },
        ),
        "response_targets": _collection_diff(
            {
                key: value.model_dump(mode="json")
                for key, value in (before_acs.response_targets.items() if before_acs else ())
            },
            {
                key: value.model_dump(mode="json")
                for key, value in (after_acs.response_targets.items() if after_acs else ())
            },
        ),
        "dialogue_messages_added": [
            message.model_dump(mode="json")
            for message in after.dialogue_history.messages
            if message.id not in before_message_ids
        ],
    }


def _turn_fields(turn_result: CognitiveTurnResult) -> dict:
    payload = turn_result.model_dump(mode="json")
    return {
        "decision_intent": payload["decision_intent"],
        "state_patch": payload["state_patch"],
        "reconciliation": payload["reconciliation"],
        "action_content": payload["system_action"]["contents"],
        "response_targets": payload["system_action"]["response_targets"],
        "reply_segments": payload["reply_segments"],
    }


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
    started_at = monotonic()
    turn_id = f"handoff-{uuid4().hex}"
    state = get_telegram_consultant_handoff(user_id)

    def finish(result, *, event, delivery_status, error=None):
        result["_diagnostic_turn_id"] = turn_id
        current_state = get_telegram_consultant_handoff(user_id)
        write_turn_diagnostic(
            {
                "conversation_id": conversation_identifier(user_id),
                "turn_id": turn_id,
                "status": "completed",
                "pipeline": "telegram_consultant_handoff",
                "user_event": (
                    "telegram_contact_provided"
                    if contact_phone is not None
                    else "handoff_follow_up_message"
                ),
                "handoff": {
                    "event": event,
                    "status_before": state.status.value,
                    "status_after": (
                        current_state.status.value if current_state is not None else None
                    ),
                    "consultant_delivery": delivery_status,
                    "contact_present": bool(
                        current_state and current_state.normalized_phone
                    ),
                    "summary_present": bool(
                        current_state and current_state.consultant_summary
                    ),
                    "consultant_summary": (
                        current_state.consultant_summary if current_state else None
                    ),
                    "destination_chat_id": CONSULTANT_CHAT_ID,
                    "error": error,
                },
                "actual_reply": result["reply"],
                "trace": [
                    {"stage": "telegram_handler_to_dialog_engine", "status": "received"},
                    {"stage": "consultant_handoff", "status": event},
                    {"stage": "dialog_engine_reply", "status": "returned"},
                ],
                "total_latency_ms": round((monotonic() - started_at) * 1000, 3),
            }
        )
        return result

    if state.status == TelegramHandoffStatus.COMPLETED:
        return finish({
            "reply": FINAL_USER_TEXT,
            "update": _handoff_update(
                profile,
                "Передача консультанту уже завершена.",
                FINAL_USER_TEXT,
                complete=True,
            ),
            "remove_contact_keyboard": True,
        }, event="already_completed", delivery_status="not_attempted")

    if state.status == TelegramHandoffStatus.AWAITING_CONTACT:
        raw_phone = contact_phone if contact_phone is not None else user_text
        normalized_phone = normalize_phone(raw_phone)
        if normalized_phone is None:
            return finish({
                "reply": INVALID_CONTACT_TEXT,
                "update": _handoff_update(
                    profile,
                    "Получен некорректный контакт.",
                    INVALID_CONTACT_TEXT,
                    complete=False,
                ),
                "request_contact": True,
            }, event="invalid_contact", delivery_status="not_attempted")
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
        return finish({
            "reply": DELIVERY_FAILURE_TEXT,
            "update": _handoff_update(
                profile,
                "Контакт сохранён; передача временно недоступна.",
                DELIVERY_FAILURE_TEXT,
                complete=False,
            ),
        }, event="sender_unavailable", delivery_status="not_attempted")

    try:
        await consultant_sender(CONSULTANT_CHAT_ID, state.consultant_summary)
    except Exception as error:
        return finish({
            "reply": DELIVERY_FAILURE_TEXT,
            "update": _handoff_update(
                profile,
                "Контакт сохранён; передача завершилась технической ошибкой.",
                DELIVERY_FAILURE_TEXT,
                complete=False,
            ),
        }, event="consultant_delivery_failed", delivery_status="failed", error={
            "type": type(error).__name__, "message": str(error)
        })

    set_telegram_consultant_handoff(
        user_id,
        TelegramConsultantHandoffState(
            status=TelegramHandoffStatus.COMPLETED,
            raw_phone=state.raw_phone,
            normalized_phone=state.normalized_phone,
            consultant_summary=state.consultant_summary,
        ),
    )
    return finish({
        "reply": FINAL_USER_TEXT,
        "update": _handoff_update(
            profile,
            "Контакт передан консультанту.",
            FINAL_USER_TEXT,
            complete=True,
        ),
        "remove_contact_keyboard": True,
    }, event="consultant_delivery_completed", delivery_status="delivered")


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
    turn_started_at = monotonic()
    diagnostic_base = {
        "conversation_id": conversation_identifier(user_id),
        "turn_id": user_message.id,
        "feature_flags": {
            "cognitive_core_diagnostics": diagnostics_enabled(),
        },
        "user_message": user_message.model_dump(mode="json"),
        "state_before": _memory_summary(discovery_memory),
        "discovery_memory_before": _memory_view(discovery_memory),
        "trace": [
            {
                "stage": "telegram_handler_to_dialog_engine",
                "status": "received",
            },
            {
                "stage": "discovery_memory_load_and_validation",
                "status": "passed",
            },
        ],
    }
    try:
        turn_result = await asyncio.to_thread(
            core.propose,
            user_message,
            discovery_memory.human_model,
            discovery_memory.active_conversation_state,
            discovery_memory.dialogue_history,
        )
    except Exception as error:
        write_turn_diagnostic(
            {
                **diagnostic_base,
                "status": "cognitive_core_failed",
                "cognitive_core": getattr(core, "last_diagnostics", {}),
                "error": {"type": type(error).__name__, "message": str(error)},
                "total_latency_ms": round((monotonic() - turn_started_at) * 1000, 3),
            }
        )
        raise
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
    state_application_started_at = monotonic()
    try:
        applied_turn = apply_cognitive_turn(
            discovery_memory.human_model,
            discovery_memory.active_conversation_state,
            discovery_memory.dialogue_history,
            user_message,
            system_message,
            turn_result,
        )
    except Exception as error:
        write_turn_diagnostic(
            {
                **diagnostic_base,
                "status": "state_application_failed",
                "cognitive_core": getattr(core, "last_diagnostics", {}),
                **_turn_fields(turn_result),
                "actual_reply": reply,
                "error": {"type": type(error).__name__, "message": str(error)},
                "validation": {
                    "semantic_turn": "passed",
                    "adapter": "passed",
                    "cognitive_turn_result": "passed",
                    "state_applier": "failed",
                    "persistence": "not_started",
                },
                "total_latency_ms": round((monotonic() - turn_started_at) * 1000, 3),
            }
        )
        raise
    state_application_latency_ms = round(
        (monotonic() - state_application_started_at) * 1000, 3
    )

    updated_memory = ShadowDiscoveryMemory(
        human_model=applied_turn.updated_human_model,
        active_conversation_state=applied_turn.new_acs,
        dialogue_history=applied_turn.updated_history,
    )

    persistence_latency_ms = None
    if user_id is not None:
        persistence_started_at = monotonic()
        try:
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
            persisted_memory = _load_discovery_memory(user_id)
            validate_discovery_memory(
                persisted_memory.human_model,
                persisted_memory.active_conversation_state,
                persisted_memory.dialogue_history,
            )
            persistence_latency_ms = round(
                (monotonic() - persistence_started_at) * 1000, 3
            )
        except Exception as error:
            try:
                observed_memory = _memory_view(_load_discovery_memory(user_id))
            except Exception:
                observed_memory = None
            write_turn_diagnostic(
                {
                    **diagnostic_base,
                    "status": "persistence_failed",
                    "cognitive_core": getattr(core, "last_diagnostics", {}),
                    **_turn_fields(turn_result),
                    "actual_reply": reply,
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "validation": {
                        "semantic_turn": "passed",
                        "adapter": "passed",
                        "cognitive_turn_result": "passed",
                        "state_applier": "passed",
                        "persistence": "failed",
                    },
                    "discovery_memory_after_application": _memory_view(updated_memory),
                    "state_diff_after_application": _state_diff(
                        discovery_memory, updated_memory
                    ),
                    "persisted_memory_observed_after_failure": observed_memory,
                    "total_latency_ms": round((monotonic() - turn_started_at) * 1000, 3),
                }
            )
            raise

    if diagnostics_enabled():
        write_turn_diagnostic(
            {
                **diagnostic_base,
                "status": "completed",
                "cognitive_core": getattr(core, "last_diagnostics", {}),
                **_turn_fields(turn_result),
                "actual_reply": reply,
                "system_message": system_message.model_dump(mode="json"),
                "reply_matches_persisted_system_text": system_message.text == reply,
                "applied_state_patch": turn_result.state_patch.model_dump(mode="json"),
                "applied_turn_mappings": {
                    "item_ids": applied_turn.item_id_mapping,
                    "relation_ids": applied_turn.relation_id_mapping,
                    "active_content_ids": applied_turn.active_content_id_mapping,
                    "response_target_ids": applied_turn.response_target_id_mapping,
                },
                "state_after": _memory_summary(updated_memory),
                "discovery_memory_after": _memory_view(updated_memory),
                "state_diff": _state_diff(discovery_memory, updated_memory),
                "handoff": {
                    "consented_transition_detected": starts_contact_collection,
                    "contact_collection_started": starts_contact_collection,
                },
                "trace": [
                    *diagnostic_base["trace"],
                    {
                        "stage": "context_assembly_retrieval_and_cognitive_core",
                        "status": "passed",
                        "details": core.last_diagnostics.get("stage_latency_ms", {}),
                    },
                    {
                        "stage": "semantic_turn_validation_and_adapter",
                        "status": "passed",
                    },
                    {
                        "stage": "state_applier",
                        "status": "passed",
                        "latency_ms": state_application_latency_ms,
                    },
                    {
                        "stage": "persistence",
                        "status": "passed" if user_id is not None else "not_requested",
                        "latency_ms": persistence_latency_ms,
                    },
                    {"stage": "dialog_engine_reply", "status": "returned"},
                ],
                "validation": {
                    "semantic_turn": "passed",
                    "adapter": "passed",
                    "cognitive_turn_result": "passed",
                    "state_applier": "passed",
                    "persistence": "passed" if user_id is not None else "not_requested",
                },
                "total_latency_ms": round((monotonic() - turn_started_at) * 1000, 3),
            }
        )

    result = {
        "reply": reply,
        "update": _compatibility_update(profile, user_text, reply),
    }
    if starts_contact_collection:
        result["request_contact"] = True
    return result
