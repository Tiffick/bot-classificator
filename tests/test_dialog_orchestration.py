import asyncio

import pytest

from ai.cognitive_core import CognitiveCoreError
from ai.cognitive_turn import (
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    DecisionIntent,
    EvidenceOrigin,
    ItemOperation,
    ItemOperationKind,
    ProposalMaterialization,
    ReconciliationOutcome,
    ReplySegment,
    SourceSpan,
    StatePatch,
    SystemAction,
    TargetResolution,
)
from ai.discovery_data_model import (
    ActiveContentKind,
    ModelItemKind,
    TargetInteractionKind,
    TargetSubjectKind,
    validate_discovery_memory,
)
from ai.discovery_state_applier import DiscoveryStateApplyError
from memory.user_memory import (
    get_shadow_discovery_memory,
    get_user_memory,
    reset_user_profile,
    user_profiles,
)


def _question_action(
    *,
    kind=ActiveContentKind.SYSTEM_QUESTION,
    interaction=TargetInteractionKind.OPEN_RESPONSE,
    semantic_content="уточнить пользовательский материал",
):
    content = ActionContent(
        local_id="content:primary",
        kind=kind,
        semantic_content=semantic_content,
    )
    target = ActionTarget(
        local_id="target:primary",
        active_content_local_id=content.local_id,
        subject=ActionSubjectReference(
            kind=TargetSubjectKind.ACTIVE_CONTENT,
            active_content_local_id=content.local_id,
        ),
        interaction=interaction,
    )
    return SystemAction(contents=(content,), response_targets=(target,))


def _result(
    reply,
    *,
    intent=DecisionIntent.HUMAN_DISCOVERY,
    action=None,
    patch=None,
    reconciliation=(),
):
    action = action if action is not None else _question_action()
    return CognitiveTurnResult(
        decision_intent=intent,
        state_patch=patch or StatePatch(),
        reconciliation=tuple(reconciliation),
        system_action=action,
        reply_segments=(
            ReplySegment(
                local_id="segment:reply",
                text=reply,
                realizes_action_content_ids=tuple(
                    content.local_id for content in action.contents
                ),
            ),
        ),
    )


def _install_core(monkeypatch, results):
    import ai.dialog_engine as dialog_engine

    class FakeCore:
        calls = []

        def __init__(self, client):
            self.client = client

        def propose(self, current, human_model, previous_acs, history):
            self.calls.append((current, human_model, previous_acs, history))
            result = results[len(self.calls) - 1]
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(dialog_engine, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(dialog_engine, "CognitiveCore", FakeCore)
    return dialog_engine, FakeCore


def test_dialog_engine_runs_one_cognitive_call_and_no_legacy_chain(monkeypatch):
    dialog_engine, core = _install_core(monkeypatch, [_result("Один ответ.")])

    result = asyncio.run(dialog_engine.run_dialog_engine("Один вход.", {}))

    assert len(core.calls) == 1
    assert result["reply"] == "Один ответ."
    for legacy_name in (
        "ShadowPerceptionEngine",
        "ShadowIntegrationEngine",
        "LLMSemanticEngine",
        "SemanticEngine",
        "HumanModelEngine",
        "ReasoningEngine",
        "DecisionEngine",
        "ImpactEngine",
        "EmotionalEngine",
        "ResponseEngine",
    ):
        assert not hasattr(dialog_engine, legacy_name)


def test_valid_turn_is_applied_persisted_and_returned_byte_for_byte(monkeypatch):
    user_id = 501
    reset_user_profile(user_id)
    user_text = "Мне трудно удерживать изменения"
    reply = "Понимаю. Что обычно меняется перед возвратом старого режима?"
    patch = StatePatch(
        item_operations=(
            ItemOperation(
                operation=ItemOperationKind.ADD,
                local_id="item:difficulty",
                kind=ModelItemKind.EXPERIENCE,
                content=user_text,
                evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                source_span=SourceSpan(char_start=0, char_end=len(user_text)),
            ),
        )
    )
    turn_result = _result(reply, patch=patch)
    dialog_engine, _ = _install_core(
        monkeypatch,
        [turn_result],
    )

    result = asyncio.run(dialog_engine.run_dialog_engine(user_text, {}, user_id))
    memory = get_shadow_discovery_memory(user_id)

    assert result["reply"] == reply
    assert memory.dialogue_history.messages[-1].text == reply
    assert "".join(segment.text for segment in turn_result.reply_segments) == reply
    assert len(memory.human_model.items) == 1
    assert memory.active_conversation_state is not None
    validate_discovery_memory(
        memory.human_model,
        memory.active_conversation_state,
        memory.dialogue_history,
    )


def test_second_turn_receives_first_turn_model_acs_and_history(monkeypatch):
    user_id = 502
    reset_user_profile(user_id)
    first_text = "Начать могу, удержать трудно"
    first_patch = StatePatch(
        item_operations=(
            ItemOperation(
                operation=ItemOperationKind.ADD,
                local_id="item:pattern",
                kind=ModelItemKind.EXPERIENCE,
                content=first_text,
                evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
                source_span=SourceSpan(char_start=0, char_end=len(first_text)),
            ),
        )
    )
    dialog_engine, core = _install_core(
        monkeypatch,
        [
            _result("Что помогает удерживаться?", patch=first_patch),
            _result("Что происходит через несколько недель?"),
        ],
    )

    asyncio.run(dialog_engine.run_dialog_engine(first_text, {}, user_id))
    first_memory = get_shadow_discovery_memory(user_id)
    asyncio.run(dialog_engine.run_dialog_engine("Постепенно сдаюсь", {}, user_id))

    _, received_model, received_acs, received_history = core.calls[1]
    assert received_model == first_memory.human_model
    assert received_acs == first_memory.active_conversation_state
    assert received_history == first_memory.dialogue_history
    assert len(received_history.messages) == 2


def test_reconciliation_materializes_previous_response_target(monkeypatch):
    user_id = 503
    reset_user_profile(user_id)
    first_action = _question_action(
        kind=ActiveContentKind.RECOGNITION_OPTION,
        interaction=TargetInteractionKind.EVALUATION,
        semantic_content="удержание, а не старт, является основной трудностью",
    )

    class ReconcilingCore:
        calls = []

        def __init__(self, client):
            pass

        def propose(self, current, human_model, previous_acs, history):
            self.calls.append((current, human_model, previous_acs, history))
            if previous_acs is None:
                return _result(
                    "Похоже, основная трудность — удержание. Это так?",
                    intent=DecisionIntent.RECOGNITION,
                    action=first_action,
                )
            target = next(iter(previous_acs.response_targets.values()))
            content = previous_acs.active_content[target.active_content_id]
            resolution = TargetResolution(
                previous_response_target_ids=(target.id,),
                outcome=ReconciliationOutcome.SUPPORTED,
                source_span=SourceSpan(char_start=0, char_end=len(current.text)),
            )
            patch = StatePatch(
                proposal_materializations=(
                    ProposalMaterialization(
                        resolution_target_id=target.id,
                        previous_active_content_id=content.id,
                        subject_kind="item",
                        kind=ModelItemKind.EXPERIENCE,
                        content=content.content,
                    ),
                )
            )
            return _result(
                "Спасибо, это проясняет картину.",
                intent=DecisionIntent.STOP_EXPLORATION,
                action=SystemAction(),
                patch=patch,
                reconciliation=(resolution,),
            )

    import ai.dialog_engine as dialog_engine

    monkeypatch.setattr(dialog_engine, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(dialog_engine, "CognitiveCore", ReconcilingCore)
    asyncio.run(dialog_engine.run_dialog_engine("Начать могу", {}, user_id))
    asyncio.run(dialog_engine.run_dialog_engine("Да", {}, user_id))
    memory = get_shadow_discovery_memory(user_id)

    assert len(memory.human_model.items) == 1
    assert next(iter(memory.human_model.items.values())).content == (
        "удержание, а не старт, является основной трудностью"
    )
    assert memory.active_conversation_state is None
    assert [message.text for message in memory.dialogue_history.messages][-2:] == [
        "Да",
        "Спасибо, это проясняет картину.",
    ]


def test_cognitive_failure_preserves_persistent_memory(monkeypatch):
    user_id = 504
    reset_user_profile(user_id)
    before = get_user_memory(user_id)["shadow_discovery_memory"]
    error = CognitiveCoreError("api_failure", "failed")
    dialog_engine, core = _install_core(monkeypatch, [error])

    with pytest.raises(CognitiveCoreError):
        asyncio.run(dialog_engine.run_dialog_engine("Новый ход", {}, user_id))

    assert len(core.calls) == 1
    assert get_user_memory(user_id)["shadow_discovery_memory"] is before


def test_state_applier_failure_does_not_partially_persist(monkeypatch):
    user_id = 505
    reset_user_profile(user_id)
    dialog_engine, core = _install_core(monkeypatch, [_result("Ответ")])
    before = get_user_memory(user_id)["shadow_discovery_memory"]

    def fail_apply(*args):
        raise DiscoveryStateApplyError("invalid transition")

    monkeypatch.setattr(dialog_engine, "apply_cognitive_turn", fail_apply)
    with pytest.raises(DiscoveryStateApplyError, match="invalid transition"):
        asyncio.run(dialog_engine.run_dialog_engine("Новый ход", {}, user_id))

    assert len(core.calls) == 1
    assert get_user_memory(user_id)["shadow_discovery_memory"] is before


def test_user_id_none_is_stateless_and_preserves_return_shape(monkeypatch):
    known_users = set(user_profiles)
    dialog_engine, core = _install_core(monkeypatch, [_result("Ответ без сохранения")])

    result = asyncio.run(
        dialog_engine.run_dialog_engine(
            "Временный ход",
            {"history": [], "discovery_complete": False},
        )
    )

    assert set(result) == {"reply", "update"}
    assert result["reply"] == "Ответ без сохранения"
    assert result["update"]["discovery_complete"] is False
    assert len(result["update"]["history"]) == 2
    assert len(core.calls) == 1
    assert set(user_profiles) == known_users


def test_compatibility_update_does_not_semantically_map_discovery_state(monkeypatch):
    dialog_engine, _ = _install_core(monkeypatch, [_result("Ответ")])

    result = asyncio.run(
        dialog_engine.run_dialog_engine(
            "Мне 30 лет",
            {"age": None, "history": [], "discovery_complete": False},
        )
    )

    assert set(result["update"]) == {"history", "discovery_complete"}
    assert "age" not in result["update"]
