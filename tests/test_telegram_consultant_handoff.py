import asyncio
from types import SimpleNamespace

import pytest

from ai.cognitive_turn import (
    ActionContent, CognitiveTurnResult, DecisionIntent, ReconciliationOutcome,
    ReplySegment, SystemAction, TargetResolution,
)
from ai.discovery_data_model import (
    ActiveContentItem, ActiveContentKind, ActiveConversationState,
    DialogueHistory, DialogueMessage, DialogueRole, HumanModel, ItemStatus,
    ModelItem, ModelItemKind, Provenance, ResponseTarget, SourceReference,
    TargetInteractionKind, TargetReference, TargetSubjectKind,
)
from ai.telegram_consultant_handoff import (
    CONSULTANT_CHAT_ID, CONTACT_BUTTON_TEXT, CONTACT_REQUEST_TEXT, DELIVERY_FAILURE_TEXT,
    FINAL_USER_TEXT, INVALID_CONTACT_TEXT, TelegramConsultantHandoffState,
    TelegramHandoffStatus, normalize_phone,
)
from memory.user_memory import (
    ShadowDiscoveryMemory, get_telegram_consultant_handoff, get_user_profile,
    reset_user_profile, set_shadow_discovery_memory,
    set_telegram_consultant_handoff,
)


def _discovery_memory(*, transition=False):
    user = DialogueMessage(
        role=DialogueRole.USER,
        text="Вес возвращается после напряжённых периодов.",
        sequence=1,
    )
    system = DialogueMessage(
        role=DialogueRole.SYSTEM,
        text="Хотите перейти к консультанту?" if transition else "Понимаю.",
        sequence=2,
    )
    problem = ModelItem(
        kind=ModelItemKind.STATED_PROBLEM,
        content="Вес возвращается после напряжённых периодов.",
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.RECORDED,
        source_refs=(SourceReference(message_id=user.id, char_start=0, char_end=43),),
        created_turn=1,
        updated_turn=1,
    )
    barrier = ModelItem(
        kind=ModelItemKind.BARRIER,
        content="После первого срыва пользователь прекращает попытку.",
        provenance=Provenance.USER_INTERPRETATION,
        status=ItemStatus.RECORDED,
        source_refs=(SourceReference(message_id=user.id),),
        created_turn=1,
        updated_turn=1,
    )
    rejected = ModelItem(
        kind=ModelItemKind.BARRIER,
        content="Причина — отсутствие силы воли.",
        provenance=Provenance.SYSTEM_PROPOSED,
        status=ItemStatus.REJECTED,
        source_refs=(SourceReference(message_id=system.id),),
        created_turn=1,
        updated_turn=1,
    )
    human_model = HumanModel(items={item.id: item for item in (problem, barrier, rejected)})
    acs = None
    target = None
    if transition:
        content = ActiveContentItem(
            kind=ActiveContentKind.SYSTEM_TRANSITION,
            content="Предложен переход к живому консультанту.",
            model_item_ids=(problem.id, barrier.id),
            source_message_id=system.id,
        )
        target = ResponseTarget(
            active_content_id=content.id,
            subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=content.id),
            interaction=TargetInteractionKind.CONSENT,
        )
        acs = ActiveConversationState(
            created_turn=1,
            source_system_message_id=system.id,
            active_content={content.id: content},
            response_targets={target.id: target},
        )
    return ShadowDiscoveryMemory(
        human_model=human_model,
        active_conversation_state=acs,
        dialogue_history=DialogueHistory(messages=(user, system)),
    ), target


def _consented_result(target):
    content = ActionContent(
        local_id="content:fake_transfer",
        kind=ActiveContentKind.SYSTEM_TRANSITION,
        semantic_content="Утверждает, что передача уже выполнена.",
    )
    return CognitiveTurnResult(
        decision_intent=DecisionIntent.TRANSITION,
        reconciliation=(TargetResolution(
            previous_response_target_ids=(target.id,),
            outcome=ReconciliationOutcome.CONSENTED,
        ),),
        system_action=SystemAction(contents=(content,)),
        reply_segments=(ReplySegment(
            local_id="segment:fake_transfer",
            text="Уже связал вас с консультантом.",
            realizes_action_content_ids=(content.local_id,),
        ),),
    )


def _install_no_core(monkeypatch, dialog_engine):
    class ForbiddenCore:
        def __init__(self, *args, **kwargs):
            raise AssertionError("contact collection must not construct CognitiveCore")

    monkeypatch.setattr(dialog_engine, "CognitiveCore", ForbiddenCore)
    monkeypatch.setattr(
        dialog_engine, "OpenAI",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("contact collection must not construct OpenAI")
        ),
    )


def test_consented_transition_enters_deterministic_contact_collection(monkeypatch):
    import ai.dialog_engine as dialog_engine

    user_id = 8101
    reset_user_profile(user_id)
    memory, target = _discovery_memory(transition=True)
    set_shadow_discovery_memory(user_id, memory)

    class FakeCore:
        calls = 0

        def __init__(self, client):
            pass

        def propose(self, *args):
            self.__class__.calls += 1
            return _consented_result(target)

    monkeypatch.setattr(dialog_engine, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(dialog_engine, "CognitiveCore", FakeCore)
    result = asyncio.run(
        dialog_engine.run_dialog_engine("Да, согласен.", get_user_profile(user_id), user_id)
    )

    assert FakeCore.calls == 1
    assert result["reply"] == CONTACT_REQUEST_TEXT
    assert result["request_contact"] is True
    assert "связал" not in result["reply"]
    assert "передал консультанту" not in result["reply"]
    assert result["reply"] != FINAL_USER_TEXT
    assert get_telegram_consultant_handoff(user_id).status == TelegramHandoffStatus.AWAITING_CONTACT
    persisted = dialog_engine._load_discovery_memory(user_id)
    assert persisted.active_conversation_state is None
    assert persisted.dialogue_history.messages[-1].text == CONTACT_REQUEST_TEXT
    assert all(
        message.text != "Уже связал вас с консультантом."
        for message in persisted.dialogue_history.messages
    )


def test_invalid_phone_stays_pending_without_core_or_delivery(monkeypatch):
    import ai.dialog_engine as dialog_engine

    user_id = 8102
    reset_user_profile(user_id)
    memory, _ = _discovery_memory()
    set_shadow_discovery_memory(user_id, memory)
    set_telegram_consultant_handoff(
        user_id, TelegramConsultantHandoffState(TelegramHandoffStatus.AWAITING_CONTACT)
    )
    _install_no_core(monkeypatch, dialog_engine)
    deliveries = []

    async def sender(*args):
        deliveries.append(args)

    result = asyncio.run(dialog_engine.run_dialog_engine(
        "позвоните вечером", get_user_profile(user_id), user_id,
        consultant_sender=sender,
    ))
    assert result["reply"] == INVALID_CONTACT_TEXT
    assert result["request_contact"] is True
    assert deliveries == []
    assert get_telegram_consultant_handoff(user_id).status == TelegramHandoffStatus.AWAITING_CONTACT


@pytest.mark.parametrize(
    ("user_text", "contact_phone", "expected_phone"),
    [("+7 (777) 123-45-67", None, "+77771234567"),
     ("", "+44 20 7946 0958", "+442079460958")],
    ids=("manual", "native-contact"),
)
def test_valid_phone_delivers_grounded_summary_and_completes(
    monkeypatch, user_text, contact_phone, expected_phone
):
    import ai.dialog_engine as dialog_engine

    user_id = 8103 if contact_phone is None else 8104
    reset_user_profile(user_id)
    memory, _ = _discovery_memory()
    set_shadow_discovery_memory(user_id, memory)
    set_telegram_consultant_handoff(
        user_id, TelegramConsultantHandoffState(TelegramHandoffStatus.AWAITING_CONTACT)
    )
    _install_no_core(monkeypatch, dialog_engine)
    deliveries = []

    async def sender(chat_id, text):
        deliveries.append((chat_id, text))

    result = asyncio.run(dialog_engine.run_dialog_engine(
        user_text, get_user_profile(user_id), user_id,
        contact_phone=contact_phone, consultant_sender=sender,
    ))
    assert result["reply"] == FINAL_USER_TEXT
    assert result["remove_contact_keyboard"] is True
    assert result["update"]["discovery_complete"] is True
    assert len(deliveries) == 1
    chat_id, summary = deliveries[0]
    assert chat_id == CONSULTANT_CHAT_ID
    assert expected_phone in summary
    assert "Вес возвращается после напряжённых периодов." in summary
    assert "После первого срыва пользователь прекращает попытку." in summary
    assert "Причина — отсутствие силы воли. [отвергнуто]" in summary
    state = get_telegram_consultant_handoff(user_id)
    assert state.status == TelegramHandoffStatus.COMPLETED
    assert state.normalized_phone == expected_phone


def test_failed_delivery_keeps_pending_payload_for_one_safe_retry(monkeypatch):
    import ai.dialog_engine as dialog_engine

    user_id = 8105
    reset_user_profile(user_id)
    memory, _ = _discovery_memory()
    set_shadow_discovery_memory(user_id, memory)
    set_telegram_consultant_handoff(
        user_id, TelegramConsultantHandoffState(TelegramHandoffStatus.AWAITING_CONTACT)
    )
    _install_no_core(monkeypatch, dialog_engine)

    async def failing_sender(chat_id, text):
        raise RuntimeError("Telegram unavailable")

    failed = asyncio.run(dialog_engine.run_dialog_engine(
        "+7 777 123 45 67", get_user_profile(user_id), user_id,
        consultant_sender=failing_sender,
    ))
    pending = get_telegram_consultant_handoff(user_id)
    assert failed["reply"] == DELIVERY_FAILURE_TEXT
    assert FINAL_USER_TEXT not in failed["reply"]
    assert pending.status == TelegramHandoffStatus.DELIVERY_PENDING
    assert pending.consultant_summary

    deliveries = []

    async def working_sender(chat_id, text):
        deliveries.append((chat_id, text))

    succeeded = asyncio.run(dialog_engine.run_dialog_engine(
        "повторить", get_user_profile(user_id), user_id,
        consultant_sender=working_sender,
    ))
    assert succeeded["reply"] == FINAL_USER_TEXT
    assert deliveries == [(CONSULTANT_CHAT_ID, pending.consultant_summary)]
    assert get_telegram_consultant_handoff(user_id).status == TelegramHandoffStatus.COMPLETED


def test_completed_handoff_never_redelivers(monkeypatch):
    import ai.dialog_engine as dialog_engine

    user_id = 8106
    reset_user_profile(user_id)
    memory, _ = _discovery_memory()
    set_shadow_discovery_memory(user_id, memory)
    set_telegram_consultant_handoff(user_id, TelegramConsultantHandoffState(
        TelegramHandoffStatus.COMPLETED,
        raw_phone="+77771234567", normalized_phone="+77771234567",
        consultant_summary="saved",
    ))
    _install_no_core(monkeypatch, dialog_engine)
    deliveries = []

    async def sender(*args):
        deliveries.append(args)

    result = asyncio.run(dialog_engine.run_dialog_engine(
        "ещё сообщение", get_user_profile(user_id), user_id,
        consultant_sender=sender,
    ))
    assert result["reply"] == FINAL_USER_TEXT
    assert deliveries == []


def test_phone_normalization_is_basic_and_not_country_specific():
    assert normalize_phone("+7 (777) 123-45-67") == "+77771234567"
    assert normalize_phone("87771234567") == "87771234567"
    assert normalize_phone("123") is None
    assert normalize_phone("+7 call me") is None


def test_native_telegram_contact_is_forwarded_to_runtime(monkeypatch):
    import bot.handlers as handlers

    reset_user_profile(8110)
    set_telegram_consultant_handoff(
        8110,
        TelegramConsultantHandoffState(TelegramHandoffStatus.AWAITING_CONTACT),
    )
    captured = {}

    async def fake_engine(user_text, profile, user_id, **kwargs):
        captured.update(user_text=user_text, user_id=user_id, **kwargs)
        return {"reply": "Готово", "update": {}, "remove_contact_keyboard": True}

    class FakeBot:
        async def send_message(self, **kwargs):
            captured["delivery"] = kwargs

    class FakeMessage:
        text = None
        contact = SimpleNamespace(phone_number="+7 777 123 45 67")
        from_user = SimpleNamespace(id=8110)
        bot = FakeBot()

        def __init__(self):
            self.answers = []

        async def answer(self, text, **kwargs):
            self.answers.append((text, kwargs))

    monkeypatch.setattr(handlers, "run_dialog_engine", fake_engine)
    monkeypatch.setattr(handlers, "get_user_profile", lambda user_id: {})
    monkeypatch.setattr(handlers, "update_user_profile", lambda user_id, data: {})
    monkeypatch.setattr(handlers, "set_last_question", lambda *args: None)
    message = FakeMessage()

    asyncio.run(handlers.message_handler(message))

    assert captured["user_text"] == ""
    assert captured["contact_phone"] == "+7 777 123 45 67"
    assert captured["user_id"] == 8110
    assert message.answers[0][0] == "Готово"


def test_post_consent_message_has_native_contact_button(monkeypatch):
    import bot.handlers as handlers

    async def fake_engine(user_text, profile, user_id, **kwargs):
        return {
            "reply": CONTACT_REQUEST_TEXT,
            "update": {},
            "request_contact": True,
        }

    class FakeBot:
        id = 1

        async def send_chat_action(self, **kwargs):
            pass

    class FakeMessage:
        text = "Да, согласен."
        contact = None
        from_user = SimpleNamespace(id=8111)
        chat = SimpleNamespace(id=8111)
        bot = FakeBot()

        def __init__(self):
            self.answers = []

        async def answer(self, text, **kwargs):
            self.answers.append((text, kwargs))

    monkeypatch.setattr(handlers, "run_dialog_engine", fake_engine)
    monkeypatch.setattr(handlers, "get_user_profile", lambda user_id: {})
    monkeypatch.setattr(handlers, "update_user_profile", lambda user_id, data: {})
    monkeypatch.setattr(handlers, "set_last_question", lambda *args: None)
    message = FakeMessage()

    asyncio.run(handlers.message_handler(message))

    reply, kwargs = message.answers[0]
    button = kwargs["reply_markup"].keyboard[0][0]
    assert reply == CONTACT_REQUEST_TEXT
    assert button.text == CONTACT_BUTTON_TEXT
    assert button.request_contact is True


def test_typing_refreshes_during_slow_cognitive_processing_without_extra_delay(
    monkeypatch,
):
    import bot.handlers as handlers

    user_id = 8112
    reset_user_profile(user_id)
    actions = []

    async def fake_engine(user_text, profile, user_id, **kwargs):
        await asyncio.sleep(0.045)
        return {"reply": "Готово", "update": {}}

    class FakeBot:
        id = 1

        async def send_chat_action(self, **kwargs):
            actions.append(kwargs)

    class FakeMessage:
        text = "Обычная реплика"
        contact = None
        from_user = SimpleNamespace(id=user_id)
        chat = SimpleNamespace(id=user_id)
        bot = FakeBot()

        async def answer(self, text, **kwargs):
            pass

    monkeypatch.setattr(handlers, "run_dialog_engine", fake_engine)
    monkeypatch.setattr(handlers, "get_user_profile", lambda user_id: {})
    monkeypatch.setattr(handlers, "update_user_profile", lambda user_id, data: {})
    monkeypatch.setattr(handlers, "set_last_question", lambda *args: None)
    monkeypatch.setattr(handlers, "TYPING_REFRESH_SECONDS", 0.01)

    async def scenario():
        loop = asyncio.get_running_loop()
        started = loop.time()
        await handlers.message_handler(FakeMessage())
        elapsed = loop.time() - started
        count_after_reply = len(actions)
        await asyncio.sleep(0.03)
        return elapsed, count_after_reply

    elapsed, count_after_reply = asyncio.run(scenario())

    assert count_after_reply >= 2
    assert len(actions) == count_after_reply
    assert elapsed < 0.2


def test_typing_stops_when_cognitive_processing_raises(monkeypatch):
    import bot.handlers as handlers

    user_id = 8113
    reset_user_profile(user_id)
    actions = []

    async def failing_engine(user_text, profile, user_id, **kwargs):
        await asyncio.sleep(0.025)
        raise RuntimeError("cognitive failure")

    class FakeBot:
        id = 1

        async def send_chat_action(self, **kwargs):
            actions.append(kwargs)

    class FakeMessage:
        text = "Обычная реплика"
        contact = None
        from_user = SimpleNamespace(id=user_id)
        chat = SimpleNamespace(id=user_id)
        bot = FakeBot()

        async def answer(self, text, **kwargs):
            raise AssertionError("failure path must not answer")

    monkeypatch.setattr(handlers, "run_dialog_engine", failing_engine)
    monkeypatch.setattr(handlers, "get_user_profile", lambda user_id: {})
    monkeypatch.setattr(handlers, "TYPING_REFRESH_SECONDS", 0.01)

    async def scenario():
        with pytest.raises(RuntimeError, match="cognitive failure"):
            await handlers.message_handler(FakeMessage())
        count_after_failure = len(actions)
        await asyncio.sleep(0.03)
        return count_after_failure

    count_after_failure = asyncio.run(scenario())

    assert count_after_failure >= 1
    assert len(actions) == count_after_failure
