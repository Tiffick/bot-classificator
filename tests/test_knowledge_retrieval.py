import asyncio
import inspect
import json

import pytest

import ai.knowledge_retrieval as retrieval
from ai.discovery_data_model import (
    ActiveContentItem,
    ActiveContentKind,
    ActiveConversationState,
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    ItemStatus,
    ModelItem,
    ModelItemKind,
    Provenance,
    ResponseTarget,
    SourceReference,
    TargetInteractionKind,
    TargetReference,
    TargetSubjectKind,
    validate_discovery_memory,
)
from ai.knowledge_retrieval import (
    EMBEDDING_MODEL_ID,
    KNOWLEDGE_RETRIEVAL_FLAG,
    KnowledgeRetrievalError,
    SemanticRetrievalRuntime,
    build_retrieval_context,
    knowledge_retrieval_enabled,
    load_knowledge_catalog,
    load_semantic_index,
    prepare_semantic_index,
    retrieve_professional_context,
)
from memory.user_memory import get_shadow_discovery_memory, reset_user_profile
from test_cognitive_core import FakeClient, _semantic_baseline


class MeaningBackend:
    """Small deterministic embedding substitute; production never uses it."""

    model_id = EMBEDDING_MODEL_ID

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        value = text.casefold()
        if any(part in value for part in ("вызываю рвоту", "очищаюсь", "слабительные", "голодаю в компенсацию")):
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        if any(part in value for part in ("резко набрал", "резко похудел", "новое лекарство", "гормональн")):
            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if any(part in value for part in ("причинить себе вред", "не хочу жить", "покончить с собой")):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if any(part in value for part in ("пропуск", "физическ", "желание поесть", "не могу остановиться", "теряю контроль")):
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if any(part in value for part in ("всё испорт", "полный провал", "бросаю на недели")):
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if any(part in value for part in ("быстр", "медленно", "скорого результата")):
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if "обычный разговор" in value:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


@pytest.fixture
def semantic_runtime(tmp_path):
    backend = MeaningBackend()
    path = prepare_semantic_index(destination=tmp_path / "index.json", backend=backend)
    return SemanticRetrievalRuntime(backend, load_semantic_index(path))


def _message(role, text, sequence):
    return DialogueMessage(role=role, text=text, sequence=sequence)


def _short_answer_acs():
    system = _message(
        DialogueRole.SYSTEM,
        "Сильное желание поесть возникло после пропуска обеда или после разговора?",
        2,
    )
    content = ActiveContentItem(
        kind=ActiveContentKind.SYSTEM_QUESTION,
        content=system.text,
        source_message_id=system.id,
    )
    target = ResponseTarget(
        active_content_id=content.id,
        subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=content.id),
        interaction=TargetInteractionKind.OPEN_RESPONSE,
    )
    return system, ActiveConversationState(
        created_turn=1,
        source_system_message_id=system.id,
        active_content={content.id: content},
        response_targets={target.id: target},
    )


def test_feature_flag_defaults_on_and_explicit_false_is_complete_rollback(monkeypatch, caplog):
    monkeypatch.delenv(KNOWLEDGE_RETRIEVAL_FLAG, raising=False)
    assert knowledge_retrieval_enabled() is True
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "true")
    assert knowledge_retrieval_enabled() is True
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "false")
    assert knowledge_retrieval_enabled() is False
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "invalid")
    assert knowledge_retrieval_enabled() is True
    assert "retrieval remains enabled" in caplog.text


def test_semantic_index_has_multiple_views_and_all_admitted_he(semantic_runtime):
    index = semantic_runtime.index
    assert index.embedding_model_id == EMBEDDING_MODEL_ID
    assert len(index.documents) == 8 * 3 + 3 + 30 * 2
    assert {document.layer for document in index.documents} == {
        "professional_knowledge",
        "human_experience",
        "safety",
    }
    assert len({document.card_id for document in index.documents if document.layer == "human_experience"}) == 30


def test_selection_preserves_complete_semantic_retrieval_trace(semantic_runtime):
    selection = retrieve_professional_context(
        "После пропуска обеда вечером трудно остановиться.",
        HumanModel(),
        None,
        DialogueHistory(),
        runtime=semantic_runtime,
    )

    assert selection.query_views[0][0] == "current_user_message"
    assert selection.query_views[0][1]
    assert selection.query_views[0][2] == 1.0
    assert selection.safety_query_views[0][0] == "current_user_message"
    assert selection.index_schema_version == retrieval.SEMANTIC_INDEX_SCHEMA_VERSION
    assert selection.semantic_view_version == retrieval.SEMANTIC_VIEW_VERSION
    assert len(selection.professional_candidate_scores) == 8
    assert len(selection.human_experience_candidate_scores) == 30
    assert selection.professional_similarity_floor is not None
    assert selection.human_experience_similarity_floor is not None


def test_primary_mechanism_view_outranks_broader_supporting_signal_view():
    class FixedBackend:
        model_id = EMBEDDING_MODEL_ID

        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    index = retrieval.SemanticIndex(
        schema_version=retrieval.SEMANTIC_INDEX_SCHEMA_VERSION,
        semantic_view_version=retrieval.SEMANTIC_VIEW_VERSION,
        embedding_model_id=EMBEDDING_MODEL_ID,
        embedding_dimension=2,
        corpus_sha256="test",
        documents=(
            retrieval.SemanticIndexDocument(
                document_id="direct::mechanism",
                card_id="direct",
                layer="professional_knowledge",
                view_kind="mechanism",
                vector=(0.85, (1 - 0.85**2) ** 0.5),
            ),
            retrieval.SemanticIndexDocument(
                document_id="broad::supporting_signals",
                card_id="broad",
                layer="professional_knowledge",
                view_kind="supporting_signals",
                vector=(0.86, (1 - 0.86**2) ** 0.5),
            ),
        ),
    )
    scores = SemanticRetrievalRuntime(FixedBackend(), index).score(
        (retrieval.SemanticQueryView("current_user_message", "query", 1.0),)
    )
    assert scores["direct"] > scores["broad"]


def test_different_wording_of_same_meaning_returns_same_mechanism(semantic_runtime):
    first = retrieve_professional_context(
        "После работы не могу остановиться с едой.",
        HumanModel(), None, DialogueHistory(), runtime=semantic_runtime,
    )
    second = retrieve_professional_context(
        "Вечером теряю контроль над едой.",
        HumanModel(), None, DialogueHistory(), runtime=semantic_runtime,
    )
    first_ids = {card.id for card in first.professional_cards}
    second_ids = {card.id for card in second.professional_cards}
    assert first_ids == second_ids
    assert first_ids


def test_production_retrieval_contains_no_lexical_search_implementation():
    source = inspect.getsource(retrieval).casefold()
    for forbidden in ("sqlite", "fts5", "bm25", "re.compile", "finditer(", "synonym"):
        assert forbidden not in source


def test_short_answer_uses_structured_active_context(semantic_runtime):
    system, acs = _short_answer_acs()
    selection = retrieve_professional_context(
        "Да.",
        HumanModel(),
        acs,
        DialogueHistory(messages=(system,)),
        runtime=semantic_runtime,
    )
    assert selection.active_context_used is True
    assert "active_conversation_content" in selection.query_view_kinds
    assert "weight.mechanism.restriction_evening_hunger" in {
        card.id for card in selection.professional_cards
    }


def test_system_authored_context_cannot_create_a_safety_signal(semantic_runtime):
    system = _message(
        DialogueRole.SYSTEM,
        "После нового лекарства вы резко набрали вес?",
        2,
    )
    content = ActiveContentItem(
        kind=ActiveContentKind.SYSTEM_QUESTION,
        content=system.text,
        source_message_id=system.id,
    )
    target = ResponseTarget(
        active_content_id=content.id,
        subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=content.id),
        interaction=TargetInteractionKind.OPEN_RESPONSE,
    )
    acs = ActiveConversationState(
        created_turn=1,
        source_system_message_id=system.id,
        active_content={content.id: content},
        response_targets={target.id: target},
    )

    selection = retrieve_professional_context(
        "Нет, речь только о том, как сидит одежда.",
        HumanModel(),
        acs,
        DialogueHistory(messages=(system,)),
        runtime=semantic_runtime,
    )

    assert selection.safety_cards == ()
    assert "active_conversation_content" in selection.query_view_kinds
    assert {
        kind for kind, _text, _weight in selection.safety_query_views
    } == {"current_user_message"}


def test_rejected_items_are_not_query_evidence_and_negation_stays_user_data(semantic_runtime):
    source = _message(DialogueRole.USER, "Ранее была предложена версия.", 1)
    rejected = ModelItem(
        kind=ModelItemKind.USER_INTERPRETATION,
        content="Один вечер означает полный провал.",
        provenance=Provenance.SYSTEM_PROPOSED,
        status=ItemStatus.REJECTED,
        source_refs=(SourceReference(message_id=source.id),),
        created_turn=1,
        updated_turn=1,
    )
    human_model = HumanModel(items={rejected.id: rejected})
    text = "Нет, один вечер не означает провал, после него я продолжаю."
    context = build_retrieval_context(text, human_model, None, DialogueHistory())
    assert context.rejected_or_corrected_item_count == 1
    assert rejected.content not in {view.text for view in context.query_views}
    selection = retrieve_professional_context(
        text, human_model, None, DialogueHistory(), runtime=semantic_runtime
    )
    assert selection.professional_input_view()["similarity_is_user_evidence"] is False


def test_kb_and_he_are_ranked_separately_and_all_he_are_eligible(semantic_runtime):
    selection = retrieve_professional_context(
        "Я ожидал быстрого результата, но изменения идут слишком медленно.",
        HumanModel(), None, DialogueHistory(), runtime=semantic_runtime,
    )
    assert selection.professional_cards
    assert "P028" in {card.id for card in selection.human_experience_cards}
    assert selection.indexed_human_experience_count == 30
    assert selection.professional_input_view()["retrieval_method"] == "local_dense_semantic_cosine_v1"
    assert selection.human_experience_input_view(enabled=True)["retrieval_method"] == "local_dense_semantic_all_eligible_v1"


def test_he_disable_keeps_professional_semantic_index_available(semantic_runtime):
    selection = retrieve_professional_context(
        "Вечером теряю контроль над едой.",
        HumanModel(), None, DialogueHistory(),
        include_human_experience=False,
        runtime=semantic_runtime,
    )
    assert selection.professional_cards
    assert selection.human_experience_cards == ()
    assert selection.indexed_human_experience_count == 30


def test_safety_lane_abstains_for_ordinary_conversation(semantic_runtime):
    selection = retrieve_professional_context(
        "Обычный разговор о снижении веса.",
        HumanModel(), None, DialogueHistory(),
        knowledge_token_budget=1,
        runtime=semantic_runtime,
    )
    assert selection.professional_cards == ()
    assert selection.safety_cards == ()
    assert len(selection.safety_candidate_scores) == 3

def test_safety_lane_abstains_for_ordinary_emotional_idiom(semantic_runtime):
    selection = retrieve_professional_context(
        "Перестать жить от попытки до попытки и снова нормально чувствовать себя в своём теле.",
        HumanModel(), None, DialogueHistory(),
        runtime=semantic_runtime,
    )

    assert selection.safety_cards == ()


def test_safety_lane_requires_cross_category_semantic_discrimination():
    catalog = load_knowledge_catalog()
    scores = {
        "weight.safety.self_harm": 0.8885,
        "weight.safety.eating_disorder": 0.8840,
        "weight.safety.medical_change": 0.8789,
    }

    selected, margin = retrieval._select_safety_cards(
        catalog.cards, scores, retrieval.SAFETY_MIN_SIMILARITY
    )

    assert selected == ()
    assert margin == pytest.approx(0.0045)


@pytest.mark.parametrize(
    ("message", "expected_id"),
    (
        ("После еды вызываю рвоту и принимаю слабительные.", "weight.safety.eating_disorder"),
        ("После нового лекарства я резко набрал вес.", "weight.safety.medical_change"),
        ("Я не хочу жить и думаю причинить себе вред.", "weight.safety.self_harm"),
    ),
)
def test_safety_lane_semantically_selects_explicit_red_flag(
    semantic_runtime, message, expected_id
):
    selection = retrieve_professional_context(
        message,
        HumanModel(), None, DialogueHistory(),
        runtime=semantic_runtime,
    )
    assert [card.id for card in selection.safety_cards] == [expected_id]
    assert selection.safety_similarity_floor is not None


def test_model_and_index_are_process_singletons(monkeypatch, semantic_runtime):
    calls = {"index": 0, "backend": 0}

    def load_index():
        calls["index"] += 1
        return semantic_runtime.index

    def create_backend(*, allow_download):
        assert allow_download is False
        calls["backend"] += 1
        return semantic_runtime.backend

    monkeypatch.setattr(retrieval, "_RUNTIME", None)
    monkeypatch.setattr(retrieval, "load_semantic_index", load_index)
    monkeypatch.setattr(retrieval, "FastEmbedBackend", create_backend)
    first = retrieval._get_runtime()
    second = retrieval._get_runtime()
    assert first is second
    assert calls == {"index": 1, "backend": 1}


def test_stale_or_missing_index_fails_closed_without_lexical_fallback(tmp_path):
    with pytest.raises(KnowledgeRetrievalError, match="missing or invalid"):
        load_semantic_index(tmp_path / "missing.json")


def test_enabled_vertical_slice_reaches_dialog_engine_with_one_model_call(
    monkeypatch, semantic_runtime
):
    import ai.dialog_engine as dialog_engine

    user_id = 880_002
    user_text = "После работы не могу остановиться с едой."
    reset_user_profile(user_id)
    client = FakeClient(json.dumps(_semantic_baseline(), ensure_ascii=False))
    monkeypatch.setenv(KNOWLEDGE_RETRIEVAL_FLAG, "true")
    monkeypatch.setattr(retrieval, "_RUNTIME", semantic_runtime)
    monkeypatch.setattr(dialog_engine, "OpenAI", lambda **kwargs: client)

    result = asyncio.run(dialog_engine.run_dialog_engine(user_text, {}, user_id))

    assert len(client.completions.calls) == 1
    context = json.loads(client.completions.calls[0]["messages"][1]["content"])
    assert context["professional_knowledge"]["selected_cards"]
    assert context["human_experience"]["selected_cards"]
    memory = get_shadow_discovery_memory(user_id)
    assert memory.dialogue_history.messages[-1].text == result["reply"]
    validate_discovery_memory(
        memory.human_model,
        memory.active_conversation_state,
        memory.dialogue_history,
    )
