"""Semantic-only local knowledge retrieval for Cognitive Core.

Similarity yields candidates, never evidence about a user. Model download and
index construction happen only through the explicit preparation command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from ai.discovery_data_model import (
    ActiveConversationState,
    DialogueHistory,
    DialogueRole,
    HumanModel,
    ItemStatus,
    Provenance,
)
from ai.human_experience import HumanExperienceCard, load_human_experience_catalog

_LOGGER = logging.getLogger(__name__)
KNOWLEDGE_RETRIEVAL_FLAG = "KNOWLEDGE_RETRIEVAL_ENABLED"
KNOWLEDGE_RETRIEVAL_INDEX_PATH = "KNOWLEDGE_RETRIEVAL_INDEX_PATH"
KNOWLEDGE_RETRIEVAL_MODEL_CACHE = "KNOWLEDGE_RETRIEVAL_MODEL_CACHE"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
SEMANTIC_INDEX_SCHEMA_VERSION = "knowledge-semantic-index.v1"
SEMANTIC_VIEW_VERSION = "weight-semantic-views.e5-v3"
KNOWLEDGE_CONTEXT_TOKEN_BUDGET = 1_800
HUMAN_EXPERIENCE_TOKEN_BUDGET = 900
# Calibrated on the real E5 index (2026-10-14), including document-view
# weighting below. An unrelated Russian control peaked at 0.7646 for KB and
# 0.7776 for HE, while clear target mechanisms were 0.8002-0.8505. The narrow
# relative window avoids returning the whole high-baseline E5 corpus and does
# not impose a fixed result count.
PROFESSIONAL_MIN_SIMILARITY = 0.80
HUMAN_EXPERIENCE_MIN_SIMILARITY = 0.79
# Calibrated on explicit red-flag and ordinary weight-management controls with
# the real local E5 model. The higher floor prevents general weight language
# from surfacing safety instructions while retaining explicit safety signals.
SAFETY_MIN_SIMILARITY = 0.875
# Dense similarity alone is not enough for safety: ordinary emotional language
# can sit near several unrelated warning cards in multilingual embedding space.
# A safety match must also be semantically discriminative against the next-best
# safety interpretation. This is category competition, not lexical routing.
SAFETY_MIN_DISCRIMINATION_MARGIN = 0.009
SEMANTIC_RELATIVE_BAND = 0.015
# Core mechanism/experience text is the primary retrieval signal. Supporting
# Supporting-signal and inquiry/limits views broaden recall, but may not outrank a
# direct mechanism match merely because they contain generic conversational
# language.
_DOCUMENT_VIEW_WEIGHTS = {
    "mechanism": 1.0,
    "supporting_signals": 0.96,
    "inquiry_limits_and_manifestations": 0.88,
    "lived_experience": 1.0,
    "applicability_alternatives_and_limits": 0.90,
    "safety_signal": 1.0,
}
_RETRIEVABLE_STATUS = "pilot_eligible_not_expert_validated"
_CATALOG_PATH = Path(__file__).parents[1] / "knowledge/weight_management/knowledge_cards_v1.json"
_HE_ADMISSION_PATH = Path(__file__).parents[1] / "knowledge/weight_management/human_experience_admission_v1.json"

PROFESSIONAL_KNOWLEDGE_SYSTEM_INSTRUCTION = """

PROFESSIONAL KNOWLEDGE CONTEXT:
- professional_knowledge contains semantically retrieved candidate mechanisms
  and independently supplied safety boundaries, not facts or diagnoses about
  this person. Similarity is not evidence that a mechanism applies.
- human_experience contains possible real-life manifestations. It has a
  different role from professional knowledge and is never user evidence.
- Use either source only when grounded in the USER material and Human Model.
  You may use none. Respect confirmed, rejected, and corrected user data,
  preserve meaningful alternatives, and make hypotheses easy to reject.
- Safety material is a reason for caution or appropriate referral, not proof
  of a condition. Do not expose internal IDs or retrieval mechanics.
""".rstrip()


class KnowledgeRetrievalError(RuntimeError):
    """Knowledge assets or local semantic retrieval could not be used safely."""


class SafetyMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    red_flag_ids: tuple[str, ...]
    referral_scope: str


class KnowledgeSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    title: str
    url: str
    access: str


class KnowledgeCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    kind: str
    status: str
    title: str
    summary: str
    mechanism: str
    supporting_signals: tuple[str, ...]
    contradicting_signals: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    discriminating_observations: tuple[str, ...]
    prohibited_inferences: tuple[str, ...]
    source_ids: tuple[str, ...]
    related_human_experience_ids: tuple[str, ...]
    safety: SafetyMetadata


class KnowledgeCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    release_id: str
    domain: str
    language: str
    status: str
    cards: tuple[KnowledgeCard, ...]
    sources: dict[str, KnowledgeSource]

    @model_validator(mode="after")
    def validate_catalog(self) -> "KnowledgeCatalog":
        if self.schema_version != "knowledge-catalog.v1":
            raise ValueError("unsupported knowledge catalog schema")
        ids = [card.id for card in self.cards]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("knowledge card IDs must be non-empty and unique")
        for card in self.cards:
            if any(source not in self.sources for source in card.source_ids):
                raise ValueError(f"{card.id} references an unknown source")
            if any(alt not in ids for alt in card.alternative_explanations):
                raise ValueError(f"{card.id} references an unknown alternative")
        return self


class HumanExperienceAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    release_id: str
    source_schema_version: str
    status: str
    basis: str
    eligible_ids: tuple[str, ...]
    excluded: tuple[dict[str, Any], ...]

    @model_validator(mode="after")
    def validate_admission(self) -> "HumanExperienceAdmission":
        if self.schema_version != "human-experience-admission.v1":
            raise ValueError("unsupported Human Experience admission schema")
        if not self.eligible_ids or len(self.eligible_ids) != len(set(self.eligible_ids)):
            raise ValueError("eligible Human Experience IDs must be unique")
        return self


@dataclass(frozen=True)
class SemanticQueryView:
    kind: str
    text: str
    weight: float


@dataclass(frozen=True)
class RetrievalContext:
    query_views: tuple[SemanticQueryView, ...]
    safety_query_views: tuple[SemanticQueryView, ...]
    rejected_or_corrected_item_count: int
    active_context_present: bool


@dataclass(frozen=True)
class SemanticDocument:
    document_id: str
    card_id: str
    layer: str
    view_kind: str
    text: str


class SemanticIndexDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    document_id: str
    card_id: str
    layer: str
    view_kind: str
    vector: tuple[float, ...]


class SemanticIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    semantic_view_version: str
    embedding_model_id: str
    embedding_dimension: int
    corpus_sha256: str
    documents: tuple[SemanticIndexDocument, ...]

    @model_validator(mode="after")
    def validate_index(self) -> "SemanticIndex":
        if self.schema_version != SEMANTIC_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported semantic index schema")
        if self.semantic_view_version != SEMANTIC_VIEW_VERSION:
            raise ValueError("semantic view version mismatch")
        if self.embedding_model_id != EMBEDDING_MODEL_ID:
            raise ValueError("embedding model mismatch")
        ids = [document.document_id for document in self.documents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("semantic index document IDs must be unique")
        for document in self.documents:
            if len(document.vector) != self.embedding_dimension:
                raise ValueError("semantic index vector dimension mismatch")
            norm = math.sqrt(sum(value * value for value in document.vector))
            if not math.isfinite(norm) or not 0.99 <= norm <= 1.01:
                raise ValueError("semantic index vectors must be normalized")
        return self


class EmbeddingBackend(Protocol):
    model_id: str
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedBackend:
    """One local ONNX model instance, reused for the process lifetime."""
    model_id = EMBEDDING_MODEL_ID

    def __init__(self, *, allow_download: bool) -> None:
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType

            if not any(
                model["model"] == self.model_id
                for model in TextEmbedding.list_supported_models()
            ):
                TextEmbedding.add_custom_model(
                    model=self.model_id,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=self.model_id),
                    dim=384,
                    model_file="onnx/model.onnx",
                )
            self._model = TextEmbedding(
                model_name=self.model_id,
                cache_dir=str(_model_cache_path()),
                local_files_only=not allow_download,
                threads=1,
            )
        except ImportError as error:
            raise KnowledgeRetrievalError("fastembed is not installed") from error
        except Exception as error:
            raise KnowledgeRetrievalError(
                "Local embedding model is unavailable; run "
                "'python -m ai.knowledge_retrieval --prepare-index'."
            ) from error

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = [vector.tolist() for vector in self._model.embed(list(texts))]
        except Exception as error:
            raise KnowledgeRetrievalError("Local embedding inference failed") from error
        return [_normalize(vector) for vector in vectors]


@dataclass(frozen=True)
class SemanticRetrievalRuntime:
    backend: EmbeddingBackend
    index: SemanticIndex

    def score_views(
        self, views: Sequence[SemanticQueryView]
    ) -> tuple[dict[str, float], ...]:
        vectors = self.backend.encode([f"query: {view.text}" for view in views])
        if len(vectors) != len(views):
            raise KnowledgeRetrievalError("Unexpected query vector count")
        results: list[dict[str, float]] = []
        for view, vector in zip(views, vectors):
            scores: dict[str, float] = {}
            for document in self.index.documents:
                document_weight = _DOCUMENT_VIEW_WEIGHTS.get(document.view_kind)
                if document_weight is None:
                    raise KnowledgeRetrievalError(
                        f"Unsupported semantic document view: {document.view_kind}"
                    )
                score = _cosine(vector, document.vector) * view.weight * document_weight
                scores[document.card_id] = max(
                    scores.get(document.card_id, -1.0), score
                )
            results.append(scores)
        return tuple(results)

    def score(self, views: Sequence[SemanticQueryView]) -> dict[str, float]:
        return _aggregate_score_maps(self.score_views(views))


@dataclass(frozen=True)
class KnowledgeRetrievalSelection:
    release_id: str
    catalog_status: str
    professional_cards: tuple[KnowledgeCard, ...]
    safety_cards: tuple[KnowledgeCard, ...]
    human_experience_cards: tuple[HumanExperienceCard, ...]
    indexed_knowledge_count: int
    indexed_human_experience_count: int
    estimated_tokens: int
    active_context_used: bool
    embedding_model_id: str = EMBEDDING_MODEL_ID
    index_corpus_sha256: str = ""
    query_view_kinds: tuple[str, ...] = ()
    retrieval_latency_ms: float = 0.0
    professional_scores: tuple[tuple[str, float], ...] = ()
    human_experience_scores: tuple[tuple[str, float], ...] = ()
    query_views: tuple[tuple[str, str, float], ...] = ()
    safety_query_views: tuple[tuple[str, str, float], ...] = ()
    professional_candidate_scores: tuple[tuple[str, float], ...] = ()
    human_experience_candidate_scores: tuple[tuple[str, float], ...] = ()
    safety_candidate_scores: tuple[tuple[str, float], ...] = ()
    professional_similarity_floor: float | None = None
    human_experience_similarity_floor: float | None = None
    safety_similarity_floor: float | None = None
    safety_discrimination_margin: float | None = None
    index_schema_version: str = ""
    semantic_view_version: str = ""

    def professional_input_view(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "release_id": self.release_id,
            "status": self.catalog_status,
            "retrieval_method": "local_dense_semantic_cosine_v1",
            "similarity_is_user_evidence": False,
            "abstained": not self.professional_cards,
            "selected_cards": [_knowledge_card_view(card) for card in self.professional_cards],
            "safety_candidates": [_knowledge_card_view(card) for card in self.safety_cards],
            "diagnostics": {
                "embedding_model": self.embedding_model_id,
                "index_corpus_sha256": self.index_corpus_sha256,
                "indexed_knowledge_count": self.indexed_knowledge_count,
                "estimated_context_tokens": self.estimated_tokens,
                "active_context_used": self.active_context_used,
                "query_view_kinds": list(self.query_view_kinds),
                "retrieval_latency_ms": round(self.retrieval_latency_ms, 3),
            },
        }

    def human_experience_input_view(self, *, enabled: bool) -> dict[str, Any]:
        if not enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "status": "experimental_hypotheses_not_evidence",
            "retrieval_method": "local_dense_semantic_all_eligible_v1",
            "similarity_is_user_evidence": False,
            "abstained": not self.human_experience_cards,
            "abstain_reason": None if self.human_experience_cards else "no_relevant_match",
            "selected_cards": [_human_experience_card_view(card) for card in self.human_experience_cards],
            "diagnostics": {
                "embedding_model": self.embedding_model_id,
                "indexed_human_experience_count": self.indexed_human_experience_count,
            },
        }


_RUNTIME: SemanticRetrievalRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def knowledge_retrieval_enabled() -> bool:
    """Enable semantic retrieval by default; explicit false is the rollback."""
    raw = os.getenv(KNOWLEDGE_RETRIEVAL_FLAG)
    if raw is None or not raw.strip():
        return True
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    _LOGGER.warning("Unrecognized %s value %r; retrieval remains enabled.", KNOWLEDGE_RETRIEVAL_FLAG, raw)
    return True


def load_knowledge_catalog(path: Path | str | None = None) -> KnowledgeCatalog:
    try:
        raw = json.loads((Path(path) if path else _CATALOG_PATH).read_text(encoding="utf-8"))
        return KnowledgeCatalog.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise KnowledgeRetrievalError("Professional knowledge catalog is invalid") from error


def load_human_experience_admission(path: Path | str | None = None) -> HumanExperienceAdmission:
    try:
        raw = json.loads((Path(path) if path else _HE_ADMISSION_PATH).read_text(encoding="utf-8"))
        return HumanExperienceAdmission.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise KnowledgeRetrievalError("Human Experience admission is invalid") from error


def build_retrieval_context(
    current_user_text: str,
    human_model: HumanModel,
    previous_acs: ActiveConversationState | None,
    dialogue_history: DialogueHistory,
) -> RetrievalContext:
    """Build independent meaning-bearing views, not a lexical query string."""
    # Current user material remains primary. Previous ACS can supply the meaning
    # of a short answer, but model-authored text must not outrank a rich current
    # answer or recursively steer retrieval toward its own wording.
    views = [SemanticQueryView("current_user_message", current_user_text, 1.0)]
    safety_views = [SemanticQueryView("current_user_message", current_user_text, 1.0)]
    if previous_acs is not None:
        views.extend(
            SemanticQueryView("active_conversation_content", item.content, 0.94)
            for item in previous_acs.active_content.values()
            if item.content.strip()
        )
        for target in previous_acs.response_targets.values():
            if target.subject.id in human_model.items:
                item = human_model.items[target.subject.id]
                if item.status not in {ItemStatus.REJECTED, ItemStatus.CORRECTED}:
                    views.append(SemanticQueryView("active_target_human_model", item.content, 0.96))
            if target.subject.id in human_model.relations:
                relation = human_model.relations[target.subject.id]
                views.append(SemanticQueryView("active_target_relation", relation.meaning, 0.96))
    rejected = 0
    for item in human_model.items.values():
        if item.status in {ItemStatus.REJECTED, ItemStatus.CORRECTED}:
            rejected += 1
        else:
            views.append(SemanticQueryView("human_model_item", item.content, 0.84))
            if item.provenance in {
                Provenance.USER_PROVIDED,
                Provenance.USER_INTERPRETATION,
            }:
                safety_views.append(
                    SemanticQueryView("human_model_item", item.content, 0.84)
                )
    views.extend(
        SemanticQueryView("human_model_relation", relation.meaning, 0.86)
        for relation in human_model.relations.values()
    )
    for message in dialogue_history.messages[-6:]:
        if message.role == DialogueRole.USER:
            view = SemanticQueryView("recent_user_message", message.text, 0.76)
            views.append(view)
            safety_views.append(view)
        else:
            views.append(SemanticQueryView("recent_system_message", message.text, 0.68))
    return RetrievalContext(
        query_views=_deduplicate_query_views(views),
        safety_query_views=_deduplicate_query_views(safety_views),
        rejected_or_corrected_item_count=rejected,
        active_context_present=previous_acs is not None,
    )


def retrieve_professional_context(
    current_user_text: str,
    human_model: HumanModel,
    previous_acs: ActiveConversationState | None,
    dialogue_history: DialogueHistory,
    *,
    include_human_experience: bool = True,
    catalog_path: Path | str | None = None,
    knowledge_token_budget: int = KNOWLEDGE_CONTEXT_TOKEN_BUDGET,
    human_experience_token_budget: int = HUMAN_EXPERIENCE_TOKEN_BUDGET,
    runtime: SemanticRetrievalRuntime | None = None,
) -> KnowledgeRetrievalSelection:
    """Retrieve candidates exclusively by dense semantic similarity."""
    started = monotonic()
    catalog = load_knowledge_catalog(catalog_path)
    indexed_he_cards = _eligible_human_experience(True)
    he_cards = indexed_he_cards if include_human_experience else ()
    expected_hash = _corpus_sha256(catalog, indexed_he_cards)
    semantic_runtime = runtime or _get_runtime()
    if semantic_runtime.index.corpus_sha256 != expected_hash:
        raise KnowledgeRetrievalError(
            "Semantic index is stale; rebuild with "
            "'python -m ai.knowledge_retrieval --prepare-index'."
        )
    context = build_retrieval_context(
        current_user_text, human_model, previous_acs, dialogue_history
    )
    per_view_scores = semantic_runtime.score_views(context.query_views)
    scores = _aggregate_score_maps(per_view_scores)
    current_scores = per_view_scores[0]
    contextual_scores = _aggregate_score_maps(per_view_scores[1:])
    # Safety may only be surfaced by user-origin material. Model-authored ACS,
    # replies and proposals are consultation context, not evidence of a medical
    # or psychological red flag.
    safety_view_keys = {
        (view.kind, view.text, view.weight) for view in context.safety_query_views
    }
    safety_scores = _aggregate_score_maps(
        tuple(
            view_scores
            for view, view_scores in zip(context.query_views, per_view_scores)
            if (view.kind, view.text, view.weight) in safety_view_keys
        )
    )
    professional_ids = {
        card.id for card in catalog.cards if card.kind != "safety_signal"
    }
    safety_ids = {
        card.id for card in catalog.cards if card.kind == "safety_signal"
    }
    he_ids = {card.id for card in he_cards}
    current_professional_floor = _candidate_floor(
        current_scores, professional_ids, PROFESSIONAL_MIN_SIMILARITY
    )
    contextual_professional_floor = _candidate_floor(
        contextual_scores, professional_ids, PROFESSIONAL_MIN_SIMILARITY
    )
    professional_floor = min(
        current_professional_floor, contextual_professional_floor
    )
    he_floor = _candidate_floor(
        scores, he_ids, HUMAN_EXPERIENCE_MIN_SIMILARITY
    )
    safety_floor = _candidate_floor(
        safety_scores, safety_ids, SAFETY_MIN_SIMILARITY
    )
    current_professional_ranked = sorted(
        (
            card for card in catalog.cards
            if card.kind != "safety_signal"
            and card.status == _RETRIEVABLE_STATUS
            and current_scores.get(card.id, -1.0) >= current_professional_floor
        ),
        key=lambda card: (-current_scores[card.id], card.id),
    )
    contextual_professional_ranked = sorted(
        (
            card for card in catalog.cards
            if card.kind != "safety_signal"
            and card.status == _RETRIEVABLE_STATUS
            and contextual_scores.get(card.id, -1.0)
            >= contextual_professional_floor
        ),
        key=lambda card: (-contextual_scores[card.id], card.id),
    )
    professional_ranked = _interleave_unique_cards(
        current_professional_ranked, contextual_professional_ranked
    )
    he_ranked = sorted(
        (
            card for card in he_cards
            if scores.get(card.id, -1.0) >= he_floor
        ),
        key=lambda card: (-scores[card.id], card.id),
    )
    # Safety remains an independent semantic lane. Absolute similarity finds a
    # possible boundary; category discrimination prevents an idiom or general
    # distress from simultaneously masquerading as several unrelated red flags.
    safety_cards, safety_discrimination_margin = _select_safety_cards(
        catalog.cards, safety_scores, safety_floor
    )
    selected_professional = _pack_cards(professional_ranked, knowledge_token_budget)
    selected_he = _pack_human_experience(he_ranked, human_experience_token_budget)
    estimated_tokens = sum(
        _estimate_tokens(json.dumps(_knowledge_card_view(card), ensure_ascii=False))
        for card in selected_professional + safety_cards
    )
    return KnowledgeRetrievalSelection(
        release_id=catalog.release_id,
        catalog_status=catalog.status,
        professional_cards=selected_professional,
        safety_cards=safety_cards,
        human_experience_cards=selected_he,
        indexed_knowledge_count=len(catalog.cards),
        indexed_human_experience_count=len(indexed_he_cards),
        estimated_tokens=estimated_tokens,
        active_context_used=context.active_context_present,
        embedding_model_id=semantic_runtime.index.embedding_model_id,
        index_corpus_sha256=semantic_runtime.index.corpus_sha256,
        query_view_kinds=tuple(view.kind for view in context.query_views),
        retrieval_latency_ms=(monotonic() - started) * 1000,
        professional_scores=tuple(
            (card.id, scores[card.id]) for card in selected_professional
        ),
        human_experience_scores=tuple(
            (card.id, scores[card.id]) for card in selected_he
        ),
        query_views=tuple(
            (view.kind, view.text, view.weight) for view in context.query_views
        ),
        safety_query_views=tuple(
            (view.kind, view.text, view.weight)
            for view in context.safety_query_views
        ),
        professional_candidate_scores=tuple(
            sorted(
                ((card_id, scores[card_id]) for card_id in professional_ids),
                key=lambda entry: (-entry[1], entry[0]),
            )
        ),
        human_experience_candidate_scores=tuple(
            sorted(
                ((card_id, scores[card_id]) for card_id in he_ids),
                key=lambda entry: (-entry[1], entry[0]),
            )
        ),
        safety_candidate_scores=tuple(
            sorted(
                (
                    (card_id, safety_scores.get(card_id, -1.0))
                    for card_id in safety_ids
                ),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        professional_similarity_floor=professional_floor,
        human_experience_similarity_floor=he_floor,
        safety_similarity_floor=safety_floor,
        safety_discrimination_margin=safety_discrimination_margin,
        index_schema_version=semantic_runtime.index.schema_version,
        semantic_view_version=semantic_runtime.index.semantic_view_version,
    )


def prepare_semantic_index(
    *,
    destination: Path | str | None = None,
    backend: EmbeddingBackend | None = None,
    allow_download: bool = True,
) -> Path:
    """Download/cache the model once and atomically build all card vectors."""
    catalog = load_knowledge_catalog()
    he_cards = _eligible_human_experience(True)
    documents = _semantic_documents(catalog, he_cards)
    embedding_backend = backend or FastEmbedBackend(allow_download=allow_download)
    vectors = embedding_backend.encode(
        [f"passage: {document.text}" for document in documents]
    )
    if len(vectors) != len(documents) or not vectors:
        raise KnowledgeRetrievalError("Unexpected document vector count")
    index = SemanticIndex(
        schema_version=SEMANTIC_INDEX_SCHEMA_VERSION,
        semantic_view_version=SEMANTIC_VIEW_VERSION,
        embedding_model_id=embedding_backend.model_id,
        embedding_dimension=len(vectors[0]),
        corpus_sha256=_corpus_sha256(catalog, he_cards),
        documents=tuple(
            SemanticIndexDocument(
                document_id=document.document_id,
                card_id=document.card_id,
                layer=document.layer,
                view_kind=document.view_kind,
                vector=tuple(_normalize(vector)),
            )
            for document, vector in zip(documents, vectors)
        ),
    )
    path = Path(destination) if destination else _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(index.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def load_semantic_index(path: Path | str | None = None) -> SemanticIndex:
    try:
        value = (Path(path) if path else _index_path()).read_text(encoding="utf-8")
        return SemanticIndex.model_validate_json(value)
    except (OSError, ValueError) as error:
        raise KnowledgeRetrievalError(
            "Semantic index is missing or invalid; run "
            "'python -m ai.knowledge_retrieval --prepare-index'."
        ) from error


def _get_runtime() -> SemanticRetrievalRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            index = load_semantic_index()
            backend = FastEmbedBackend(allow_download=False)
            probe = backend.encode(["query: Проверка локальной модели."])
            if not probe or len(probe[0]) != index.embedding_dimension:
                raise KnowledgeRetrievalError("Embedding model/index dimension mismatch")
            _RUNTIME = SemanticRetrievalRuntime(backend, index)
    return _RUNTIME


def _eligible_human_experience(include: bool) -> tuple[HumanExperienceCard, ...]:
    if not include:
        return ()
    catalog = load_human_experience_catalog()
    admission = load_human_experience_admission()
    catalog_ids = {card.id for card in catalog.entries}
    if admission.source_schema_version != catalog.schema_version:
        raise KnowledgeRetrievalError("Human Experience schema mismatch")
    if any(card_id not in catalog_ids for card_id in admission.eligible_ids):
        raise KnowledgeRetrievalError("Admission references unknown Human Experience")
    eligible = set(admission.eligible_ids)
    return tuple(card for card in catalog.entries if card.id in eligible)


def _semantic_documents(
    catalog: KnowledgeCatalog,
    human_experience: Sequence[HumanExperienceCard],
) -> tuple[SemanticDocument, ...]:
    """Create a few meaning-bearing views linked to each source card."""
    he_by_id = {card.id: card for card in human_experience}
    documents: list[SemanticDocument] = []
    for card in catalog.cards:
        if card.status != _RETRIEVABLE_STATUS:
            continue
        if card.kind == "safety_signal":
            # Safety retrieval is intentionally grounded in explicit red-flag
            # manifestations. Generic title/summary language about weight would
            # make ordinary consultations look safety-relevant in high-baseline
            # multilingual embedding space.
            text = " ".join(
                (
                    *card.supporting_signals,
                    *card.discriminating_observations,
                )
            )
            documents.append(
                SemanticDocument(
                    f"{card.id}::safety_signal",
                    card.id,
                    "safety",
                    "safety_signal",
                    text,
                )
            )
            continue
        manifestations = " ".join(
            he_by_id[card_id].possible_inner_experience
            for card_id in card.related_human_experience_ids
            if card_id in he_by_id
        )
        views = {
            "mechanism": " ".join((card.title, card.summary, card.mechanism)),
            "supporting_signals": " ".join(card.supporting_signals),
            "inquiry_limits_and_manifestations": " ".join(
                (
                    *card.contradicting_signals,
                    *card.discriminating_observations,
                    *card.prohibited_inferences,
                    manifestations,
                )
            ),
        }
        for kind, text in views.items():
            documents.append(
                SemanticDocument(
                    f"{card.id}::{kind}",
                    card.id,
                    "professional_knowledge",
                    kind,
                    text,
                )
            )
    for card in human_experience:
        views = {
            "lived_experience": " ".join(
                (
                    card.title,
                    card.lens,
                    card.possible_inner_experience,
                    *card.recognition_signals,
                )
            ),
            "applicability_alternatives_and_limits": " ".join(
                (card.use_when, card.avoid, *card.alternative_explanations)
            ),
        }
        for kind, text in views.items():
            documents.append(
                SemanticDocument(
                    f"{card.id}::{kind}",
                    card.id,
                    "human_experience",
                    kind,
                    text,
                )
            )
    return tuple(documents)


def _corpus_sha256(
    catalog: KnowledgeCatalog,
    human_experience: Sequence[HumanExperienceCard],
) -> str:
    payload = {
        "semantic_view_version": SEMANTIC_VIEW_VERSION,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "knowledge_catalog": catalog.model_dump(mode="json"),
        "human_experience": [
            card.model_dump(mode="json") for card in human_experience
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_root() -> Path:
    local = os.getenv("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".cache"
    return base / "ai_lead_bot/knowledge_retrieval"


def _index_path() -> Path:
    configured = os.getenv(KNOWLEDGE_RETRIEVAL_INDEX_PATH)
    return Path(configured) if configured else _cache_root() / "weight_management_semantic_index_v1.json"


def _model_cache_path() -> Path:
    configured = os.getenv(KNOWLEDGE_RETRIEVAL_MODEL_CACHE)
    return Path(configured) if configured else _cache_root() / "models"


def _normalize(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not values or not math.isfinite(norm) or norm <= 0:
        raise KnowledgeRetrievalError("Embedding vector is empty or invalid")
    return [value / norm for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise KnowledgeRetrievalError("Embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right))


def _candidate_floor(
    scores: dict[str, float],
    eligible_ids: set[str],
    absolute_floor: float,
) -> float:
    best = max((scores.get(card_id, -1.0) for card_id in eligible_ids), default=-1.0)
    if best < absolute_floor:
        return absolute_floor
    return max(absolute_floor, best - SEMANTIC_RELATIVE_BAND)


def _select_safety_cards(
    cards: Sequence[KnowledgeCard],
    scores: dict[str, float],
    floor: float,
) -> tuple[tuple[KnowledgeCard, ...], float | None]:
    """Select one specific semantic safety category or abstain.

    Safety cards represent distinct escalation boundaries. When the best match
    barely differs from another unrelated safety category, the embedding has
    detected broad emotional/weight language rather than a specific red flag.
    """
    ranked = sorted(
        (
            card
            for card in cards
            if card.kind == "safety_signal"
            and card.status == _RETRIEVABLE_STATUS
        ),
        key=lambda card: (-scores.get(card.id, -1.0), card.id),
    )
    if not ranked or scores.get(ranked[0].id, -1.0) < floor:
        return (), None
    runner_up = scores.get(ranked[1].id, -1.0) if len(ranked) > 1 else -1.0
    margin = scores[ranked[0].id] - runner_up
    if margin < SAFETY_MIN_DISCRIMINATION_MARGIN:
        return (), margin
    return (ranked[0],), margin


def _aggregate_score_maps(
    score_maps: Sequence[dict[str, float]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for scores in score_maps:
        for card_id, score in scores.items():
            result[card_id] = max(result.get(card_id, -1.0), score)
    return result


def _interleave_unique_cards(
    primary: Sequence[KnowledgeCard],
    contextual: Sequence[KnowledgeCard],
) -> tuple[KnowledgeCard, ...]:
    """Preserve the current leader and the narrow structured-context frontier."""
    result: list[KnowledgeCard] = []
    seen: set[str] = set()
    ordered = tuple(primary[:1]) + tuple(contextual) + tuple(primary[1:])
    for card in ordered:
        if card.id not in seen:
            result.append(card)
            seen.add(card.id)
    return tuple(result)


def _deduplicate_query_views(
    views: Iterable[SemanticQueryView],
) -> tuple[SemanticQueryView, ...]:
    result: list[SemanticQueryView] = []
    seen: set[tuple[str, str]] = set()
    for view in views:
        text = view.text.strip()
        key = (view.kind, text)
        if text and key not in seen:
            seen.add(key)
            result.append(SemanticQueryView(view.kind, text, view.weight))
    return tuple(result)


def _pack_cards(cards: Iterable[KnowledgeCard], budget: int) -> tuple[KnowledgeCard, ...]:
    selected: list[KnowledgeCard] = []
    used = 0
    for card in cards:
        cost = _estimate_tokens(json.dumps(_knowledge_card_view(card), ensure_ascii=False))
        if used + cost <= budget:
            selected.append(card)
            used += cost
    return tuple(selected)


def _pack_human_experience(
    cards: Iterable[HumanExperienceCard], budget: int
) -> tuple[HumanExperienceCard, ...]:
    selected: list[HumanExperienceCard] = []
    used = 0
    for card in cards:
        cost = _estimate_tokens(json.dumps(_human_experience_card_view(card), ensure_ascii=False))
        if used + cost <= budget:
            selected.append(card)
            used += cost
    return tuple(selected)


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _knowledge_card_view(card: KnowledgeCard) -> dict[str, Any]:
    return {
        "id": card.id,
        "kind": card.kind,
        "review_status": card.status,
        "summary": card.summary,
        "mechanism": card.mechanism,
        "supporting_signals": list(card.supporting_signals),
        "contradicting_signals": list(card.contradicting_signals),
        "alternative_explanations": list(card.alternative_explanations),
        "discriminating_observations": list(card.discriminating_observations),
        "prohibited_inferences": list(card.prohibited_inferences),
        "source_ids": list(card.source_ids),
        "related_human_experience_ids": list(card.related_human_experience_ids),
        "safety": card.safety.model_dump(mode="json"),
    }


def _human_experience_card_view(card: HumanExperienceCard) -> dict[str, Any]:
    return {
        "id": card.id,
        "review_status": "research_catalog_not_expert_validated",
        "professional_hypothesis": card.possible_inner_experience,
        "recognition_example": card.recognition_to_user,
        "discriminating_question": card.discriminating_question,
        "alternative_explanations": list(card.alternative_explanations),
        "use_when": card.use_when,
        "avoid": card.avoid,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-index", action="store_true")
    parser.add_argument("--verify-index", action="store_true")
    args = parser.parse_args()
    if args.prepare_index:
        print(f"semantic index prepared: {prepare_semantic_index(allow_download=True)}")
        return 0
    if args.verify_index:
        index = load_semantic_index()
        print(
            f"semantic index valid: {len(index.documents)} views; "
            f"corpus_sha256={index.corpus_sha256}"
        )
        return 0
    parser.error("choose --prepare-index or --verify-index")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
