"""Experimental Human Experience V2 loading and conservative retrieval.

The module is deliberately not a new reasoning engine.  It validates research
cards and prepares at most three optional hypotheses for the existing, single
Cognitive Core call.  Human Experience never mutates Human Model directly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


_LOGGER = logging.getLogger(__name__)
HUMAN_EXPERIENCE_FLAG = "HUMAN_EXPERIENCE_ENABLED"
MAX_SELECTED_CARDS = 3
MAX_CONTEXT_CHARS = 6_000
_EXPECTED_IDS = {f"P{index:03d}" for index in range(1, 31)}
_CATALOG_PATH = (
    Path(__file__).parents[1]
    / "Research"
    / "Human_Experience"
    / "Weight"
    / "experimental"
    / "Human_Experience_Pain_Map_V2"
    / "human_experience_pain_map_v2.json"
)


HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION = """

EXPERIMENTAL HUMAN EXPERIENCE MODE:
- In this request human_experience.enabled=true and selected_cards contains
  only the bounded experimental context for this turn.
- selected_cards are optional professional working versions, never facts or
  evidence about this person. Use a card only when the current USER material
  and history genuinely support it; otherwise ignore it.
- Cards are inputs to your internal Formulation, not scripts for the reply.
  Express any useful connection naturally in the current context; do not
  announce labels such as "working hypothesis" or reproduce a card as a
  prefabricated observation or question.
- recognition_example illustrates the useful human meaning carried by a card,
  and discriminating_question illustrates what would distinguish it when that
  distinction materially changes the next move. Adapt the meaning naturally;
  neither field is a required phrase or a required question.
- You may use none of the cards. Do not expose card IDs, internal titles or the
  retrieval mechanism. Do not name a hidden or "real" pain, invent feelings,
  motives or causes, or make the card stronger than the user's material.
- Prefer one grounded, easily correctable observation over a list of versions.
  Do not mechanically repeat the USER message and do not demand confirmation.
  A useful Reflection or Recognition may end without a question.
- If the user rejects or contradicts a version, drop it. Human Model remains the
  only source of claims about the specific person.
""".rstrip()

class HumanExperienceError(RuntimeError):
    """The experimental catalog could not be safely loaded or validated."""


class HumanExperienceCard(BaseModel):
    """One validated V2 research card; V1 cards use another schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    lens: str
    recognition_signals: tuple[str, ...]
    possible_inner_experience: str
    recognition_to_user: str
    discriminating_question: str
    alternative_explanations: tuple[str, ...]
    inspiration_sources: tuple[str, ...]
    use_when: str
    avoid: str

    @field_validator(
        "id",
        "title",
        "lens",
        "possible_inner_experience",
        "recognition_to_user",
        "discriminating_question",
        "use_when",
        "avoid",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("card text fields must be non-empty strings")
        return value

    @field_validator(
        "recognition_signals", "alternative_explanations", "inspiration_sources"
    )
    @classmethod
    def _non_empty_string_list(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("card list fields must contain non-empty strings")
        return value


class HumanExperienceSource(BaseModel):
    """Minimal provenance metadata required by the supplied research schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    url: str
    access: str

    @field_validator("title", "url", "access")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source metadata must be non-empty")
        return value


class HumanExperienceCatalog(BaseModel):
    """Strict envelope for the supplied research artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    status: str
    purpose: str
    basis_note: str
    sources: dict[str, HumanExperienceSource]
    entries: tuple[HumanExperienceCard, ...]
    original_v1: tuple[dict[str, Any], ...]

    @field_validator("purpose", "basis_note")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("catalog metadata must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_research_contract(self) -> "HumanExperienceCatalog":
        if self.schema_version != "2.0-research":
            raise ValueError("unexpected Human Experience schema_version")
        if self.status != "DESIGN_ONLY_NOT_PRODUCTION":
            raise ValueError("research status must remain DESIGN_ONLY_NOT_PRODUCTION")
        if len(self.entries) != 30:
            raise ValueError("the V2 pilot requires exactly 30 entries")
        ids = [card.id for card in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Human Experience entry IDs must be unique")
        if set(ids) != _EXPECTED_IDS:
            raise ValueError("Human Experience entry IDs must be P001..P030")
        source_ids = set(self.sources)
        if any(
            source_id not in source_ids
            for card in self.entries
            for source_id in card.inspiration_sources
        ):
            raise ValueError("card references an unknown inspiration source")
        return self


@dataclass(frozen=True)
class HumanExperienceSelection:
    """A bounded optional context fragment for one Cognitive Core request."""

    cards: tuple[HumanExperienceCard, ...]
    abstain_reason: str | None = None

    def input_view(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "status": "experimental_hypotheses_not_evidence",
            "retrieval_method": "conservative_explicit_scenarios_v1",
            "abstained": not self.cards,
            "abstain_reason": self.abstain_reason,
            "selected_cards": [_card_view(card) for card in self.cards],
        }


def human_experience_enabled() -> bool:
    """Enable Human Experience by default, with an emergency env override."""
    raw_value = os.getenv(HUMAN_EXPERIENCE_FLAG)
    if raw_value is None or not raw_value.strip():
        return True

    value = raw_value.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    _LOGGER.warning(
        "Unrecognized %s value %r; Human Experience remains enabled.",
        HUMAN_EXPERIENCE_FLAG,
        raw_value,
    )
    return True


def load_human_experience_catalog(
    path: Path | str | None = None,
) -> HumanExperienceCatalog:
    """Load and strictly validate the 30-card V2 research catalog."""
    catalog_path = Path(path) if path is not None else _CATALOG_PATH
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        return HumanExperienceCatalog.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HumanExperienceError(
            "Human Experience V2 catalog is unavailable or invalid."
        ) from error


def select_human_experience(
    current_user_text: str,
    previous_user_texts: tuple[str, ...],
    *,
    catalog_path: Path | str | None = None,
) -> HumanExperienceSelection:
    """Select a small, conservative pilot set or explicitly abstain.

    This first slice intentionally covers only three explicit semantic
    scenarios.  It requires conjunctions of meaning-bearing signals and never
    treats a single keyword as evidence.  Unknown scenarios abstain.
    """
    try:
        catalog = load_human_experience_catalog(catalog_path)
    except HumanExperienceError:
        _LOGGER.warning(
            "Human Experience retrieval abstained because the catalog is "
            "unavailable or invalid.",
            exc_info=True,
        )
        return HumanExperienceSelection(cards=(), abstain_reason="catalog_unavailable")

    texts = tuple(previous_user_texts[-12:]) + (current_user_text,)
    normalized = tuple(_normalize(text) for text in texts if text.strip())
    card_by_id = {card.id: card for card in catalog.entries}
    selected_ids = [
        card_id
        for card_id, predicate in (
            ("P001", _supports_p001),
            ("P006", _supports_p006),
            ("P007", _supports_p007),
        )
        if predicate(normalized)
    ]

    selected: list[HumanExperienceCard] = []
    for card_id in selected_ids[:MAX_SELECTED_CARDS]:
        candidate = selected + [card_by_id[card_id]]
        if _context_size(candidate) <= MAX_CONTEXT_CHARS:
            selected.append(card_by_id[card_id])
    if not selected:
        return HumanExperienceSelection(cards=(), abstain_reason="no_reliable_match")
    return HumanExperienceSelection(cards=tuple(selected))


def _normalize(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


def _has(texts: tuple[str, ...], patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for text in texts for pattern in patterns)


def _supports_p001(texts: tuple[str, ...]) -> bool:
    recent = " ".join(texts[-3:])
    if re.search(
        r"^(?:нет|не совсем|это не так)\b|"
        r"\bне (?:считаю|думаю|думал|думала|означает).{0,55}"
        r"(?:испорт|провал|обнул)|\bпродолж(?:аю|ил|ила|или)\b",
        recent,
    ):
        return False
    return all(
        (
            _has(texts, patterns)
            for patterns in (
                (r"\bсорв\w*", r"\bпереед\w*", r"неудачн\w*\s+(?:вечер|день)", r"\bнаруш\w*"),
                (r"все\s+(?:испорт|пропал|потер)", r"(?:вся|всю)\s+попыт\w*.{0,25}(?:провал|испорт)", r"\bobнул\w*"),
                (r"\bброс\w*", r"\bпрекрат\w*", r"\bсдал\w*", r"смысл\w*\s+нет"),
            )
        )
    )


def _supports_p006(texts: tuple[str, ...]) -> bool:
    recent = " ".join(texts[-3:])
    if re.search(r"\bне (?:устаю|выжат|считаю еду наградой)", recent):
        return False
    return all(
        (
            _has(texts, patterns)
            for patterns in (
                (r"\bустал\w*", r"\bвыжат\w*", r"тяжел\w*\s+(?:дн|недел|смен)\w*"),
                (r"\bвечер\w*", r"после работы", r"прихожу домой"),
                (r"еда.{0,60}(?:раду|утеш|единствен)", r"\bзаслуж\w*", r"хоть что-то приятн", r"чтобы.{0,30}(?:расслаб|переключ)"),
            )
        )
    )


def _supports_p007(texts: tuple[str, ...]) -> bool:
    recent = " ".join(texts[-3:])
    if re.search(r"\bне\s+заед\w*|еда\s+не\s+(?:успока|помога)", recent):
        return False
    return _has(
        texts,
        (
            r"\bзаед\w*.{0,35}(?:стресс|тревог|нерв|напряж)",
            r"(?:стресс|тревог|нервнича|напряж).{0,35}\bзаед\w*",
            r"когда\s+нервнича\w*.{0,25}\b(?:ем|переед\w*)\b",
            r"еда.{0,25}(?:успока|помога.{0,15}(?:переключ|расслаб))",
        ),
    )


def _card_view(card: HumanExperienceCard) -> dict[str, Any]:
    return {
        "id": card.id,
        "professional_hypothesis": card.possible_inner_experience,
        "recognition_example": card.recognition_to_user,
        "discriminating_question": card.discriminating_question,
        "alternative_explanations": list(card.alternative_explanations),
        "use_when": card.use_when,
        "avoid": card.avoid,
    }


def _context_size(cards: list[HumanExperienceCard]) -> int:
    return len(
        json.dumps(
            [_card_view(card) for card in cards],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
