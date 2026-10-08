"""Concrete deterministic Telegram handoff for the current consultant flow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ai.discovery_data_model import HumanModel, ItemStatus, ModelItemKind, Provenance


CONSULTANT_CHAT_ID = 402960599
CONTACT_REQUEST_TEXT = (
    "Отлично. Ваша ситуация нам знакома. И хорошая новость здесь в том, что "
    "теперь речь уже не просто о «лишнем весе» вообще. Мы увидели, где именно "
    "у вас возникает основная трудность и почему прежние попытки не давали "
    "устойчивого результата.\n\n"
    "Наши консультанты имеют большой опыт работы с такими ситуациями и знают, "
    "как двигаться дальше. Поэтому следующая консультация начнётся уже не с "
    "поиска проблемы — а с понимания того, что именно нужно решить в вашем "
    "случае.\n\n"
    "Оставьте, пожалуйста, номер телефона, по которому консультант сможет "
    "связаться с вами в Telegram."
)
CONTACT_BUTTON_TEXT = "Поделиться номером телефона"
INVALID_CONTACT_TEXT = (
    "Пожалуйста, отправьте корректный номер телефона через кнопку "
    "«Поделиться номером телефона» или введите его вручную."
)
DELIVERY_FAILURE_TEXT = (
    "Сейчас не удалось передать информацию консультанту из-за технической "
    "ошибки. Ваш номер и подготовленное резюме сохранены; отправьте сообщение "
    "ещё раз, чтобы повторить передачу."
)
FINAL_USER_TEXT = (
    "Спасибо. Я передал консультанту главное из нашего разговора, чтобы вам не "
    "пришлось начинать всё сначала."
)


class TelegramHandoffStatus(str, Enum):
    AWAITING_CONTACT = "awaiting_contact"
    DELIVERY_PENDING = "delivery_pending"
    COMPLETED = "completed"


@dataclass(frozen=True)
class TelegramConsultantHandoffState:
    status: TelegramHandoffStatus
    raw_phone: Optional[str] = None
    normalized_phone: Optional[str] = None
    consultant_summary: Optional[str] = None


_ALLOWED_PHONE = re.compile(r"^\+?[0-9][0-9 ()-]*$")


def normalize_phone(value: Optional[str]) -> Optional[str]:
    """Return a conservative international-or-local phone representation."""
    if value is None:
        return None
    raw = value.strip()
    if not raw or not _ALLOWED_PHONE.fullmatch(raw):
        return None
    digits = "".join(character for character in raw if character.isdigit())
    if not 7 <= len(digits) <= 15:
        return None
    return f"+{digits}" if raw.startswith("+") else digits


_SECTION_BY_KIND = {
    ModelItemKind.STATED_PROBLEM: "Заявленная проблема",
    ModelItemKind.EXPERIENCE: "Значимый опыт и влияние",
    ModelItemKind.EMOTION: "Значимый опыт и влияние",
    ModelItemKind.PERSONAL_MEANING: "Значимый опыт и влияние",
    ModelItemKind.LIFE_CHANGE: "Изменения и контекст",
    ModelItemKind.BEFORE: "Изменения и контекст",
    ModelItemKind.NOW: "Изменения и контекст",
    ModelItemKind.DESIRED_CHANGE: "Желаемое изменение",
    ModelItemKind.PREVIOUS_ATTEMPT: "Предыдущие попытки",
    ModelItemKind.BARRIER: "Барьеры и повторяющиеся механизмы",
    ModelItemKind.USER_INTERPRETATION: "Понимание пользователя",
    ModelItemKind.OTHER_PERSON_VIEW: "Другой релевантный контекст",
    ModelItemKind.OTHER_RELEVANT_USER_MATERIAL: "Другой релевантный контекст",
}


def _epistemic_label(provenance: Provenance, status: ItemStatus) -> str:
    if status == ItemStatus.REJECTED:
        return "отвергнуто"
    if status == ItemStatus.UNCERTAIN:
        return "остаётся неопределённым"
    if status == ItemStatus.PARTIALLY_SUPPORTED:
        return "частично подтверждено"
    if status == ItemStatus.CORRECTED:
        return "скорректировано"
    if provenance == Provenance.SYSTEM_PROPOSED:
        return "рабочая версия системы, подтверждена пользователем"
    if provenance == Provenance.USER_INTERPRETATION:
        return "интерпретация пользователя"
    return ""


def build_consultant_summary(human_model: HumanModel, phone: str) -> str:
    """Build a grounded, provenance-aware summary without another LLM call."""
    sections: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for item in sorted(human_model.items.values(), key=lambda value: (value.created_turn, value.id)):
        section = _SECTION_BY_KIND[item.kind]
        label = _epistemic_label(item.provenance, item.status)
        line = item.content if not label else f"{item.content} [{label}]"
        key = (section, line)
        if key in seen:
            continue
        seen.add(key)
        sections.setdefault(section, []).append(line)

    relation_lines = []
    for relation in sorted(
        human_model.relations.values(), key=lambda value: (value.created_turn, value.id)
    ):
        label = _epistemic_label(relation.provenance, relation.status)
        relation_lines.append(
            relation.meaning if not label else f"{relation.meaning} [{label}]"
        )
    if relation_lines:
        sections["Значимые связи"] = relation_lines

    lines = ["Новая заявка на консультацию", "", "Согласие на консультацию: получено."]
    for title, entries in sections.items():
        lines.extend(("", f"{title}:"))
        lines.extend(f"• {entry}" for entry in entries)
    lines.extend(("", f"Контактный телефон: {phone}"))
    return "\n".join(lines)
