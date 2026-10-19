import json
from pathlib import Path

import pytest

from ai.human_experience import (
    HUMAN_EXPERIENCE_FLAG,
    MAX_CONTEXT_CHARS,
    MAX_SELECTED_CARDS,
    HumanExperienceError,
    human_experience_enabled,
    load_human_experience_catalog,
    select_human_experience,
)


CATALOG_PATH = (
    Path(__file__).parents[1]
    / "Research"
    / "Human_Experience"
    / "Weight"
    / "experimental"
    / "Human_Experience_Pain_Map_V2"
    / "human_experience_pain_map_v2.json"
)


def _raw_catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _write_catalog(tmp_path, payload):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_catalog_validates_only_the_30_v2_entries():
    catalog = load_human_experience_catalog(CATALOG_PATH)

    assert catalog.status == "DESIGN_ONLY_NOT_PRODUCTION"
    assert len(catalog.entries) == 30
    assert {card.id for card in catalog.entries} == {
        f"P{index:03d}" for index in range(1, 31)
    }
    assert len(catalog.original_v1) == 12


@pytest.mark.parametrize("content", ("{broken", "[]"))
def test_missing_or_corrupt_catalog_fails_closed(tmp_path, content):
    path = tmp_path / "catalog.json"
    if content:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(HumanExperienceError):
        load_human_experience_catalog(path)


def test_wrong_or_duplicate_entry_id_is_rejected(tmp_path):
    wrong = _raw_catalog()
    wrong["entries"][0]["id"] = "X001"
    with pytest.raises(HumanExperienceError):
        load_human_experience_catalog(_write_catalog(tmp_path, wrong))

    duplicate = _raw_catalog()
    duplicate["entries"][1]["id"] = "P001"
    with pytest.raises(HumanExperienceError):
        load_human_experience_catalog(_write_catalog(tmp_path, duplicate))


def test_missing_required_card_or_source_field_is_rejected(tmp_path):
    missing_card_field = _raw_catalog()
    del missing_card_field["entries"][0]["avoid"]
    with pytest.raises(HumanExperienceError):
        load_human_experience_catalog(
            _write_catalog(tmp_path, missing_card_field)
        )

    missing_source_field = _raw_catalog()
    del missing_source_field["sources"]["BECK"]["access"]
    with pytest.raises(HumanExperienceError):
        load_human_experience_catalog(
            _write_catalog(tmp_path, missing_source_field)
        )


def test_explicit_recurring_failure_pattern_selects_p001_from_history():
    selection = select_human_experience(
        "Да, почти каждый раз происходит именно так.",
        (
            "После тяжёлой рабочей недели я пропускаю обед и вечером переедаю.",
            "Потом думаю, что всё испортил.",
            "После этого бросаю попытку на несколько недель.",
        ),
        catalog_path=CATALOG_PATH,
    )

    assert [card.id for card in selection.cards] == ["P001"]
    view = selection.input_view()
    assert view["status"] == "experimental_hypotheses_not_evidence"
    assert view["abstained"] is False
    assert len(view["selected_cards"]) <= MAX_SELECTED_CARDS
    assert len(json.dumps(view["selected_cards"], ensure_ascii=False)) <= MAX_CONTEXT_CHARS
    assert "original_v1" not in view
    assert "professional_hypothesis" in view["selected_cards"][0]
    assert "recognition_example" in view["selected_cards"][0]
    assert "discriminating_question" in view["selected_cards"][0]


@pytest.mark.parametrize(
    ("text", "expected_id"),
    (
        (
            "После тяжёлого дня вечером еда — единственное, что меня радует.",
            "P006",
        ),
        (
            "Когда нервничаю, я прямо заедаю стресс, чтобы немного успокоиться.",
            "P007",
        ),
    ),
)
def test_other_pilot_cards_require_explicit_functional_meaning(text, expected_id):
    selection = select_human_experience(text, (), catalog_path=CATALOG_PATH)

    assert [card.id for card in selection.cards] == [expected_id]


@pytest.mark.parametrize(
    ("current", "history"),
    (
        ("Сегодня на улице солнечно.", ("Хочу обсудить расписание.",)),
        ("Вечером переел после того, как пропустил обед.", ()),
        ("После тяжёлого разговора вечером переел.", ()),
        (
            "Нет, один вечер не означает, что всё испорчено; я продолжаю.",
            ("Раньше после переедания думал, что всё испортил, и бросал.",),
        ),
        (
            "Нет, это не так.",
            ("После срыва думал, что всё испортил, и бросал попытку.",),
        ),
        (
            "Давайте разберёмся дальше.",
            (
                "После срыва думал, что всё испортил, и бросал попытку.",
                "Нет, сейчас я так не думаю и продолжаю.",
            ),
        ),
    ),
)
def test_unclear_irrelevant_or_contradicted_context_abstains(current, history):
    selection = select_human_experience(
        current,
        history,
        catalog_path=CATALOG_PATH,
    )

    assert selection.cards == ()
    assert selection.abstain_reason == "no_reliable_match"


def test_catalog_failure_becomes_logged_safe_retrieval_abstention(tmp_path, caplog):
    selection = select_human_experience(
        "После срыва думаю, что всё испортил, и бросаю.",
        (),
        catalog_path=tmp_path / "missing.json",
    )

    assert selection.cards == ()
    assert selection.abstain_reason == "catalog_unavailable"
    assert "retrieval abstained" in caplog.text


def test_flag_defaults_on_and_supports_emergency_override(monkeypatch):
    monkeypatch.delenv(HUMAN_EXPERIENCE_FLAG, raising=False)
    assert human_experience_enabled() is True

    monkeypatch.setenv(HUMAN_EXPERIENCE_FLAG, "true")
    assert human_experience_enabled() is True

    monkeypatch.setenv(HUMAN_EXPERIENCE_FLAG, "false")
    assert human_experience_enabled() is False


def test_unrecognized_flag_value_fails_open_with_diagnostic(monkeypatch, caplog):
    monkeypatch.setenv(HUMAN_EXPERIENCE_FLAG, "unexpected")

    assert human_experience_enabled() is True
    assert "Unrecognized HUMAN_EXPERIENCE_ENABLED" in caplog.text
