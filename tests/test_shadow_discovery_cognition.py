import pytest

from ai.discovery_data_model import (
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    ItemStatus,
    ModelItemKind,
    Provenance,
    validate_discovery_memory,
)
from ai.shadow_discovery_cognition import (
    InteractionSignals,
    MaterialOperation,
    PerceptionMaterial,
    PerceptionReference,
    PerceptionRelationCandidate,
    PerceptionResult,
    ShadowCognitionError,
    ShadowIntegrationEngine,
    SourceSpan,
)


def _message(text: str, sequence: int = 1) -> DialogueMessage:
    return DialogueMessage(role=DialogueRole.USER, text=text, sequence=sequence)


def _span(text: str) -> SourceSpan:
    return SourceSpan(char_start=0, char_end=len(text))


def _integrate(message, perception, human_model=None, turn=1):
    return ShadowIntegrationEngine().integrate(
        perception,
        human_model or HumanModel(),
        message,
        turn,
    )


def test_integrates_independent_user_material_with_current_message_reference():
    text = "Я набрал вес за последние три года."
    message = _message(text)
    result = _integrate(
        message,
        PerceptionResult(
            materials=(
                PerceptionMaterial(
                    local_id="weight_change",
                    normalized_content="пользователь набрал вес за последние три года",
                    kind=ModelItemKind.LIFE_CHANGE,
                    provenance=Provenance.USER_PROVIDED,
                    source_span=_span(text),
                    operation=MaterialOperation.NEW,
                ),
            )
        ),
    )

    item = result.human_model.items[result.added_item_ids[0]]
    assert item.provenance == Provenance.USER_PROVIDED
    assert item.status == ItemStatus.RECORDED
    assert item.source_refs[0].message_id == message.id
    validate_discovery_memory(result.human_model, None, DialogueHistory(messages=(message,)))


def test_integrates_explicit_attempt_barrier_and_user_expressed_relation():
    text = "Я пробовал меньше есть, но из-за голода вечером срывался."
    message = _message(text)
    result = _integrate(
        message,
        PerceptionResult(
            materials=(
                PerceptionMaterial(
                    local_id="attempt",
                    normalized_content="пользователь пробовал есть меньше",
                    kind=ModelItemKind.PREVIOUS_ATTEMPT,
                    provenance=Provenance.USER_PROVIDED,
                    source_span=SourceSpan(char_start=0, char_end=20),
                    operation=MaterialOperation.NEW,
                ),
                PerceptionMaterial(
                    local_id="hunger",
                    normalized_content="вечерний голод мешал попытке",
                    kind=ModelItemKind.BARRIER,
                    provenance=Provenance.USER_PROVIDED,
                    source_span=SourceSpan(char_start=29, char_end=len(text)),
                    operation=MaterialOperation.NEW,
                ),
            ),
            relation_candidates=(
                PerceptionRelationCandidate(
                    source_references=(PerceptionReference(local_id="hunger"),),
                    target_references=(PerceptionReference(local_id="attempt"),),
                    normalized_meaning="пользователь связывает вечерний голод со срывами попытки есть меньше",
                    provenance=Provenance.USER_INTERPRETATION,
                    source_span=_span(text),
                ),
            ),
        ),
    )

    items = tuple(result.human_model.items.values())
    relation = result.human_model.relations[result.added_relation_ids[0]]
    assert {item.kind for item in items} == {
        ModelItemKind.PREVIOUS_ATTEMPT,
        ModelItemKind.BARRIER,
    }
    assert relation.provenance == Provenance.USER_INTERPRETATION
    assert relation.status == ItemStatus.RECORDED
    assert relation.source_refs[0].message_id == message.id
    assert all(item.provenance != Provenance.SYSTEM_PROPOSED for item in items)


def test_keeps_other_person_view_separate_from_user_provided_material():
    text = "Жена говорит, что я стал менее активным."
    message = _message(text)
    result = _integrate(
        message,
        PerceptionResult(
            materials=(
                PerceptionMaterial(
                    local_id="wife_view",
                    normalized_content="жена считает пользователя менее активным",
                    kind=ModelItemKind.OTHER_PERSON_VIEW,
                    provenance=Provenance.OTHER_PERSON,
                    source_span=_span(text),
                    operation=MaterialOperation.NEW,
                ),
            )
        ),
    )

    item = result.human_model.items[result.added_item_ids[0]]
    assert item.kind == ModelItemKind.OTHER_PERSON_VIEW
    assert item.provenance == Provenance.OTHER_PERSON
    assert item.status == ItemStatus.RECORDED
    assert all(candidate.provenance != Provenance.USER_PROVIDED for candidate in result.human_model.items.values())


def test_correction_retains_old_item_and_creates_a_superseding_item():
    first_text = "Вес начал расти два года назад."
    first_message = _message(first_text)
    first_result = _integrate(
        first_message,
        PerceptionResult(
            materials=(
                PerceptionMaterial(
                    local_id="duration_two",
                    normalized_content="вес начал расти два года назад",
                    kind=ModelItemKind.LIFE_CHANGE,
                    provenance=Provenance.USER_PROVIDED,
                    source_span=_span(first_text),
                    operation=MaterialOperation.NEW,
                ),
            )
        ),
    )
    old_item_id = first_result.added_item_ids[0]

    second_text = "Нет, скорее три года назад."
    second_message = _message(second_text, sequence=3)
    second_result = _integrate(
        second_message,
        PerceptionResult(
            materials=(
                PerceptionMaterial(
                    local_id="duration_three",
                    normalized_content="вес начал расти три года назад",
                    kind=ModelItemKind.LIFE_CHANGE,
                    provenance=Provenance.USER_PROVIDED,
                    source_span=_span(second_text),
                    operation=MaterialOperation.CORRECT,
                    existing_item_id=old_item_id,
                ),
            )
        ),
        first_result.human_model,
        turn=2,
    )

    new_item_id = second_result.corrected_item_ids[1]
    assert second_result.human_model.items[old_item_id].status == ItemStatus.CORRECTED
    assert second_result.human_model.items[new_item_id].supersedes_item_ids == (old_item_id,)
    assert second_result.human_model.items[old_item_id].source_refs[0].message_id == first_message.id
    assert second_result.human_model.items[new_item_id].source_refs[0].message_id == second_message.id


def test_reinforcement_adds_current_source_reference_without_duplicate_item():
    first_text = "Я стесняюсь живота."
    first_message = _message(first_text)
    first_result = _integrate(
        first_message,
        PerceptionResult(
            materials=(
                PerceptionMaterial(
                    local_id="shame",
                    normalized_content="пользователь стесняется живота",
                    kind=ModelItemKind.EMOTION,
                    provenance=Provenance.USER_PROVIDED,
                    source_span=_span(first_text),
                    operation=MaterialOperation.NEW,
                ),
            )
        ),
    )
    item_id = first_result.added_item_ids[0]

    second_text = "Да, живот меня реально смущает."
    second_result = _integrate(
        _message(second_text, sequence=3),
        PerceptionResult(
            materials=(
                PerceptionMaterial(
                    local_id="same_shame",
                    normalized_content="пользователь стесняется живота",
                    kind=ModelItemKind.EMOTION,
                    provenance=Provenance.USER_PROVIDED,
                    source_span=_span(second_text),
                    operation=MaterialOperation.REINFORCE,
                    existing_item_id=item_id,
                ),
            )
        ),
        first_result.human_model,
        turn=2,
    )

    assert tuple(second_result.human_model.items) == (item_id,)
    assert second_result.reinforced_item_ids == (item_id,)
    assert len(second_result.human_model.items[item_id].source_refs) == 2


@pytest.mark.parametrize(
    "text, signals",
    [
        ("Почему ты это спрашиваешь?", InteractionSignals(direct_question=True)),
        ("Не хочу об этом говорить.", InteractionSignals(refusal_or_pause=True)),
    ],
)
def test_transient_interaction_signals_do_not_create_human_model_items(text, signals):
    result = _integrate(_message(text), PerceptionResult(interaction_signals=signals))

    assert result.human_model.items == {}
    assert result.human_model.relations == {}


def test_invalid_existing_item_id_is_rejected_without_returning_an_update():
    text = "Я всё ещё стесняюсь живота."
    with pytest.raises(ShadowCognitionError, match="unknown ModelItem"):
        _integrate(
            _message(text),
            PerceptionResult(
                materials=(
                    PerceptionMaterial(
                        local_id="bad",
                        normalized_content="пользователь стесняется живота",
                        kind=ModelItemKind.EMOTION,
                        provenance=Provenance.USER_PROVIDED,
                        source_span=_span(text),
                        operation=MaterialOperation.REINFORCE,
                        existing_item_id="mi_00000000000000000000000000000000",
                    ),
                )
            ),
        )
