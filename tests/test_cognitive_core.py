import json
import inspect
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ai.cognitive_core import (
    COGNITIVE_TURN_JSON_SCHEMA,
    CognitiveCore,
    CognitiveCoreError,
    _normalize_strict_schema,
)
import ai.cognitive_core as cognitive_core_module
from ai.cognitive_turn import (
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    ReplySegment,
    SystemAction,
)
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
    Relation,
    ResponseTarget,
    SourceReference,
    TargetInteractionKind,
    TargetReference,
    TargetSubjectKind,
)
from ai.discovery_state_applier import apply_cognitive_turn


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def _message(role, text, sequence):
    return DialogueMessage(role=role, text=text, sequence=sequence)


def _result(reply="Что для тебя сейчас важнее?"):
    content = ActionContent(
        local_id="content:question",
        kind=ActiveContentKind.SYSTEM_QUESTION,
        semantic_content="уточнить значимую область",
    )
    return CognitiveTurnResult(
        system_action=SystemAction(
            contents=(content,),
            response_targets=(
                ActionTarget(
                    local_id="target:question",
                    active_content_local_id=content.local_id,
                    subject=ActionSubjectReference(
                        kind=TargetSubjectKind.ACTIVE_CONTENT,
                        active_content_local_id=content.local_id,
                    ),
                    interaction=TargetInteractionKind.OPEN_RESPONSE,
                ),
            ),
        ),
        reply_segments=(
            ReplySegment(
                local_id="segment:reply",
                text=reply,
                realizes_action_content_ids=(content.local_id,),
            ),
        ),
    )


def _fixture(with_acs=True):
    history_messages = tuple(
        message
        for turn in range(1, 7)
        for message in (
            _message(DialogueRole.USER, f"старое user {turn}", (turn * 2) - 1),
            _message(DialogueRole.SYSTEM, f"старое system {turn}", turn * 2),
        )
    )
    old_user = history_messages[0]
    old_system = history_messages[1]
    item = ModelItem(
        kind=ModelItemKind.EXPERIENCE,
        content="старый значимый опыт",
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.REJECTED,
        source_refs=(SourceReference(message_id=old_user.id),),
        created_turn=1,
        updated_turn=1,
    )
    corrected = ModelItem(
        kind=ModelItemKind.EXPERIENCE,
        content="скорректированный опыт",
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.CORRECTED,
        source_refs=(SourceReference(message_id=old_user.id),),
        created_turn=1,
        updated_turn=2,
    )
    relation = Relation(
        meaning="пользователь видит связь",
        source_item_ids=(item.id,),
        target_item_ids=(corrected.id,),
        provenance=Provenance.USER_INTERPRETATION,
        status=ItemStatus.UNCERTAIN,
        source_refs=(SourceReference(message_id=old_user.id),),
        created_turn=1,
        updated_turn=2,
    )
    model = HumanModel(items={item.id: item, corrected.id: corrected}, relations={relation.id: relation})
    acs = None
    if with_acs:
        content = ActiveContentItem(
            kind=ActiveContentKind.SYSTEM_PROPOSAL,
            content="предыдущая системная версия",
            model_item_ids=(item.id,),
            source_message_id=old_system.id,
        )
        target = ResponseTarget(
            active_content_id=content.id,
            subject=TargetReference(kind=TargetSubjectKind.MODEL_ITEM, id=item.id),
            interaction=TargetInteractionKind.EVALUATION,
        )
        acs = ActiveConversationState(
            created_turn=1,
            source_system_message_id=old_system.id,
            active_content={content.id: content},
            response_targets={target.id: target},
        )
    current = _message(DialogueRole.USER, "текущая реплика", 13)
    return current, model, acs, DialogueHistory(messages=history_messages)


def _core(result=None):
    client = FakeClient((result or _result()).model_dump_json())
    return CognitiveCore(client=client), client


def _payload(client):
    return json.loads(client.completions.calls[0]["messages"][1]["content"])


def _objects_with_properties(schema):
    if isinstance(schema, dict):
        if isinstance(schema.get("properties"), dict):
            yield schema
        for value in schema.values():
            yield from _objects_with_properties(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects_with_properties(value)


def test_input_view_includes_current_message_model_and_disabled_he():
    current, model, acs, history = _fixture()
    core, client = _core()

    core.propose(current, model, acs, history)

    payload = _payload(client)
    assert payload["current_user_message"] == {
        "id": current.id,
        "sequence": 13,
        "text": "текущая реплика",
    }
    assert payload["human_experience"] == {"enabled": False}
    assert {item["status"] for item in payload["human_model"]["items"]} == {
        "rejected",
        "corrected",
    }
    relation = payload["human_model"]["relations"][0]
    assert relation["status"] == "uncertain"
    assert relation["source_message_ids"] == [history.messages[0].id]


def test_previous_acs_view_and_system_prompt_are_sent():
    current, model, acs, history = _fixture()
    core, client = _core()

    core.propose(current, model, acs, history)

    payload = _payload(client)
    assert payload["previous_active_conversation_state"]["active_content"][0]["id"] == next(iter(acs.active_content))
    assert payload["previous_active_conversation_state"]["response_targets"][0]["interaction"] == "evaluation"
    assert "один полный Cognitive Discovery turn" in client.completions.calls[0]["messages"][0]["content"]


def test_system_prompt_includes_critical_conditional_contract_rules():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "CONTRACT RULES",
        "ItemOperation: ADD requires local_id",
        "existing_item_id=null",
        "non-null kind, content, evidence_origin, source_span",
        "REINFORCE requires existing_item_id",
        "CORRECT requires both local_id and",
        "RelationOperation follows the same ADD/REINFORCE/CORRECT",
        "exactly one existing ID or one local ID",
        "ProposalMaterialization with subject_kind=item",
        "outcome AMBIGUOUS names multiple distinct previous targets",
        "Every targeted ActionContent MUST be named",
    ):
        assert instruction in prompt


def test_history_closure_is_deduplicated_and_sequence_ordered():
    current, model, acs, history = _fixture()
    core, client = _core()

    core.propose(current, model, acs, history)

    selected = _payload(client)["dialogue_history"]
    sequences = [message["sequence"] for message in selected]
    ids = [message["id"] for message in selected]
    assert sequences == sorted(sequences)
    assert len(ids) == len(set(ids))
    assert history.messages[0].id in ids  # old Human Model source
    assert history.messages[1].id in ids  # previous ACS system source
    assert {message["sequence"] for message in selected}.issuperset(range(5, 13))


def test_missing_structural_history_reference_fails_explicitly():
    current, model, acs, history = _fixture()
    missing_item = next(iter(model.items.values()))
    invalid = ModelItem(
        id=missing_item.id,
        kind=missing_item.kind,
        content=missing_item.content,
        provenance=missing_item.provenance,
        status=missing_item.status,
        source_refs=(SourceReference(message_id=DialogueMessage(role=DialogueRole.USER, text="нет", sequence=99).id),),
        created_turn=1,
        updated_turn=1,
    )
    invalid_model = model.with_updates(items={invalid.id: invalid, **{key: value for key, value in model.items.items() if key != invalid.id}})
    core, client = _core()

    with pytest.raises(CognitiveCoreError, match="absent") as error:
        core.propose(current, invalid_model, acs, history)

    assert error.value.category == "history_reference_failure"
    assert client.completions.calls == []


def test_valid_output_uses_existing_contract_and_can_be_applied_manually():
    current = _message(DialogueRole.USER, "Мне тяжело", 1)
    history = DialogueHistory()
    result = _result("Что сейчас особенно трудно?")
    core, client = _core(result)

    proposed = core.propose(current, HumanModel(), None, history)
    system = _message(DialogueRole.SYSTEM, "Что сейчас особенно трудно?", 2)
    applied = apply_cognitive_turn(HumanModel(), None, history, current, system, proposed)

    assert isinstance(proposed, CognitiveTurnResult)
    assert applied.new_acs is not None
    request = client.completions.calls[0]
    assert request["response_format"]["json_schema"]["schema"] == COGNITIVE_TURN_JSON_SCHEMA
    assert "timeout" not in request


def test_strict_schema_normalization_requires_every_property_recursively():
    object_schemas = list(_objects_with_properties(COGNITIVE_TURN_JSON_SCHEMA))

    assert object_schemas
    for schema in object_schemas:
        assert schema["required"] == list(schema["properties"])


def test_schema_normalization_does_not_mutate_pydantic_source_schema():
    source_schema = CognitiveTurnResult.model_json_schema()
    original_schema = deepcopy(source_schema)

    normalized = _normalize_strict_schema(source_schema)

    assert source_schema == original_schema
    action_content = normalized["$defs"]["ActionContent"]
    assert action_content["required"] == [
        "local_id",
        "kind",
        "semantic_content",
        "model_item_refs",
        "relation_refs",
    ]


def test_adapter_has_no_runtime_or_persistence_dependency():
    source = inspect.getsource(cognitive_core_module)

    assert "ai.dialog_engine" not in source
    assert "memory.user_memory" not in source
    assert "apply_cognitive_turn(" not in source


@pytest.mark.parametrize(
    ("content", "category"),
    (("not-json", "parsing_failure"), (json.dumps({"reply_segments": []}), "validation_failure")),
)
def test_invalid_output_fails_explicitly_without_second_call(content, category):
    current, model, acs, history = _fixture()
    client = FakeClient(content)
    core = CognitiveCore(client=client)

    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)

    assert error.value.category == category
    assert len(client.completions.calls) == 1
    assert core.last_diagnostics["success"] is False
