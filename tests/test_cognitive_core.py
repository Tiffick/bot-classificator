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
    DecisionIntent,
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
        decision_intent=DecisionIntent.HUMAN_DISCOVERY,
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
        "non-null kind, content, evidence_origin, source_quote",
        "REINFORCE requires existing_item_id",
        "CORRECT requires both local_id and",
        "RelationOperation follows the same ADD/REINFORCE/CORRECT",
        "exactly one existing ID or one local ID",
        "ProposalMaterialization with subject_kind=item",
        "outcome AMBIGUOUS names multiple distinct previous targets",
        "Every targeted ActionContent MUST be named",
    ):
        assert instruction in prompt


def test_system_prompt_requires_local_namespaces_and_exact_reference_reuse():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for field, namespace in (
        ("ItemOperation.local_id", "item"),
        ("RelationOperation.local_id", "relation"),
        ("ActionContent.local_id", "content"),
        ("ActionTarget.local_id", "target"),
        ("ReplySegment.local_id", "segment"),
    ):
        assert f"{field} MUST use {namespace}:<name>" in prompt

    assert "References MUST reuse the exact declared local ID including its namespace" in prompt
    for reference in (
        "active_content_local_id",
        "subject.active_content_local_id",
        "realizes_action_content_ids",
        "ItemReference.local_item_id",
        "RelationReference.local_relation_id",
    ):
        assert reference in prompt
    assert "exactly content:question_1, not question_1, ac1 or another alias" in prompt
    assert "rt_ / aci_ IDs from previous ACS, never new local IDs" in prompt


def test_system_prompt_defines_item_and_relation_reference_ownership():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "when an item or relation already exists in the supplied HumanModel",
        "exact persistent ID into existing_item_id (mi_*) or existing_relation_id (rel_*)",
        "set the corresponding local_*_id to null",
        "Never create a local alias for an existing persistent object",
        "local_item_id is allowed only when that exact item:* is declared by an ADD or CORRECT ItemOperation in the CURRENT state_patch",
        "local_relation_id is allowed only when that exact relation:* is declared by an ADD or CORRECT RelationOperation in the CURRENT state_patch",
        "every local reference must resolve within this same CognitiveTurnResult",
        "confirmation, local reflection, working-picture Reflection",
        "creates no new ModelItem/Relation",
        "exact existing mi_*/rel_* ID",
        "leave the corresponding refs list empty",
    ):
        assert instruction in prompt


def test_system_prompt_explains_active_content_target_ownership():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "ActionTarget.active_content_local_id belongs only to an ActionContent declared in the CURRENT SystemAction",
        "content that carries/creates the target",
        "subject.active_content_local_id also belongs only to the CURRENT SystemAction",
        "semantic content about which a response is expected",
        "neither may contain a persistent aci_ ID from previous ACS",
        "Previous ACS is addressed only through reconciliation",
        "Owner and subject may match, but need not",
        "target owner content:question and target subject content:reflection",
    ):
        assert instruction in prompt


def test_system_prompt_distinguishes_consent_from_open_response():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "OPEN_RESPONSE asks the user for their own open material or answer",
        "It is not a request to accept or decline a proposed action",
        "CLARIFICATION asks the user to clarify the addressed meaning",
        "EVALUATION asks the user to evaluate a system-proposed version",
        "CONSENT asks the user to accept or decline a concrete proposed action",
        "ResponseTarget MUST use interaction=CONSENT, not OPEN_RESPONSE",
        "semantic examples, not keyword triggers",
        "CONSENTED and DECLINED resolve only a CONSENT target",
        "Do not use REFUSED as a substitute for DECLINED",
        "AMBIGUOUS means the user's response cannot be assigned to one specific previous target",
    ):
        assert instruction in prompt


def test_system_prompt_defines_dynamic_value_and_sufficiency_boundary():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "compare the expected value of eligible actions inside the current product boundary",
        "materially change useful understanding of the person",
        "a substantial mechanism of the problem",
        "the next eligible Decision",
        "the other Discovery line, Reflection, Transition, Respect Pause / Refusal or Stop Exploration",
        "Human / Meaningful Change and Mechanism / Expertise are both full Discovery lines",
        "no mandatory order",
        "required slots, completion checklist or sufficiency score",
        "Local sufficiency means stop or leave THAT branch",
        "It never means that permission for a concrete solution has been earned",
    ):
        assert instruction in prompt


def test_system_prompt_defines_operational_candidate_selection_protocol():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "OPERATIONAL CANDIDATE-SELECTION PROTOCOL",
        "A confirmed \"do not know / did not observe / cannot determine\" may complete Discovery on that semantic axis",
        "do not force the user to resolve an uncertainty they cannot resolve",
        "CLOSE AN EXHAUSTED SEMANTIC AXIS",
        "merely splits an already confirmed uncertainty into more details has low value",
        "UNKNOWN does not mean rephrase the same question",
        "GENERATE DISTINCT ELIGIBLE CANDIDATES",
        "only materially different, context-appropriate candidates among the existing DecisionIntent values",
        "Do not continue a sufficiently understood branch merely because more detail can be obtained",
        "no fixed order, mandatory alternation or balancing quota",
        "Local confirmation / repair checks one newly heard element only",
        "Working-picture Reflection connects multiple supported elements",
        "another question has lower expected value, prefer a working-picture Reflection",
        "Detail useful mainly for concrete solution design",
        "does not justify continuing Discovery",
        "Do not output the comparison, reasoning, rationale, scores or any additional field",
    ):
        assert instruction in prompt

    assert set(COGNITIVE_TURN_JSON_SCHEMA["properties"]) == {
        "decision_intent",
        "state_patch",
        "reconciliation",
        "system_action",
        "reply_segments",
    }


def test_system_prompt_enforces_solution_action_product_boundary():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "First check whether the proposed action is eligible",
        "Transition to a human consultant with meaningful consent",
        "does not select or tune a concrete solution for the user",
        "concrete behavioral experiment, recommendation, plan, practice",
        "beginning of solution-option selection by the Core",
        "Apparent usefulness and local mechanism sufficiency do not make such content eligible",
        "SYSTEM_PROPOSAL remains available for semantically valid in-scope content",
        "The action kind itself does not authorize solution content",
        "Answer a direct user question normally",
        "does not automatically authorize program or solution selection",
        "Reflection is a value-based candidate",
        "not a ritual after a fixed number of turns",
    ):
        assert instruction in prompt


def test_system_prompt_respects_declined_without_inventing_problem_barrier():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "After DECLINED, the declined action loses value",
        "Do not automatically offer a neighbouring variant of the same solution",
        "Give Respect Pause / Refusal and Stop Exploration high priority",
        "Ask why only when the answer can materially change an eligible in-scope action without pressure",
        "Declining a system proposal does not automatically create a BARRIER",
        "Reconciliation is sufficient unless the user also provides a substantive constraint",
        "only that independent material may support a BARRIER",
    ):
        assert instruction in prompt


def test_system_prompt_defines_ephemeral_decision_intents_and_structural_limits():
    prompt = " ".join(CognitiveCore._system_prompt().split())

    for instruction in (
        "Choose Decision before Response",
        "architectural function of the already selected next Cognitive Cycle action",
        "not the topic of its text",
        "HUMAN_DISCOVERY continues understanding the person and Meaningful Change",
        "MECHANISM_DISCOVERY continues understanding Previous Attempts, Barriers",
        "REFLECTION returns a coherent working picture",
        "RECOGNITION offers context-close working versions or options",
        "TRANSITION explicitly offers movement beyond the current Discovery boundary",
        "RESPECT_PAUSE_OR_REFUSAL does not continue a declined or unwanted direction",
        "STOP_EXPLORATION ends the current Discovery",
        "ANSWER_USER_QUESTION answers a direct user question",
        "decision_intent does not prove semantic Product Boundary compliance",
        "Never disguise a concrete experiment, recommendation, plan or practice as Discovery",
        "A CONSENT target subject must be current SYSTEM_PROPOSAL or SYSTEM_TRANSITION",
        "SYSTEM_REFLECTION is evaluated, not accepted or declined",
    ):
        assert instruction in prompt


def test_api_schema_requires_all_decision_intent_values():
    properties = COGNITIVE_TURN_JSON_SCHEMA["properties"]
    assert "decision_intent" in properties
    assert "decision_intent" in COGNITIVE_TURN_JSON_SCHEMA["required"]
    decision_schema = properties["decision_intent"]
    definition_name = decision_schema["$ref"].rsplit("/", 1)[-1]
    assert set(COGNITIVE_TURN_JSON_SCHEMA["$defs"][definition_name]["enum"]) == {
        intent.value for intent in DecisionIntent
    }


def test_core_returns_explicit_decision_after_exactly_one_llm_call():
    current, model, acs, history = _fixture()
    core, client = _core()

    result = core.propose(current, model, acs, history)

    assert result.decision_intent is DecisionIntent.HUMAN_DISCOVERY
    assert len(client.completions.calls) == 1


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


def test_system_prompt_requires_exact_unique_current_message_quote():
    prompt = " ".join(CognitiveCore._system_prompt().split())
    assert "exact verbatim substring of the CURRENT USER message" in prompt
    assert "exactly one exact occurrence" in prompt
    assert "Do not paraphrase, normalize or invent" in prompt
    assert "Python computes evidence coordinates" in prompt
    assert "Reconciliation source_quote may be null" in prompt
    for obsolete in ("source_span", "char_start", "char_end", "count characters"):
        assert obsolete not in prompt


def _quote_payload(branch, quote, model, acs):
    payload = _result().model_dump(mode="json")
    if branch == "reconciliation":
        entry = {
            "previous_response_target_ids": [next(iter(acs.response_targets))],
            "outcome": "supported",
            "source_quote": quote,
        }
        payload[branch] = [entry]
    else:
        field, object_id = (
            ("existing_item_id", next(iter(model.items)))
            if branch == "item_operations"
            else ("existing_relation_id", next(iter(model.relations)))
        )
        entry = {
            "operation": "reinforce",
            field: object_id,
            "evidence_origin": "current_user_material",
            "source_quote": quote,
        }
        payload["state_patch"][branch] = [entry]
    return payload, entry


@pytest.mark.parametrize("branch", ["item_operations", "relation_operations", "reconciliation"])
@pytest.mark.parametrize("overflow", [0, 1])
def test_current_message_span_boundary_after_single_response(branch, overflow, monkeypatch):
    current, model, acs, history = _fixture()
    payload, _ = _quote_payload(branch, current.text, model, acs)
    resolve = cognitive_core_module._resolve_source_quotes
    expected = CognitiveTurnResult.model_validate(resolve(payload, current.text))
    path = f"{'' if branch == 'reconciliation' else 'state_patch.'}{branch}[0].source_span"
    if overflow:
        # Simulate an adapter defect: the independent numeric boundary must
        # still reject a Pydantic-valid, but out-of-message span.
        def faulty_resolve(raw, text):
            converted = resolve(raw, text)
            container = converted if branch == "reconciliation" else converted["state_patch"]
            container[branch][0]["source_span"]["char_end"] += overflow
            return converted
        monkeypatch.setattr(cognitive_core_module, "_resolve_source_quotes", faulty_resolve)
    client = FakeClient(json.dumps(payload))
    core = CognitiveCore(client)
    original_input = current.model_dump()
    original_content = client.completions.content

    if overflow:
        with pytest.raises(CognitiveCoreError) as error:
            core.propose(current, model, acs, history)
        assert error.value.category == "validation_failure"
        assert path in str(error.value)
        assert core.last_diagnostics["success"] is False
        assert core.last_diagnostics["failure_category"] == "validation_failure"
    else:
        result = core.propose(current, model, acs, history)
        assert result == expected
        assert core.last_diagnostics["success"] is True

    assert len(client.completions.calls) == 1
    assert current.model_dump() == original_input
    assert client.completions.content == original_content


@pytest.mark.parametrize("branch", ["item_operations", "relation_operations", "reconciliation"])
@pytest.mark.parametrize("text,quote", [
    ("Начало. Хочется есть вечером.", "Хочется есть вечером."),
    ("🙂 е\u0308ж — текст!", "е\u0308ж"),
    (" a  b ", " a  b "),
])
def test_quote_resolves_exactly_without_mutation(branch, text, quote):
    _, model, acs, history = _fixture()
    current = _message(DialogueRole.USER, text, 13)
    payload, _ = _quote_payload(branch, quote, model, acs)
    original = deepcopy(payload)
    converted = cognitive_core_module._resolve_source_quotes(payload, text)
    assert payload == original
    client = FakeClient(json.dumps(payload))
    result = CognitiveCore(client).propose(current, model, acs, history)
    assert isinstance(result, CognitiveTurnResult)
    assert result == CognitiveTurnResult.model_validate(converted)
    rows = result.reconciliation if branch == "reconciliation" else getattr(result.state_patch, branch)
    assert rows[0].source_span.char_start == text.index(quote)
    assert rows[0].source_span.char_end == text.index(quote) + len(quote)
    assert "source_quote" not in result.model_dump_json()
    assert current.text == text
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("text,quote,reason", [
    ("Текст", "нет", "exact matches=0"),
    ("да, да", "да", "exact matches=2"),
    ("aaaa", "aa", "exact matches=3"),
    ("Текст", "", "non-empty string"),
    ("Текст", None, "non-empty string"),
    ("Текст", 123, "non-empty string"),
    ("Текст", "текст", "exact matches=0"),
    ("a  b", "a b", "exact matches=0"),
    ("é", "e\u0301", "exact matches=0"),
    ("Текст.", "Текст!", "exact matches=0"),
    ("Текст", " Текст ", "exact matches=0"),
])
def test_bad_quote_fails_after_one_call(text, quote, reason):
    _, model, acs, history = _fixture()
    current = _message(DialogueRole.USER, text, 13)
    payload, _ = _quote_payload("item_operations", quote, model, acs)
    client = FakeClient(json.dumps(payload))
    core = CognitiveCore(client)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert "state_patch.item_operations[0].source_quote" in str(error.value)
    assert reason in str(error.value)
    assert core.last_diagnostics["failure_category"] == "validation_failure"
    assert core.last_diagnostics["success"] is False
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("branch", ["item_operations", "relation_operations"])
@pytest.mark.parametrize("operation", ["add", "reinforce", "correct"])
@pytest.mark.parametrize("omit_quote", [False, True])
def test_operation_quote_requirement(branch, operation, omit_quote):
    current, model, acs, history = _fixture()
    payload, entry = _quote_payload(branch, current.text, model, acs)
    entry["operation"] = operation
    if operation != "reinforce":
        if branch == "item_operations":
            entry.update(local_id="item:new", kind="experience", content=current.text)
            if operation == "add":
                entry.pop("existing_item_id")
        else:
            first, second = model.items
            entry.update(
                local_id="relation:new",
                source_items=[{"existing_item_id": first}],
                target_items=[{"existing_item_id": second}],
                meaning="выраженная пользователем связь",
            )
            if operation == "add":
                entry.pop("existing_relation_id")
    # All other fields satisfy the ordinary contract, isolating evidence.
    expected = CognitiveTurnResult.model_validate(
        cognitive_core_module._resolve_source_quotes(payload, current.text)
    )
    if omit_quote:
        del entry["source_quote"]
    client = FakeClient(json.dumps(payload))
    core = CognitiveCore(client)
    if omit_quote:
        with pytest.raises(CognitiveCoreError) as error:
            core.propose(current, model, acs, history)
        assert error.value.category == "validation_failure"
        assert f"state_patch.{branch}[0].source_quote: required quote is missing" in str(error.value)
    else:
        assert core.propose(current, model, acs, history) == expected
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("omit", [False, True])
def test_reconciliation_quote_can_be_null_or_absent(omit):
    current, model, acs, history = _fixture()
    payload, entry = _quote_payload("reconciliation", None, model, acs)
    if omit:
        del entry["source_quote"]
    client = FakeClient(json.dumps(payload))
    result = CognitiveCore(client).propose(current, model, acs, history)
    assert result.reconciliation[0].source_span is None
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("branch", ["item_operations", "relation_operations", "reconciliation"])
def test_api_numeric_spans_cannot_bypass_quote_resolution(branch):
    current, model, acs, history = _fixture()
    payload, entry = _quote_payload(branch, current.text, model, acs)
    entry["source_span"] = {"char_start": 0, "char_end": len(current.text)}
    client = FakeClient(json.dumps(payload))
    with pytest.raises(CognitiveCoreError) as error:
        CognitiveCore(client).propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert "numeric evidence is not accepted" in str(error.value)
    assert len(client.completions.calls) == 1


def test_api_schema_projects_quotes_without_changing_internal_contract():
    internal = CognitiveTurnResult.model_json_schema()
    original = deepcopy(internal)
    schema = cognitive_core_module._api_facing_schema()
    assert CognitiveTurnResult.model_json_schema() == original
    assert "SourceSpan" in internal["$defs"]
    for name in ("ItemOperation", "RelationOperation", "TargetResolution"):
        definition = schema["$defs"][name]
        assert "source_quote" in definition["required"]
        assert "source_span" in internal["$defs"][name]["properties"]
        quote = definition["properties"]["source_quote"]
        if name == "TargetResolution":
            assert {"type": "null"} in quote["anyOf"]
        else:
            assert quote == {"type": "string", "minLength": 1}
    for removed in ("source_span", "SourceSpan", "char_start", "char_end"):
        assert removed not in json.dumps(schema)
    for definition in _objects_with_properties(schema):
        assert definition["additionalProperties"] is False
        assert definition["required"] == list(definition["properties"])


def test_resolved_quotes_in_all_paths_can_be_applied_without_contract_changes():
    current, model, acs, history = _fixture()
    payload = _result().model_dump(mode="json")
    for branch in ("item_operations", "relation_operations", "reconciliation"):
        _, entry = _quote_payload(branch, "реплика", model, acs)
        if branch == "reconciliation":
            payload[branch] = [entry]
        else:
            payload["state_patch"][branch] = [entry]
    client = FakeClient(json.dumps(payload))
    result = CognitiveCore(client).propose(current, model, acs, history)
    system = _message(DialogueRole.SYSTEM, "".join(s.text for s in result.reply_segments), 14)
    applied = apply_cognitive_turn(model, acs, history, current, system, result)
    expected = SourceReference(
        message_id=current.id,
        char_start=current.text.index("реплика"),
        char_end=len(current.text),
    )
    assert expected in applied.updated_human_model.items[next(iter(model.items))].source_refs
    assert expected in applied.updated_human_model.relations[next(iter(model.relations))].source_refs
    assert result.reconciliation[0].source_span.char_end == len(current.text)
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("payload,path", [
    ([], "$"),
    ({"state_patch": None}, "state_patch"),
    ({"state_patch": {"item_operations": None}}, "state_patch.item_operations"),
    ({"reconciliation": [None]}, "reconciliation[0]"),
])
def test_malformed_quote_container_is_validation_failure(payload, path):
    current, model, acs, history = _fixture()
    client = FakeClient(json.dumps(payload))
    core = CognitiveCore(client)
    with pytest.raises(CognitiveCoreError) as error:
        core.propose(current, model, acs, history)
    assert error.value.category == "validation_failure"
    assert str(error.value).startswith(path + ":")
    assert core.last_diagnostics["success"] is False
    assert len(client.completions.calls) == 1


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


def test_api_schema_restricts_active_content_local_references_to_current_namespace():
    definitions = COGNITIVE_TURN_JSON_SCHEMA["$defs"]
    target_reference = definitions["ActionTarget"]["properties"][
        "active_content_local_id"
    ]
    subject_reference = definitions["ActionSubjectReference"]["properties"][
        "active_content_local_id"
    ]
    subject_string = next(
        branch for branch in subject_reference["anyOf"] if branch.get("type") == "string"
    )

    expected_pattern = r"^content:[A-Za-z0-9][A-Za-z0-9._-]*$"
    assert target_reference["pattern"] == expected_pattern
    assert subject_string["pattern"] == expected_pattern
    assert "current SystemAction" in target_reference["description"]
    assert "current SystemAction" in subject_reference["description"]


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
