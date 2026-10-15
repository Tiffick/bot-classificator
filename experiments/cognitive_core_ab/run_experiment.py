"""Frozen A/B/C/D Cognitive Core paraphrasing experiment.

Lab-only: never called by production.  The live path performs exactly one
OpenAI request per scheduled sample, with no retries, fallback or evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.cognitive_core import CognitiveCore  # noqa: E402
from ai.cognitive_semantic_wire import (  # noqa: E402
    SemanticWireError,
    adapt_semantic_turn,
    semantic_turn_json_schema,
)
from ai.discovery_data_model import (  # noqa: E402
    ActiveContentItem,
    ActiveContentKind,
    ActiveConversationState,
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    ResponseTarget,
    TargetInteractionKind,
    TargetReference,
    TargetSubjectKind,
)
from ai.human_experience import (  # noqa: E402
    HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION,
    select_human_experience,
)


MODEL = "gpt-5-mini-2025-08-07"
REASONING_EFFORT = "minimal"
REPEATS = 3
MAX_CALLS = 72
PILOT_CALLS = 8
MAX_COMPLETION_TOKENS = 4000
INPUT_USD_PER_MILLION = 0.25
CACHED_INPUT_USD_PER_MILLION = 0.025
OUTPUT_USD_PER_MILLION = 2.00
COST_GATE_USD = 1.00
RESULTS_ROOT = Path(__file__).with_name("results")

CONDITIONAL_PARAPHRASING_POLICY = """

USER INFORMATION REPETITION — CONDITIONAL POLICY

Default: do not repeat, summarize or paraphrase information the USER has just
provided. Treat it as input for understanding and choosing what to do next,
not as material that must be reproduced in the response.

Repetition is permitted only when the minimum necessary fragment serves one
of these concrete purposes:
1. CLARIFICATION — genuine ambiguity requires briefly restating the relevant
   fragment to verify its meaning.
2. CONTRADICTION RESOLUTION — conflicting USER statements must be explicitly
   compared to resolve the conflict.
3. NEW MEANINGFUL CONNECTION — a short reference is necessary to explain a new,
   grounded relationship, mechanism or distinction that adds understanding.
4. EMOTIONAL ACKNOWLEDGEMENT — a brief reflection of an expressed emotion is
   genuinely useful for recognition or rapport. Never invent emotion or use
   this as a routine opening.

Before finalizing, decide internally: "If I remove the repeated information,
does the response lose necessary meaning, clarity or conversational value?"
If no, remove it. If yes, retain only the minimum necessary fragment. Do not
expose this internal check.

After USER correction: update the internal understanding, abandon unsupported
interpretations, and then choose the useful next conversational move. Do not
spend the whole response acknowledging or paraphrasing the correction. Ask a
question only when it resolves meaningful uncertainty or advances the
consultation. Do not force an insight without evidence. A short answer or a
natural end to the branch is acceptable.
""".strip()

REDUCED_CONSULTANT_PROMPT = """You perform one Cognitive Discovery turn.
Use the USER's material and history to choose one useful next move. Help the
person understand a meaningful pattern, distinction or consequence; do not
merely demonstrate that their latest words were received. Accept corrections
without defensiveness and update the working understanding before selecting
the next move. Ask one focused question only when its answer can materially
change understanding or the next decision. Clarify genuine ambiguity and
resolve actual contradictions. Do not invent facts, hidden motives, emotions,
psychological causes or diagnoses. Human Experience cards are optional
professional hypotheses, not evidence or reply templates. Stay inside the
existing Product Boundary and use consent for a consultant transition.

Return only Structured Output matching the supplied strict JSON Schema.
""".strip()

MINIMAL_CONVERSATIONAL_PROMPT = """Act as a natural, careful Discovery
consultant. Use the supplied conversation and working context to write the one
most useful next reply. Add understanding or make purposeful progress rather
than merely restating the latest message. Accept corrections. Clarify only
genuine ambiguity or contradiction. Do not invent motives, emotions, causes or
diagnoses. Human Experience is optional professional knowledge, not evidence
and not a reply template. Do not design a personal solution. Return only the
user-facing reply as plain text, with no JSON and no internal terminology.
""".strip()


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    current: DialogueMessage
    human_model: HumanModel
    previous_acs: ActiveConversationState | None
    history: DialogueHistory


FROZEN_TIME = datetime(2026, 10, 11, tzinfo=timezone.utc)


def _fixed_id(prefix: str, name: str) -> str:
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, 'cognitive-core-ab/' + name).hex}"


def _msg(
    role: DialogueRole, text: str, sequence: int, fixture_key: str
) -> DialogueMessage:
    return DialogueMessage(
        id=_fixed_id("msg", fixture_key),
        role=role,
        text=text,
        sequence=sequence,
        created_at=FROZEN_TIME,
    )


def _targeted_history(
    *, user_text: str, system_text: str, interaction: TargetInteractionKind,
    content_kind: ActiveContentKind, fixture_key: str, second: bool = False,
) -> tuple[DialogueHistory, ActiveConversationState]:
    user = _msg(DialogueRole.USER, user_text, 1, fixture_key + "/history-user")
    system = _msg(DialogueRole.SYSTEM, system_text, 2, fixture_key + "/history-system")
    first_content = ActiveContentItem(
        id=_fixed_id("aci", fixture_key + "/content-1"),
        kind=content_kind, content=system_text, source_message_id=system.id
    )
    first_target = ResponseTarget(
        id=_fixed_id("rt", fixture_key + "/target-1"),
        active_content_id=first_content.id,
        subject=TargetReference(
            kind=TargetSubjectKind.ACTIVE_CONTENT, id=first_content.id
        ),
        interaction=interaction,
    )
    contents = {first_content.id: first_content}
    targets = {first_target.id: first_target}
    if second:
        other = ActiveContentItem(
            id=_fixed_id("aci", fixture_key + "/content-2"),
            kind=content_kind,
            content="Второй отдельный смысл предыдущего вопроса.",
            source_message_id=system.id,
        )
        other_target = ResponseTarget(
            id=_fixed_id("rt", fixture_key + "/target-2"),
            active_content_id=other.id,
            subject=TargetReference(
                kind=TargetSubjectKind.ACTIVE_CONTENT, id=other.id
            ),
            interaction=interaction,
        )
        contents[other.id] = other
        targets[other_target.id] = other_target
    acs = ActiveConversationState(
        id=_fixed_id("acs", fixture_key + "/acs"),
        created_turn=1,
        source_system_message_id=system.id,
        active_content=contents,
        response_targets=targets,
    )
    return DialogueHistory(messages=(user, system)), acs


def scenarios() -> tuple[Scenario, ...]:
    empty = HumanModel(updated_at=FROZEN_TIME)
    s2_history, s2_acs = _targeted_history(
        user_text=(
            "После тяжёлой недели пропускаю обед, вечером переедаю и иногда "
            "делаю паузу в попытках."
        ),
        system_text=(
            "Возможно, один вечер воспринимается как провал всей попытки. "
            "Похоже ли это на ваш опыт?"
        ),
        interaction=TargetInteractionKind.EVALUATION,
        content_kind=ActiveContentKind.RECOGNITION_OPTION,
        fixture_key="S2",
    )
    s3_history, s3_acs = _targeted_history(
        user_text=(
            "Сначала я сказал, что после переедания всё бросаю, но потом сказал, "
            "что обычно продолжаю следить за питанием."
        ),
        system_text=(
            "Эти два описания расходятся. Как точнее происходит обычно: вы "
            "продолжаете или делаете длительную паузу?"
        ),
        interaction=TargetInteractionKind.CLARIFICATION,
        content_kind=ActiveContentKind.SYSTEM_QUESTION,
        fixture_key="S3",
    )
    s5_history, s5_acs = _targeted_history(
        user_text="Мне трудно и начать изменения, и удерживать их.",
        system_text=(
            "Что сейчас важнее разобрать: трудность начать или то, почему "
            "изменения не удерживаются?"
        ),
        interaction=TargetInteractionKind.OPEN_RESPONSE,
        content_kind=ActiveContentKind.SYSTEM_QUESTION,
        fixture_key="S5",
        second=True,
    )
    return (
        Scenario(
            "S1", "Recurring difficulty",
            _msg(
                DialogueRole.USER,
                "После тяжёлой рабочей недели я пропускаю обед и вечером "
                "переедаю. Потом думаю, что всё испортил, и бросаю попытку на "
                "несколько недель. Так бывает почти каждый раз.",
                1,
                "S1/current-user",
            ),
            empty, None, DialogueHistory(),
        ),
        Scenario(
            "S2", "Rejection of a hypothesis",
            _msg(
                DialogueRole.USER,
                "Нет, для меня один вечер не означает провал, и после него я "
                "обычно продолжаю.",
                3,
                "S2/current-user",
            ),
            empty, s2_acs, s2_history,
        ),
        Scenario(
            "S3", "Resolution of contradiction",
            _msg(
                DialogueRole.USER,
                "Я сначала неправильно выразился: после переедания обычно "
                "продолжаю следить за питанием.",
                3,
                "S3/current-user",
            ),
            empty, s3_acs, s3_history,
        ),
        Scenario(
            "S4", "Insufficient evidence",
            _msg(
                DialogueRole.USER,
                "После тяжёлого разговора на работе я вечером переел. Не знаю, "
                "связано ли это вообще.",
                1,
                "S4/current-user",
            ),
            empty, None, DialogueHistory(),
        ),
        Scenario(
            "S5", "Ambiguous short answer",
            _msg(DialogueRole.USER, "Да.", 3, "S5/current-user"),
            empty, s5_acs, s5_history,
        ),
        Scenario(
            "S6", "Meaningful mechanism",
            _msg(
                DialogueRole.USER,
                "Я заметил повторяющийся момент: пока неделя спокойная, режим "
                "держится. Когда несколько дней работаю допоздна и пропускаю "
                "обед, к вечеру прихожу очень голодным и переедаю. Похоже, дело "
                "не в одном вечере, а в том, как накапливается усталость и голод.",
                1,
                "S6/current-user",
            ),
            empty, None, DialogueHistory(),
        ),
    )


def _sha(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest().upper()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def reconstruct_model(value: Any) -> Any:
    """Create an independent validated Pydantic value without deepcopy.

    Discovery models intentionally contain immutable ``FrozenDict`` mappings;
    Python's generic deepcopy tries to populate those mappings in place and is
    therefore incompatible with the persistence contract.
    """
    if value is None:
        return None
    return type(value).model_validate(value.model_dump(mode="python"))


def _offline_probe_payload() -> dict[str, Any]:
    """Minimal valid SemanticTurn used only to exercise fixture adaptation."""
    return {
        "state_patch": {
            "item_changes": [],
            "relation_changes": [],
            "proposal_materializations": [],
        },
        "reconciliation": [],
        "selected_action": {
            "decision_intent": "reflection",
            "primary_content": {
                "handle": "fixture_probe",
                "kind": "system_reflection",
                "semantic_content": "Техническая проверка семантического хода.",
                "model_item_refs": [],
                "relation_refs": [],
            },
            "additional_contents": [],
            "response_targets": [],
        },
        "reply_segments": [
            {"text": "Техническая проверка.", "realizes": ["fixture_probe"]}
        ],
    }


def validate_scenario_fixtures() -> dict[str, str]:
    """Round-trip and adapt every frozen scenario without external calls."""
    validated: dict[str, str] = {}
    for scenario in scenarios():
        current = reconstruct_model(scenario.current)
        human_model = reconstruct_model(scenario.human_model)
        previous_acs = reconstruct_model(scenario.previous_acs)
        history = reconstruct_model(scenario.history)
        assert current.model_dump(mode="json") == scenario.current.model_dump(mode="json")
        assert human_model.model_dump(mode="json") == scenario.human_model.model_dump(mode="json")
        assert history.model_dump(mode="json") == scenario.history.model_dump(mode="json")
        if scenario.previous_acs is not None:
            assert previous_acs.model_dump(mode="json") == scenario.previous_acs.model_dump(mode="json")
        result = adapt_semantic_turn(
            _offline_probe_payload(),
            current_user_text=current.text,
            human_model=human_model,
            previous_acs=previous_acs,
        )
        validated[scenario.scenario_id] = _sha(
            _json_bytes(result.model_dump(mode="json"))
        )
    return validated


def _git(command: list[str]) -> str:
    return subprocess.check_output(
        ["git", *command], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _input_payload(scenario: Scenario) -> dict[str, Any]:
    selection = select_human_experience(
        scenario.current.text,
        tuple(
            message.text
            for message in scenario.history.messages
            if message.role == DialogueRole.USER
        ),
    )
    return CognitiveCore._input_view(
        scenario.current,
        scenario.human_model,
        scenario.previous_acs,
        scenario.history,
        selection,
    )


def _production_system_prompt() -> str:
    return (
        CognitiveCore._system_prompt().rstrip()
        + "\n"
        + HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION
        + "\n"
    )


def _contract_section(production_prompt: str) -> str:
    marker = "C. SEMANTIC OUTPUT CONTRACT"
    return marker + production_prompt.split(marker, 1)[1]


def condition_prompt(condition: str, production_prompt: str) -> str:
    if condition == "A":
        return production_prompt
    if condition == "D":
        return production_prompt.rstrip() + "\n\n" + CONDITIONAL_PARAPHRASING_POLICY + "\n"
    if condition == "B":
        return (
            REDUCED_CONSULTANT_PROMPT
            + "\n\n"
            + _contract_section(CognitiveCore._system_prompt())
            + "\n"
            + HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION
            + "\n"
        )
    if condition == "C":
        return MINIMAL_CONVERSATIONAL_PROMPT
    raise ValueError(condition)


def schedule() -> list[dict[str, Any]]:
    conditions = ("A", "B", "C", "D")
    rows: list[dict[str, Any]] = []
    scenario_ids = [scenario.scenario_id for scenario in scenarios()]
    for repeat in range(1, REPEATS + 1):
        for scenario_index, scenario_id in enumerate(scenario_ids):
            offset = (repeat - 1 + scenario_index) % len(conditions)
            order = conditions[offset:] + conditions[:offset]
            for condition in order:
                rows.append(
                    {
                        "call_index": len(rows) + 1,
                        "key": f"{condition}.{scenario_id}.{repeat}",
                        "condition": condition,
                        "scenario_id": scenario_id,
                        "repeat": repeat,
                    }
                )
    assert len(rows) == MAX_CALLS
    return rows


def pilot_keys() -> tuple[str, ...]:
    return (
        "A.S1.1", "B.S1.1", "C.S1.1", "D.S1.1",
        "A.S3.1", "B.S3.1", "C.S3.1", "D.S3.1",
    )


def execution_schedule() -> list[dict[str, Any]]:
    by_key = {row["key"]: row for row in schedule()}
    pilot = [by_key[key] for key in pilot_keys()]
    pilot_set = set(pilot_keys())
    ordered = pilot + [row for row in schedule() if row["key"] not in pilot_set]
    return [
        {
            **row,
            "original_schedule_index": row["call_index"],
            "call_index": index,
        }
        for index, row in enumerate(ordered, 1)
    ]


def request_for(
    row: dict[str, Any], scenario: Scenario, production_prompt: str
) -> dict[str, Any]:
    condition = row["condition"]
    prompt = condition_prompt(condition, production_prompt)
    payload = _input_payload(scenario)
    request: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }
    if condition != "C":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_turn",
                "strict": True,
                "schema": semantic_turn_json_schema(scenario.previous_acs),
            },
        }
    return request


def estimate(production_prompt: str) -> dict[str, Any]:
    by_id = {scenario.scenario_id: scenario for scenario in scenarios()}
    estimated_input = 0
    for row in schedule():
        request = request_for(row, by_id[row["scenario_id"]], production_prompt)
        estimated_input += (len(_json_bytes(request)) + 1) // 2
    estimated_output = MAX_CALLS * MAX_COMPLETION_TOKENS
    cost = (
        estimated_input * INPUT_USD_PER_MILLION
        + estimated_output * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return {
        "method": "conservative_utf8_bytes_divided_by_2_plus_max_output",
        "estimated_input_tokens": estimated_input,
        "maximum_output_tokens": estimated_output,
        "estimated_max_cost_usd": round(cost, 6),
        "cost_gate_usd": COST_GATE_USD,
        "pricing_usd_per_million": {
            "input": INPUT_USD_PER_MILLION,
            "cached_input": CACHED_INPUT_USD_PER_MILLION,
            "output": OUTPUT_USD_PER_MILLION,
        },
    }


def _append_event(path: Path, event: dict[str, Any], previous_hash: str | None) -> str:
    core = {**event, "previous_event_sha256": previous_hash}
    event_hash = _sha(_json_bytes(core))
    row = {**core, "event_sha256": event_hash}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event_hash


def _load_checkpoint(
    path: Path,
) -> tuple[set[str], set[str], str | None, int, float]:
    completed: set[str] = set()
    attempted: set[str] = set()
    previous: str | None = None
    calls = 0
    actual_cost = 0.0
    if not path.exists():
        return completed, attempted, previous, calls, actual_cost
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        claimed = row.pop("event_sha256")
        if row.get("previous_event_sha256") != previous or _sha(_json_bytes(row)) != claimed:
            raise RuntimeError(f"checkpoint hash-chain failure at line {line_number}")
        previous = claimed
        if row["event_type"] == "API_REQUEST":
            if row["key"] in attempted:
                raise RuntimeError(f"duplicate API attempt for {row['key']}")
            attempted.add(row["key"])
            calls += 1
        if row["event_type"] == "API_RESPONSE":
            completed.add(row["key"])
            actual_cost += float(row.get("actual_cost_usd", 0.0))
    return completed, attempted, previous, calls, actual_cost


def _usage(response: Any) -> dict[str, int]:
    usage = response.usage
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _actual_cost(usage: dict[str, int]) -> float:
    uncached = max(0, usage["input_tokens"] - usage["cached_input_tokens"])
    return (
        uncached * INPUT_USD_PER_MILLION
        + usage["cached_input_tokens"] * CACHED_INPUT_USD_PER_MILLION
        + usage["output_tokens"] * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def _manifest(production_prompt: str) -> dict[str, Any]:
    diff = _git(["diff", "--binary", "HEAD", "--", "ai", "Research/Human_Experience/Weight", "tests"])
    prompts = {
        condition: condition_prompt(condition, production_prompt)
        for condition in ("A", "B", "C", "D")
    }
    return {
        "git_head": _git(["rev-parse", "HEAD"]),
        "working_patch_sha256": _sha(diff),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "repeats": REPEATS,
        "api_call_budget": MAX_CALLS,
        "pilot_call_budget": PILOT_CALLS,
        "pilot_keys": list(pilot_keys()),
        "max_retries": 0,
        "prompts_sha256": {key: _sha(value) for key, value in prompts.items()},
        "condition_d_minus_a": CONDITIONAL_PARAPHRASING_POLICY,
        "schema_sha256_by_scenario": {
            scenario.scenario_id: _sha(
                _json_bytes(semantic_turn_json_schema(scenario.previous_acs))
            )
            for scenario in scenarios()
        },
        "schedule": execution_schedule(),
        "scenario_inputs_sha256": {
            scenario.scenario_id: _sha(_json_bytes(_input_payload(scenario)))
            for scenario in scenarios()
        },
        "estimate": estimate(production_prompt),
    }


def dry_run() -> None:
    production_prompt = _production_system_prompt()
    manifest = _manifest(production_prompt)
    assert manifest["api_call_budget"] == 72
    assert manifest["pilot_call_budget"] == 8
    assert len(execution_schedule()) == 72
    assert len(set(row["key"] for row in execution_schedule())) == 72
    assert [row["call_index"] for row in execution_schedule()] == list(range(1, 73))
    first_input_hashes = {
        scenario.scenario_id: _sha(_json_bytes(_input_payload(scenario)))
        for scenario in scenarios()
    }
    second_input_hashes = {
        scenario.scenario_id: _sha(_json_bytes(_input_payload(scenario)))
        for scenario in scenarios()
    }
    assert first_input_hashes == second_input_hashes
    fixture_probe_hashes = validate_scenario_fixtures()
    assert set(fixture_probe_hashes) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert manifest["estimate"]["estimated_max_cost_usd"] <= COST_GATE_USD
    assert condition_prompt("D", production_prompt) == (
        production_prompt.rstrip() + "\n\n" + CONDITIONAL_PARAPHRASING_POLICY + "\n"
    )
    for scenario in scenarios():
        payload = _input_payload(scenario)
        assert payload["human_experience"]["enabled"] is True
        for condition in ("A", "B", "D"):
            request = request_for(
                {"condition": condition}, scenario, production_prompt
            )
            assert request["response_format"]["json_schema"]["strict"] is True
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"fixture_probe_sha256": fixture_probe_hashes}, indent=2))
    print("DRY_RUN_PASS API_CALLS=0")


def _pilot_status(checkpoint: Path) -> tuple[bool, list[str]]:
    if not checkpoint.exists():
        return False, ["pilot checkpoint is absent"]
    rows = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    responses = {
        row["key"]: row
        for row in rows
        if row["event_type"] == "API_RESPONSE" and row["key"] in pilot_keys()
    }
    validations = {
        row["key"]: row["validation"]
        for row in rows
        if row["event_type"] == "SEMANTIC_VALIDATION"
        and row["key"] in pilot_keys()
    }
    errors: list[str] = []
    for key in pilot_keys():
        row = responses.get(key)
        if row is None:
            errors.append(f"{key}: missing API_RESPONSE")
            continue
        choice = row["raw_response"]["choices"][0]
        if not row.get("text", "").strip():
            errors.append(f"{key}: empty visible output")
        if choice.get("finish_reason") == "length":
            errors.append(f"{key}: output token exhaustion")
        validation = validations.get(key)
        if validation is None:
            errors.append(f"{key}: missing semantic validation event")
            continue
        if row["condition"] in {"A", "B", "D"} and not validation.get("valid"):
            errors.append(f"{key}: invalid SemanticTurn")
        if row["condition"] == "C" and not validation.get("plain_text_nonempty"):
            errors.append(f"{key}: empty plain text")
    failures = [row for row in rows if row["event_type"] == "API_FAILURE"]
    if failures:
        errors.append("pilot contains API_FAILURE")
    return not errors, errors


def _conservative_call_cost(request: dict[str, Any]) -> float:
    estimated_input = (len(_json_bytes(request)) + 1) // 2
    return (
        estimated_input * INPUT_USD_PER_MILLION
        + MAX_COMPLETION_TOKENS * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def run(*, pilot_only: bool) -> None:
    load_dotenv(ROOT / ".env")
    production_prompt = _production_system_prompt()
    manifest = _manifest(production_prompt)
    current_manifest = {
        **manifest,
        "experiment_revision": "corrected-v3-durable-raw-first",
        "persistence_order": [
            "API_REQUEST",
            "API_RESPONSE",
            "SEMANTIC_VALIDATION",
        ],
    }
    if manifest["estimate"]["estimated_max_cost_usd"] > COST_GATE_USD:
        raise RuntimeError("cost gate exceeded before first API call")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path for path in RESULTS_ROOT.iterdir()
        if path.is_dir() and (path / "checkpoint.jsonl").exists()
        and (path / "manifest.json").exists()
        and json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        .get("experiment_revision") == "corrected-v3-durable-raw-first"
        and not (path / "RUN_COMPLETED").exists()
        and not (path / "PILOT_FAILED").exists()
        and not (path / "PILOT_FAILED.json").exists()
    )
    if existing:
        run_dir = existing[-1]
        saved_manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if saved_manifest != current_manifest:
            raise RuntimeError("current frozen configuration does not match resumable run")
    else:
        run_id = (
            "cc-abcd-v3-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        run_dir = RESULTS_ROOT / run_id
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                current_manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "prompts.json").write_text(
            json.dumps(
                {
                    condition: condition_prompt(condition, production_prompt)
                    for condition in ("A", "B", "C", "D")
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    checkpoint = run_dir / "checkpoint.jsonl"
    completed, attempted, previous_hash, calls, actual_cost = _load_checkpoint(checkpoint)
    if calls > MAX_CALLS:
        raise RuntimeError("checkpoint exceeds hard API budget")
    if attempted - completed:
        raise RuntimeError("checkpoint contains an attempted call without a response; retry forbidden")
    if calls >= PILOT_CALLS:
        passed, errors = _pilot_status(checkpoint)
        if not passed:
            raise RuntimeError("pilot failed: " + "; ".join(errors))
    if not pilot_only and calls < PILOT_CALLS:
        raise RuntimeError("full run requires a completed passing pilot")
    client = OpenAI(max_retries=0)
    by_id = {scenario.scenario_id: scenario for scenario in scenarios()}

    target_calls = PILOT_CALLS if pilot_only else MAX_CALLS
    for row in execution_schedule():
        if row["key"] not in set(pilot_keys()) and calls < PILOT_CALLS:
            raise RuntimeError("qualification gate violated before full schedule")
        if row["key"] in completed:
            continue
        if calls >= target_calls:
            break
        scenario = by_id[row["scenario_id"]]
        request = request_for(row, scenario, production_prompt)
        if actual_cost + _conservative_call_cost(request) > COST_GATE_USD:
            raise RuntimeError("USD 1.00 cost gate would be exceeded by next call")
        request_hash = _sha(_json_bytes(request))
        previous_hash = _append_event(
            checkpoint,
            {
                "event_type": "API_REQUEST",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "key": row["key"],
                "call_index": row["call_index"],
                "request_sha256": request_hash,
            },
            previous_hash,
        )
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**request)
        except Exception as error:
            _append_event(
                checkpoint,
                {
                    "event_type": "API_FAILURE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "key": row["key"],
                    "call_index": row["call_index"],
                    "request_sha256": request_hash,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                previous_hash,
            )
            raise
        latency_seconds = time.perf_counter() - started
        calls += 1
        raw = response.model_dump(mode="json")
        text = response.choices[0].message.content or ""
        usage = _usage(response)
        response_cost = _actual_cost(usage)
        actual_cost += response_cost
        previous_hash = _append_event(
            checkpoint,
            {
                "event_type": "API_RESPONSE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "key": row["key"],
                "call_index": row["call_index"],
                "condition": row["condition"],
                "scenario_id": row["scenario_id"],
                "repeat": row["repeat"],
                "request_sha256": request_hash,
                "actual_model": response.model,
                "raw_response": raw,
                "text": text,
                "usage": usage,
                "latency_seconds": latency_seconds,
                "actual_cost_usd": response_cost,
            },
            previous_hash,
        )

        validation: dict[str, Any]
        if row["condition"] == "C":
            validation = {"applicable": False, "plain_text_nonempty": bool(text.strip())}
        else:
            try:
                payload = json.loads(text)
                adapted = adapt_semantic_turn(
                    payload,
                    current_user_text=scenario.current.text,
                    human_model=reconstruct_model(scenario.human_model),
                    previous_acs=reconstruct_model(scenario.previous_acs),
                )
                validation = {
                    "applicable": True,
                    "valid": True,
                    "cognitive_turn_result": adapted.model_dump(mode="json"),
                    "reply": "".join(segment.text for segment in adapted.reply_segments),
                }
            except Exception as error:
                validation = {
                    "applicable": True,
                    "valid": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        previous_hash = _append_event(
            checkpoint,
            {
                "event_type": "SEMANTIC_VALIDATION",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "key": row["key"],
                "call_index": row["call_index"],
                "condition": row["condition"],
                "scenario_id": row["scenario_id"],
                "repeat": row["repeat"],
                "request_sha256": request_hash,
                "validation": validation,
            },
            previous_hash,
        )
        print(f"{calls}/{MAX_CALLS} {row['key']} complete", flush=True)

    completed, attempted, previous_hash, calls, actual_cost = _load_checkpoint(checkpoint)
    if pilot_only:
        passed, errors = _pilot_status(checkpoint)
        marker = run_dir / ("PILOT_PASSED" if passed else "PILOT_FAILED")
        marker.write_text(
            json.dumps(
                {"last_event_sha256": previous_hash, "errors": errors},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"PILOT_{'PASSED' if passed else 'FAILED'} calls={calls} "
            f"cost_usd={actual_cost:.8f} path={run_dir}"
        )
        if not passed:
            raise RuntimeError("pilot failed: " + "; ".join(errors))
        return
    if calls != MAX_CALLS or len(completed) != MAX_CALLS:
        raise RuntimeError("run stopped before the frozen schedule completed")
    (run_dir / "RUN_COMPLETED").write_text(previous_hash or "", encoding="ascii")
    print(f"RUN_COMPLETED calls={calls} path={run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        run(pilot_only=args.pilot)


if __name__ == "__main__":
    main()
