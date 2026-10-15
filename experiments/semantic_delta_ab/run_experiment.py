"""Frozen lab-only A/B experiment for Cognitive Core semantic delta.

Dry-run is the default and performs no network calls. Live execution requires
both ``--execute`` and the exact authorization token. There are no retries,
fallbacks, adaptive calls or evaluator calls.
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
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.cognitive_core import CognitiveCore  # noqa: E402
from ai.cognitive_semantic_wire import (  # noqa: E402
    SemanticTurn,
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
    ItemStatus,
    ModelItem,
    ModelItemKind,
    Provenance,
    ResponseTarget,
    SourceReference,
    TargetInteractionKind,
    TargetReference,
    TargetSubjectKind,
)
from ai.human_experience import (  # noqa: E402
    HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION,
    select_human_experience,
)
from experiments.cognitive_core_ab.run_experiment import (  # noqa: E402
    Scenario,
    _input_payload as legacy_input_payload,
    scenarios as legacy_scenarios,
)
from experiments.semantic_reference_ab.run_experiment import _schema_errors  # noqa: E402

MODEL = "gpt-5-mini-2025-08-07"
REASONING_EFFORT = "minimal"
MAX_COMPLETION_TOKENS = 4000
MAX_RETRIES = 0
REPEATS = 3
CONDITIONS = ("A", "B")
SCENARIO_IDS = ("S1", "S3", "S4", "S5", "S6")
MAX_CALLS = len(CONDITIONS) * len(SCENARIO_IDS) * REPEATS
AUTHORIZATION = "I_AUTHORIZE_EXACTLY_30_SEMANTIC_DELTA_CALLS"
MANIFEST_PATH = PACKAGE / "manifest.json"
PACKAGE_HASHES_PATH = PACKAGE / "package_hashes.json"
APPENDIX_PATH = PACKAGE / "treatment_appendix.txt"
SCENARIOS_PATH = PACKAGE / "scenarios.json"
RESULTS_ROOT = PACKAGE / "results"
FROZEN_TIME = datetime(2026, 10, 13, tzinfo=timezone.utc)


def _sha(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest().upper()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _fixed_id(prefix: str, name: str) -> str:
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, 'semantic-delta-ab/' + name).hex}"


def _msg(role: DialogueRole, text: str, sequence: int, key: str) -> DialogueMessage:
    return DialogueMessage(
        id=_fixed_id("msg", key),
        role=role,
        text=text,
        sequence=sequence,
        created_at=FROZEN_TIME,
    )


def _item(
    *, kind: ModelItemKind, content: str, source: DialogueMessage, key: str
) -> ModelItem:
    return ModelItem(
        id=_fixed_id("mi", key),
        kind=kind,
        content=content,
        provenance=Provenance.USER_PROVIDED,
        status=ItemStatus.RECORDED,
        source_refs=(SourceReference(message_id=source.id),),
        created_turn=1,
        updated_turn=1,
    )


def _targeted_fixture(
    *,
    scenario_id: str,
    history_user_text: str,
    system_text: str,
    current_text: str,
    item_kind: ModelItemKind,
    item_content: str,
    content_kind: ActiveContentKind,
    interaction: TargetInteractionKind,
) -> Scenario:
    user = _msg(DialogueRole.USER, history_user_text, 1, scenario_id + "/history-user")
    system = _msg(DialogueRole.SYSTEM, system_text, 2, scenario_id + "/history-system")
    current = _msg(DialogueRole.USER, current_text, 3, scenario_id + "/current-user")
    item = _item(
        kind=item_kind,
        content=item_content,
        source=user,
        key=scenario_id + "/item",
    )
    content = ActiveContentItem(
        id=_fixed_id("aci", scenario_id + "/content"),
        kind=content_kind,
        content=system_text,
        model_item_ids=(item.id,),
        source_message_id=system.id,
    )
    target = ResponseTarget(
        id=_fixed_id("rt", scenario_id + "/target"),
        active_content_id=content.id,
        subject=TargetReference(kind=TargetSubjectKind.ACTIVE_CONTENT, id=content.id),
        interaction=interaction,
    )
    acs = ActiveConversationState(
        id=_fixed_id("acs", scenario_id + "/acs"),
        created_turn=1,
        source_system_message_id=system.id,
        active_content={content.id: content},
        response_targets={target.id: target},
    )
    return Scenario(
        scenario_id=scenario_id,
        title={
            "S4": "Insufficient information for a mechanism",
            "S5": "Simple factual correction",
            "S6": "Useful short reflection",
        }[scenario_id],
        current=current,
        human_model=HumanModel(
            items={item.id: item}, relations={}, updated_at=FROZEN_TIME
        ),
        previous_acs=acs,
        history=DialogueHistory(messages=(user, system)),
    )


def scenarios() -> tuple[Scenario, ...]:
    legacy = {row.scenario_id: row for row in legacy_scenarios()}
    s4 = _targeted_fixture(
        scenario_id="S4",
        history_user_text=(
            "Я хочу похудеть, но пока не понимаю, почему вечером иногда ем "
            "больше, чем собирался."
        ),
        system_text=(
            "Расскажите про последний такой вечер: что вы заметили в тот день "
            "перед тем, как съели больше?"
        ),
        current_text=(
            "Вчера вечером снова съел больше, чем собирался. День был обычный, "
            "и я не заметил, что именно к этому привело."
        ),
        item_kind=ModelItemKind.STATED_PROBLEM,
        item_content="Иногда вечером съедает больше, чем собирался, и не понимает почему.",
        content_kind=ActiveContentKind.SYSTEM_QUESTION,
        interaction=TargetInteractionKind.OPEN_RESPONSE,
    )
    s5 = _targeted_fixture(
        scenario_id="S5",
        history_user_text=(
            "Обычно после переедания я на несколько дней перестаю следить за "
            "питанием."
        ),
        system_text=(
            "Получается, после такого вечера вы обычно выпадаете из привычного "
            "режима на несколько дней — это верное описание?"
        ),
        current_text=(
            "Нет, я неточно сказал: на следующий день я обычно возвращаюсь к "
            "привычному режиму."
        ),
        item_kind=ModelItemKind.USER_INTERPRETATION,
        item_content="После переедания на несколько дней перестаёт следить за питанием.",
        content_kind=ActiveContentKind.SYSTEM_REFLECTION,
        interaction=TargetInteractionKind.EVALUATION,
    )
    s6 = _targeted_fixture(
        scenario_id="S6",
        history_user_text=(
            "После набора веса мне стало тяжело смотреть на свои фотографии."
        ),
        system_text="Что именно в этих фотографиях задевает вас сильнее всего?",
        current_text=(
            "Я как будто перестал узнавать себя. Это не про конкретную цифру на "
            "весах — мне просто тяжело видеть, насколько я изменился."
        ),
        item_kind=ModelItemKind.EXPERIENCE,
        item_content="После набора веса тяжело смотреть на свои фотографии.",
        content_kind=ActiveContentKind.SYSTEM_QUESTION,
        interaction=TargetInteractionKind.OPEN_RESPONSE,
    )
    return (legacy["S1"], legacy["S3"], s4, s5, s6)


def _production_prompt() -> str:
    return (
        CognitiveCore._system_prompt().rstrip()
        + "\n"
        + HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION
        + "\n"
    )


def condition_prompts() -> dict[str, str]:
    control = _production_prompt()
    appendix = APPENDIX_PATH.read_text(encoding="utf-8").strip()
    treatment = control.rstrip() + "\n\n" + appendix + "\n"
    return {"A": control, "B": treatment}


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


def schedule() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(1, REPEATS + 1):
        scenario_order = SCENARIO_IDS if repeat != 2 else tuple(reversed(SCENARIO_IDS))
        for scenario_index, scenario_id in enumerate(scenario_order):
            a_first = (repeat + scenario_index) % 2 == 1
            conditions = CONDITIONS if a_first else tuple(reversed(CONDITIONS))
            for condition in conditions:
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


def _request(row: dict[str, Any], fixtures: dict[str, Scenario]) -> dict[str, Any]:
    scenario = fixtures[row["scenario_id"]]
    prompts = condition_prompts()
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompts[row["condition"]]},
            {
                "role": "user",
                "content": json.dumps(_input_payload(scenario), ensure_ascii=False),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_turn",
                "strict": True,
                "schema": semantic_turn_json_schema(scenario.previous_acs),
            },
        },
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _dependency_snapshot() -> dict[str, Any]:
    prompts = condition_prompts()
    fixtures = {row.scenario_id: row for row in scenarios()}
    return {
        "git_head": _git(["rev-parse", "HEAD"]),
        "production_prompt_file_sha256": _file_sha(
            ROOT / "ai/prompts/cognitive_core_system_prompt.txt"
        ),
        "cognitive_core_sha256": _file_sha(ROOT / "ai/cognitive_core.py"),
        "semantic_wire_sha256": _file_sha(ROOT / "ai/cognitive_semantic_wire.py"),
        "discovery_data_model_sha256": _file_sha(ROOT / "ai/discovery_data_model.py"),
        "human_experience_sha256": _file_sha(ROOT / "ai/human_experience.py"),
        "legacy_fixture_runner_sha256": _file_sha(
            ROOT / "experiments/cognitive_core_ab/run_experiment.py"
        ),
        "schema_validator_runner_sha256": _file_sha(
            ROOT / "experiments/semantic_reference_ab/run_experiment.py"
        ),
        "action_selection_audit_sha256": _file_sha(
            ROOT / "experiments/COGNITIVE_CORE_ACTION_SELECTION_AUDIT.md"
        ),
        "control_prompt_sha256": _sha(prompts["A"]),
        "treatment_prompt_sha256": _sha(prompts["B"]),
        "treatment_appendix_sha256": _file_sha(APPENDIX_PATH),
        "scenario_catalog_sha256": _file_sha(SCENARIOS_PATH),
        "scenario_input_sha256": {
            key: _sha(_json_bytes(_input_payload(fixtures[key])))
            for key in SCENARIO_IDS
        },
        "schema_sha256": {
            key: _sha(
                _json_bytes(semantic_turn_json_schema(fixtures[key].previous_acs))
            )
            for key in SCENARIO_IDS
        },
    }


def _verify_package_hashes() -> None:
    expected = json.loads(PACKAGE_HASHES_PATH.read_text(encoding="utf-8"))["files"]
    for relative, frozen_hash in expected.items():
        actual = _file_sha(PACKAGE / relative)
        if actual != frozen_hash:
            raise RuntimeError(
                f"Experiment package hash mismatch for {relative}: "
                f"expected {frozen_hash}, got {actual}"
            )


def _probe_payload() -> dict[str, Any]:
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
                "handle": "offline_probe",
                "kind": "system_reflection",
                "semantic_content": "Offline contract probe.",
                "model_item_refs": [],
                "relation_refs": [],
            },
            "additional_contents": [],
            "response_targets": [],
        },
        "reply_segments": [
            {"text": "Offline contract probe.", "realizes": ["offline_probe"]}
        ],
    }


def _catalog_by_id() -> dict[str, dict[str, Any]]:
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    return {row["id"]: row for row in raw["scenarios"]}


def _build_blinded(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    ordered = sorted(records, key=lambda row: _sha("semantic-delta-blind-v1/" + row["key"]))
    blinded: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for index, row in enumerate(ordered, 1):
        blind_id = f"R{index:02d}"
        mapping[blind_id] = row["key"]
        blinded.append(
            {
                "blind_id": blind_id,
                "scenario_id": row["scenario_id"],
                "context": row["context"],
                "reply": row["reply"],
            }
        )
    return blinded, mapping


def dry_run(*, quiet: bool = False) -> dict[str, Any]:
    _verify_package_hashes()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual_dependencies = _dependency_snapshot()
    if actual_dependencies != manifest["frozen_dependencies"]:
        raise RuntimeError(
            "Frozen dependency mismatch:\n"
            + json.dumps(
                {
                    "expected": manifest["frozen_dependencies"],
                    "actual": actual_dependencies,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if manifest["schedule"] != schedule():
        raise RuntimeError("Frozen schedule mismatch.")
    if manifest["api_call_budget"] != MAX_CALLS or MAX_CALLS != 30:
        raise RuntimeError("API budget mismatch.")
    if manifest["max_retries"] != 0 or MAX_RETRIES != 0:
        raise RuntimeError("Retries must remain disabled.")

    prompts = condition_prompts()
    appendix = APPENDIX_PATH.read_text(encoding="utf-8").strip()
    if prompts["B"] != prompts["A"].rstrip() + "\n\n" + appendix + "\n":
        raise RuntimeError("B is not exactly A plus the frozen appendix.")
    if "SEMANTIC REFERENCE QUICK MAP" in appendix:
        raise RuntimeError("Prior technical-reference treatment leaked into B.")

    fixtures = {row.scenario_id: row for row in scenarios()}
    catalog = _catalog_by_id()
    if set(fixtures) != set(SCENARIO_IDS) or set(catalog) != set(SCENARIO_IDS):
        raise RuntimeError("Scenario set mismatch.")
    probe = _probe_payload()
    for scenario_id in SCENARIO_IDS:
        fixture = fixtures[scenario_id]
        if fixture.current.text != catalog[scenario_id]["current_user_message"]:
            raise RuntimeError(f"Frozen USER mismatch for {scenario_id}.")
        first = _input_payload(fixture)
        second = _input_payload(fixture)
        if _json_bytes(first) != _json_bytes(second):
            raise RuntimeError(f"Non-deterministic input for {scenario_id}.")
        schema = semantic_turn_json_schema(fixture.previous_acs)
        if _schema_errors(probe, schema):
            raise RuntimeError(f"Offline probe violates schema for {scenario_id}.")
        SemanticTurn.model_validate(probe)
        adapt_semantic_turn(
            probe,
            current_user_text=fixture.current.text,
            human_model=fixture.human_model,
            previous_acs=fixture.previous_acs,
        )
        for condition in CONDITIONS:
            request = _request(
                {"condition": condition, "scenario_id": scenario_id}, fixtures
            )
            if request["response_format"]["json_schema"]["schema"] != schema:
                raise RuntimeError(f"Schema differs by condition for {scenario_id}.")

    mock_records = [
        {
            "key": row["key"],
            "scenario_id": row["scenario_id"],
            "context": {"current_user_message": catalog[row["scenario_id"]]["current_user_message"]},
            "reply": f"offline-placeholder-{row['key']}",
        }
        for row in schedule()
    ]
    blinded, mapping = _build_blinded(mock_records)
    if len(blinded) != 30 or len(mapping) != 30 or len(set(mapping.values())) != 30:
        raise RuntimeError("Blind-set preparation is not one-to-one.")

    result = {
        "status": "READY",
        "api_calls": 0,
        "future_api_budget": MAX_CALLS,
        "scenarios": list(SCENARIO_IDS),
        "conditions": list(CONDITIONS),
        "repeats": REPEATS,
        "frozen_dependencies": actual_dependencies,
    }
    if not quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("DRY_RUN_PASS API_CALLS=0")
    return result


def _append_event(path: Path, event: dict[str, Any], previous_hash: str | None) -> str:
    core = {**event, "previous_event_sha256": previous_hash}
    event_hash = _sha(_json_bytes(core))
    row = {**core, "event_sha256": event_hash}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event_hash


def _checkpoint_state(path: Path) -> tuple[set[str], str | None, int]:
    attempted: set[str] = set()
    previous: str | None = None
    calls = 0
    if not path.exists():
        return attempted, previous, calls
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        claimed = row.pop("event_sha256")
        if row.get("previous_event_sha256") != previous or _sha(_json_bytes(row)) != claimed:
            raise RuntimeError(f"Checkpoint hash-chain failure at line {line_number}.")
        previous = claimed
        if row["event_type"] == "API_REQUEST":
            if row["key"] in attempted:
                raise RuntimeError(f"Duplicate API attempt for {row['key']}.")
            attempted.add(row["key"])
            calls += 1
    return attempted, previous, calls


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


def _validation(
    text: str, scenario: Scenario
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result: dict[str, Any] = {
        "json_parsing": False,
        "json_schema": False,
        "semantic_turn": False,
        "adapter": False,
        "cognitive_turn_result": False,
        "error": None,
    }
    try:
        payload = json.loads(text)
        result["json_parsing"] = True
        errors = _schema_errors(payload, semantic_turn_json_schema(scenario.previous_acs))
        if errors:
            raise ValueError("; ".join(errors))
        result["json_schema"] = True
        SemanticTurn.model_validate(payload)
        result["semantic_turn"] = True
        adapted = adapt_semantic_turn(
            payload,
            current_user_text=scenario.current.text,
            human_model=scenario.human_model,
            previous_acs=scenario.previous_acs,
        )
        result["adapter"] = True
        result["cognitive_turn_result"] = True
        return result, adapted.model_dump(mode="json")
    except (json.JSONDecodeError, ValidationError, SemanticWireError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result, None


def _write_blinded_outputs(run_dir: Path, checkpoint: Path) -> None:
    fixtures = {row.scenario_id: row for row in scenarios()}
    records: list[dict[str, Any]] = []
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event_type"] != "API_RESPONSE":
            continue
        scenario = fixtures[row["scenario_id"]]
        validation, adapted = _validation(row["text"], scenario)
        reply = None if adapted is None else "".join(
            segment["text"] for segment in adapted["reply_segments"]
        )
        records.append(
            {
                "key": row["key"],
                "scenario_id": row["scenario_id"],
                "context": _input_payload(scenario),
                "reply": reply,
                "semantic_turn": (
                    json.loads(row["text"])
                    if validation["json_parsing"]
                    else None
                ),
                "technical_validation": validation,
            }
        )
    blinded, mapping = _build_blinded(records)
    by_key = {row["key"]: row for row in records}
    blinded_action_mapping = [
        {
            "blind_id": blind_id,
            "scenario_id": by_key[key]["scenario_id"],
            "selected_action": (
                None
                if by_key[key]["semantic_turn"] is None
                else by_key[key]["semantic_turn"].get("selected_action")
            ),
            "reply_segments": (
                None
                if by_key[key]["semantic_turn"] is None
                else by_key[key]["semantic_turn"].get("reply_segments")
            ),
        }
        for blind_id, key in mapping.items()
    ]
    (run_dir / "BLINDED_RESPONSES.json").write_text(
        json.dumps(blinded, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "BLINDED_ACTION_MAPPING.json").write_text(
        json.dumps(blinded_action_mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "BLINDING_KEY.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_live(authorization: str | None) -> None:
    if authorization != AUTHORIZATION:
        raise RuntimeError("Explicit authorization token is required.")
    dry_run(quiet=True)
    load_dotenv(ROOT / ".env")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    resumable = sorted(
        path
        for path in RESULTS_ROOT.iterdir()
        if path.is_dir()
        and (path / "manifest.json").exists()
        and not (path / "RUN_COMPLETED").exists()
        and json.loads((path / "manifest.json").read_text(encoding="utf-8")) == manifest
    )
    if resumable:
        run_dir = resumable[-1]
    else:
        run_id = (
            "semantic-delta-ab-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        run_dir = RESULTS_ROOT / run_id
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "prompts.json").write_text(
            json.dumps(condition_prompts(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    checkpoint = run_dir / "checkpoint.jsonl"
    attempted, previous_hash, call_count = _checkpoint_state(checkpoint)
    fixtures = {row.scenario_id: row for row in scenarios()}
    client = OpenAI()
    for row in schedule():
        if row["key"] in attempted:
            continue
        if call_count >= MAX_CALLS:
            raise RuntimeError("Hard API budget exhausted.")
        request = _request(row, fixtures)
        previous_hash = _append_event(
            checkpoint,
            {
                **row,
                "event_type": "API_REQUEST",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_sha256": _sha(_json_bytes(request)),
            },
            previous_hash,
        )
        call_count += 1
        started = time.monotonic()
        try:
            response = client.chat.completions.create(**request)
        except Exception as error:
            previous_hash = _append_event(
                checkpoint,
                {
                    **row,
                    "event_type": "API_FAILURE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                },
                previous_hash,
            )
            continue
        text = response.choices[0].message.content or ""
        previous_hash = _append_event(
            checkpoint,
            {
                **row,
                "event_type": "API_RESPONSE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_seconds": time.monotonic() - started,
                "actual_model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": _usage(response),
                "text": text,
                "raw_response": response.model_dump(mode="json"),
            },
            previous_hash,
        )
        validation, adapted = _validation(text, fixtures[row["scenario_id"]])
        previous_hash = _append_event(
            checkpoint,
            {
                **row,
                "event_type": "SEMANTIC_VALIDATION",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validation": validation,
                "cognitive_turn_result": adapted,
            },
            previous_hash,
        )

    attempted, final_hash, final_calls = _checkpoint_state(checkpoint)
    if final_calls != MAX_CALLS or len(attempted) != MAX_CALLS:
        raise RuntimeError("Run did not attempt the exact frozen schedule.")
    _write_blinded_outputs(run_dir, checkpoint)
    (run_dir / "RUN_COMPLETED").write_text(
        json.dumps(
            {
                "api_calls": final_calls,
                "final_event_sha256": final_hash,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"RUN_COMPLETED {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    if args.execute:
        run_live(args.authorization)
    else:
        dry_run()


if __name__ == "__main__":
    main()
