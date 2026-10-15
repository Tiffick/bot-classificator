"""Lab-only A/B test of SemanticTurn reference instructions.

Dry-run is the default. Live execution requires both ``--execute`` and the
explicit authorization token. The runner never retries a request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
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
MANIFEST_PATH = PACKAGE / "manifest.json"
APPENDIX_PATH = PACKAGE / "treatment_appendix.txt"
EXAMPLES_PATH = PACKAGE / "examples.json"
PACKAGE_HASHES_PATH = PACKAGE / "package_hashes.json"
RESULTS_ROOT = PACKAGE / "results"

from ai.cognitive_core import CognitiveCore  # noqa: E402
from ai.cognitive_semantic_wire import (  # noqa: E402
    SemanticTurn,
    adapt_semantic_turn,
    semantic_turn_json_schema,
)
from ai.cognitive_turn import CognitiveTurnResult  # noqa: E402
from ai.discovery_data_model import HumanModel  # noqa: E402
from ai.human_experience import HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION  # noqa: E402
from experiments.cognitive_core_ab.run_experiment import (  # noqa: E402
    _input_payload,
    reconstruct_model,
    scenarios,
)

MODEL = "gpt-5-mini-2025-08-07"
REASONING_EFFORT = "minimal"
MAX_COMPLETION_TOKENS = 4000
MAX_RETRIES = 0
LIVE_AUTHORIZATION = "I_AUTHORIZE_EXACTLY_12_CALLS"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest().upper()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _production_prompt() -> str:
    return (
        CognitiveCore._system_prompt().rstrip()
        + "\n"
        + HUMAN_EXPERIENCE_SYSTEM_INSTRUCTION
        + "\n"
    )


def _condition_prompts() -> dict[str, str]:
    control = _production_prompt()
    appendix = APPENDIX_PATH.read_text(encoding="utf-8").strip()
    treatment = control.rstrip() + "\n\n" + appendix + "\n"
    return {"A": control, "B": treatment}


def _scenario_map() -> dict[str, Any]:
    selected = {row.scenario_id: row for row in scenarios() if row.scenario_id in {"S1", "S3"}}
    if set(selected) != {"S1", "S3"}:
        raise RuntimeError("Frozen S1/S3 fixtures are unavailable.")
    return selected


def _schedule() -> list[dict[str, Any]]:
    # Counterbalance which condition appears first within the six matched pairs.
    pairs = (
        ("S1", 1, ("A", "B")),
        ("S3", 1, ("B", "A")),
        ("S1", 2, ("B", "A")),
        ("S3", 2, ("A", "B")),
        ("S1", 3, ("A", "B")),
        ("S3", 3, ("B", "A")),
    )
    rows: list[dict[str, Any]] = []
    for scenario_id, repeat, conditions in pairs:
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
    return rows


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" not in schema:
        return schema
    current: Any = root
    for part in schema["$ref"].split("/")[1:]:
        current = current[part.replace("~1", "/").replace("~0", "~")]
    return current


def _schema_errors(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any] | None = None,
    path: str = "$",
) -> list[str]:
    """Validate every keyword used by the current generated JSON Schema.

    The project intentionally has no third-party JSON Schema validator. This
    lab validator fails closed if the production schema starts using a keyword
    outside the explicitly implemented set.
    """
    root = schema if root is None else root
    if "$ref" in schema:
        return _schema_errors(value, _resolve_ref(schema, root), root, path)
    if "anyOf" in schema:
        branch_errors = [_schema_errors(value, branch, root, path) for branch in schema["anyOf"]]
        if any(not errors for errors in branch_errors):
            return []
        return [f"{path}: no anyOf branch matched"]

    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is outside enum")

    expected = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "null": value is None,
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }.get(expected, True)
    if not type_ok:
        return errors + [f"{path}: expected type {expected}"]

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}: missing required property {required}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key}: additional property forbidden")
        for key, item in value.items():
            if key in properties:
                errors.extend(_schema_errors(item, properties[key], root, f"{path}.{key}"))

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, schema["items"], root, f"{path}[{index}]"))

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: does not match pattern")
    return errors


def _validate_schema_vocabulary(schema: dict[str, Any]) -> None:
    allowed = {
        "$defs", "$ref", "additionalProperties", "anyOf", "const", "enum",
        "items", "maxItems", "maxLength", "minItems", "minLength", "pattern",
        "properties", "required", "title", "type",
    }

    def visit(node: Any, *, schema_node: bool = True) -> None:
        if not isinstance(node, dict):
            return
        if schema_node:
            unknown = set(node) - allowed
            if unknown:
                raise RuntimeError(f"Unsupported JSON Schema keywords: {sorted(unknown)}")
        for key, child in node.items():
            if key in {"$defs", "properties"}:
                for nested in child.values():
                    visit(nested)
            elif key == "items":
                visit(child)
            elif key == "anyOf":
                for nested in child:
                    visit(nested)

    visit(schema)


def _exception_text(error: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = error
    while current is not None:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    return " | ".join(parts)


def _validate_examples() -> dict[str, Any]:
    document = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    appendix = APPENDIX_PATH.read_text(encoding="utf-8")
    results: dict[str, Any] = {"correct": {}, "incorrect": {}}
    for example in document["correct_examples"]:
        rendered = json.dumps(example["payload"], ensure_ascii=False, separators=(",", ":"))
        if rendered not in appendix:
            raise RuntimeError(f"Correct example {example['id']} is not embedded verbatim in treatment.")
        human_model = HumanModel.model_validate(example["human_model"])
        schema = semantic_turn_json_schema(None)
        _validate_schema_vocabulary(schema)
        schema_errors = _schema_errors(example["payload"], schema)
        if schema_errors:
            raise RuntimeError(f"Correct example {example['id']} failed JSON Schema: {schema_errors}")
        semantic = SemanticTurn.model_validate(example["payload"])
        adapted = adapt_semantic_turn(
            semantic,
            current_user_text=example["current_user_text"],
            human_model=human_model,
            previous_acs=None,
        )
        CognitiveTurnResult.model_validate(adapted.model_dump(mode="python"))
        results["correct"][example["id"]] = {
            "json_schema": "PASS",
            "semantic_turn": "PASS",
            "adapter": "PASS",
            "internal_contract": "PASS",
        }

    for example in document["incorrect_examples"]:
        invalid_ref = json.dumps(
            example["payload"]["selected_action"]["primary_content"]["model_item_refs"][0],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if invalid_ref not in appendix:
            raise RuntimeError(f"Incorrect example {example['id']} is absent from treatment.")
        human_model = HumanModel.model_validate(example["human_model"])
        schema = semantic_turn_json_schema(None)
        schema_errors = _schema_errors(example["payload"], schema)
        if schema_errors:
            raise RuntimeError(
                f"Incorrect example {example['id']} unexpectedly failed JSON Schema: {schema_errors}"
            )
        try:
            adapt_semantic_turn(
                example["payload"],
                current_user_text=example["current_user_text"],
                human_model=human_model,
                previous_acs=None,
            )
        except Exception as error:
            detail = _exception_text(error)
            expected = example["expected_error_fragment"]
            if expected not in detail:
                raise RuntimeError(
                    f"Incorrect example {example['id']} failed for the wrong reason: {detail}"
                ) from error
            results["incorrect"][example["id"]] = {
                "json_schema": "PASS",
                "semantic_turn": "EXPECTED_FAIL",
                "error": expected,
            }
        else:
            raise RuntimeError(f"Incorrect example {example['id']} was accepted.")
    return results


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _verify_frozen_dependencies(manifest: dict[str, Any]) -> dict[str, Any]:
    prompts = _condition_prompts()
    fixtures = _scenario_map()
    actual = {
        "git_head": _git(["rev-parse", "HEAD"]),
        "production_prompt_file_sha256": _file_sha(ROOT / "ai/prompts/cognitive_core_system_prompt.txt"),
        "semantic_wire_sha256": _file_sha(ROOT / "ai/cognitive_semantic_wire.py"),
        "source_fixture_runner_sha256": _file_sha(ROOT / "experiments/cognitive_core_ab/run_experiment.py"),
        "source_pilot_manifest_sha256": _file_sha(
            ROOT / "experiments/cognitive_core_ab/results/cc-abcd-v3-20261011T023640Z-d13f5f46/manifest.json"
        ),
        "control_prompt_sha256": _sha(prompts["A"]),
        "treatment_appendix_sha256": _file_sha(APPENDIX_PATH),
        "treatment_prompt_sha256": _sha(prompts["B"]),
        "examples_sha256": _file_sha(EXAMPLES_PATH),
        "scenario_input_sha256": {
            key: _sha(_json_bytes(_input_payload(fixtures[key]))) for key in ("S1", "S3")
        },
        "schema_sha256": {
            key: _sha(_json_bytes(semantic_turn_json_schema(fixtures[key].previous_acs)))
            for key in ("S1", "S3")
        },
    }
    expected = manifest["frozen_dependencies"]
    if actual != expected:
        raise RuntimeError(
            "Frozen dependency mismatch:\n"
            + json.dumps({"expected": expected, "actual": actual}, ensure_ascii=False, indent=2)
        )
    if not prompts["B"].startswith(prompts["A"].rstrip() + "\n\n"):
        raise RuntimeError("Treatment is not control plus the frozen appendix.")
    return actual


def _verify_package_hashes() -> None:
    frozen = json.loads(PACKAGE_HASHES_PATH.read_text(encoding="utf-8"))["files"]
    for relative, expected in frozen.items():
        actual = _file_sha(PACKAGE / relative)
        if actual != expected:
            raise RuntimeError(
                f"Experiment package hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )


def _request(row: dict[str, Any], prompts: dict[str, str], fixtures: dict[str, Any]) -> dict[str, Any]:
    scenario = fixtures[row["scenario_id"]]
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompts[row["condition"]]},
            {
                "role": "user",
                "content": json.dumps(_input_payload(scenario), ensure_ascii=False, indent=2),
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
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }


def dry_run() -> dict[str, Any]:
    _verify_package_hashes()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dependencies = _verify_frozen_dependencies(manifest)
    if manifest["schedule"] != _schedule():
        raise RuntimeError("Frozen schedule mismatch.")
    if len(_schedule()) != 12:
        raise RuntimeError("API budget must be exactly 12 calls.")
    prompts = _condition_prompts()
    fixtures = _scenario_map()
    request_hashes: dict[str, str] = {}
    for row in _schedule():
        request = _request(row, prompts, fixtures)
        request_hashes[row["key"]] = _sha(_json_bytes(request))
        if request["response_format"]["json_schema"]["strict"] is not True:
            raise RuntimeError("Strict schema is not enabled.")
    return {
        "status": "READY",
        "api_calls": 0,
        "dependencies": dependencies,
        "examples": _validate_examples(),
        "request_hashes": request_hashes,
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


def execute(authorization: str) -> Path:
    if authorization != LIVE_AUTHORIZATION:
        raise RuntimeError("Explicit authorization token is required for live execution.")
    dry_run()
    load_dotenv(ROOT / ".env")
    client = OpenAI(max_retries=MAX_RETRIES)
    run_id = f"semantic-ref-ab-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "checkpoint.jsonl"
    prompts = _condition_prompts()
    fixtures = _scenario_map()
    previous_hash: str | None = None
    calls = 0

    for row in _schedule():
        request = _request(row, prompts, fixtures)
        request_hash = _sha(_json_bytes(request))
        previous_hash = _append_event(
            checkpoint,
            {
                "event_type": "API_REQUEST",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **row,
                "request_sha256": request_hash,
            },
            previous_hash,
        )
        calls += 1
        if calls > 12:
            raise RuntimeError("Hard API budget exceeded.")
        started = time.monotonic()
        try:
            response = client.chat.completions.create(**request)
        except Exception as error:
            _append_event(
                checkpoint,
                {
                    "event_type": "API_FAILURE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **row,
                    "request_sha256": request_hash,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                previous_hash,
            )
            raise RuntimeError(f"API call failed at {row['key']}; no retry is permitted.") from error

        raw = response.model_dump(mode="json")
        text = response.choices[0].message.content or ""
        previous_hash = _append_event(
            checkpoint,
            {
                "event_type": "API_RESPONSE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **row,
                "request_sha256": request_hash,
                "latency_seconds": time.monotonic() - started,
                "actual_model": response.model,
                "raw_response": raw,
                "text": text,
                "usage": _usage(response),
            },
            previous_hash,
        )

        scenario = fixtures[row["scenario_id"]]
        try:
            payload = json.loads(text)
            schema_errors = _schema_errors(
                payload, semantic_turn_json_schema(scenario.previous_acs)
            )
            if schema_errors:
                raise ValueError(f"JSON Schema errors: {schema_errors}")
            semantic = SemanticTurn.model_validate(payload)
            adapted = adapt_semantic_turn(
                semantic,
                current_user_text=scenario.current.text,
                human_model=reconstruct_model(scenario.human_model),
                previous_acs=reconstruct_model(scenario.previous_acs),
            )
            CognitiveTurnResult.model_validate(adapted.model_dump(mode="python"))
            validation = {
                "valid": True,
                "reply": "".join(segment.text for segment in adapted.reply_segments),
                "cognitive_turn_result": adapted.model_dump(mode="json"),
            }
        except Exception as error:
            validation = {
                "valid": False,
                "error_type": type(error).__name__,
                "error": _exception_text(error),
            }
        previous_hash = _append_event(
            checkpoint,
            {
                "event_type": "SEMANTIC_VALIDATION",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **row,
                "request_sha256": request_hash,
                "validation": validation,
            },
            previous_hash,
        )

    _append_event(
        checkpoint,
        {
            "event_type": "RUN_COMPLETED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_calls": calls,
        },
        previous_hash,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", default="")
    args = parser.parse_args()
    if args.execute:
        print(execute(args.authorization))
        return
    print(json.dumps(dry_run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
