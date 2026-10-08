"""Fail-closed offline validator for the frozen FSB-D model swap."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HEAD = "1e08316"
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / "fsb_d_model_swap_manifest_1e08316.json"
EXPECTED_DEPENDENCIES = {
    "Conversation_Lab/frozen/fsb_01_revised_policy_1e08316/fsb_d_case_1e08316.json": "27B60F09C230C6326EF5641C45B851A1346E6D9878E191DB7CE6A4BCAEC8117F",
    "Conversation_Lab/frozen/fsb_01_revised_policy_1e08316/fsb_d_policy_pair_1e08316.json": "58819C30A951CE6A2BF68EC24611B270BA8D3548BB9344EAF78F679FC73DA973",
    "ai/cognitive_core.py": "E22BA7ECB4E90B5BF89F9E7B8C8DD65099D42D239F77CA8D2EE3E473AAC96543",
    "ai/cognitive_turn.py": "BB8AF8ED4DF114964C3D5717FD15B6D8AAB52F592C070CE95AE9EFA69C51C7B6",
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sha_value(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual_head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_head != HEAD or manifest.get("head") != HEAD:
        fail(f"HEAD mismatch: actual={actual_head!r}")

    for relative, expected in EXPECTED_DEPENDENCIES.items():
        actual = sha_file(ROOT / relative)
        if actual != expected:
            fail(f"Dependency SHA mismatch: {relative}: {actual}")

    prompt_spec = manifest["production_prompt"]
    prompt_bytes = subprocess.run(
        ["git", "show", prompt_spec["git_path"]], cwd=ROOT, check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != prompt_spec["sha256"]:
        fail("Production prompt SHA mismatch")

    from Conversation_Lab.fsb_01_live_runner_672455a import (
        _load_production_input,
    )
    from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA, CognitiveCore
    from ai.cognitive_turn import CognitiveTurnResult

    case = json.loads(
        (ROOT / manifest["reused_frozen_artifacts"]["case_path"]).read_text(
            encoding="utf-8"
        )
    )
    current, human_model, previous_acs, history = _load_production_input(case)
    shared = {
        "messages": CognitiveCore._messages(
            current, human_model, previous_acs, history
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cognitive_turn_result",
                "strict": True,
                "schema": COGNITIVE_TURN_JSON_SCHEMA,
            },
        },
    }
    mini = {"model": manifest["baseline_model"], **shared}
    astra = {"model": manifest["production_model"], **shared}
    identity = manifest["request_identity"]
    if sha_value(mini) != identity["historical_mini_request_sha256"]:
        fail("Historical mini request SHA is not reproducible")
    if sha_value(astra) != identity["expected_astra_request_sha256"]:
        fail("Expected Astra request SHA is not reproducible")
    if sha_value(shared) != identity["model_excluded_request_sha256"]:
        fail("Model-excluded request SHA mismatch")
    differing = sorted(key for key in mini if mini[key] != astra[key])
    if differing != identity["allowed_changed_request_keys"]:
        fail(f"Unexpected request differences: {differing}")

    if sha_value(COGNITIVE_TURN_JSON_SCHEMA) != manifest["contract_boundary"]["wire_schema_sha256"]:
        fail("Wire schema SHA mismatch")
    if sha_value(CognitiveTurnResult.model_json_schema()) != manifest["contract_boundary"]["cognitive_turn_schema_sha256"]:
        fail("CognitiveTurnResult schema SHA mismatch")
    params = manifest["production_parameters"]
    forbidden = (
        params["retry"], params["smoke_calls"], params["evaluator_calls"],
        params["confirmations"], params["adaptive_sampling"],
        params["candidate_materialization"],
    )
    if any(forbidden) or not params["selected_only"]:
        fail("Forbidden calls/adaptation are enabled")
    if manifest["api_budget"] != {"production": 3, "evaluator": 0, "maximum_total": 3}:
        fail("API budget mismatch")
    if len(manifest["sample_order"]) != 3 or len(set(manifest["sample_order"])) != 3:
        fail("Sample order must contain three unique frozen slots")
    if manifest["observable_classes"] != [
        "DIRECT_DISCRIMINATION", "LOCAL_REFLECTION", "OTHER"
    ]:
        fail("Observable class contract changed")

    print(json.dumps({
        "result": "READY",
        "api_calls": 0,
        "head": actual_head,
        "state_validation": "PASS",
        "historical_mini_request_sha256": sha_value(mini),
        "expected_astra_request_sha256": sha_value(astra),
        "model_excluded_request_sha256": sha_value(shared),
        "request_differences": differing,
        "samples": manifest["sample_order"],
        "budget": manifest["api_budget"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
