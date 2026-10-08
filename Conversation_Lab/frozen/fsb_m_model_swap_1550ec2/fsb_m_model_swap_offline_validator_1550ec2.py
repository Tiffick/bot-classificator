"""Offline validator for the frozen FSB-M model-swap experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HEAD = "1550ec2"
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / "fsb_m_model_swap_manifest_1550ec2.json"
SOURCE = ROOT / "Conversation_Lab" / "frozen" / "fsb_01_mrs_policy_patch_1fe07e9"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA, CognitiveCore
    from ai.cognitive_turn import CognitiveTurnResult
    from Conversation_Lab.fsb_01_live_runner_672455a import (
        Checkpoint,
        _load_production_input,
        _sha_value,
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert head == HEAD == manifest["head"]

    identity = manifest["identity"]
    case_path = SOURCE / "fsb_mrs_cases_1fe07e9.json"
    assert sha_file(case_path) == identity["source_case_file_sha256"]
    cases_doc = json.loads(case_path.read_text(encoding="utf-8"))
    case = next(item for item in cases_doc["cases"] if item["case_id"] == "FSB-M")
    assert _sha_value(case) == identity["case_sha256"]
    assert case["state_sha256"] == identity["state_sha256"]

    current, human_model, previous_acs, history = _load_production_input(case)
    prompt_bytes = git_bytes(manifest["prompt"]["git_path"])
    assert hashlib.sha256(prompt_bytes).hexdigest().upper() == manifest["prompt"]["sha256"]
    assert (ROOT / "ai/prompts/cognitive_core_system_prompt.txt").read_bytes() == prompt_bytes

    assert sha_file(ROOT / "ai/cognitive_core.py") == identity["cognitive_core_file_sha256"]
    assert sha_file(ROOT / "ai/cognitive_turn.py") == identity["cognitive_turn_file_sha256"]
    assert _sha_value(COGNITIVE_TURN_JSON_SCHEMA) == identity["wire_schema_sha256"]
    assert _sha_value(CognitiveTurnResult.model_json_schema()) == identity["cognitive_turn_schema_sha256"]

    prompt_text = prompt_bytes.decode("utf-8")
    FrozenCore = type(
        "FrozenPromptCore", (CognitiveCore,),
        {"_system_prompt": staticmethod(lambda: prompt_text)},
    )

    def request(model: str) -> dict:
        return {
            "model": model,
            "messages": FrozenCore._messages(current, human_model, previous_acs, history),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cognitive_turn_result",
                    "strict": True,
                    "schema": COGNITIVE_TURN_JSON_SCHEMA,
                },
            },
        }

    astra_request = request(manifest["astra_reference"]["model"])
    mini_request = request(manifest["mini_arm"]["model"])
    assert _sha_value(astra_request) == identity["astra_request_sha256"]
    assert _sha_value(mini_request) == identity["mini_request_sha256"]
    astra_without_model = dict(astra_request)
    mini_without_model = dict(mini_request)
    astra_without_model.pop("model")
    mini_without_model.pop("model")
    assert _sha_value(astra_without_model) == _sha_value(mini_without_model)
    assert _sha_value(mini_without_model) == identity["model_excluded_request_sha256"]

    reference = manifest["astra_reference"]
    checkpoint_path = ROOT / reference["checkpoint_path"]
    run_json_path = ROOT / reference["run_json_path"]
    assert sha_file(checkpoint_path) == reference["checkpoint_sha256"]
    assert sha_file(run_json_path) == reference["run_json_sha256"]
    events = Checkpoint(checkpoint_path).read()
    assert events[-1]["event_type"] == "RUN_COMPLETED"
    assert events[-1]["api_calls"] == 24
    assert not [event for event in events if event["event_type"] == "STOP"]

    reference_ids = [item["sample_id"] for item in reference["samples"]]
    assert reference_ids == ["FSB-M-B1", "FSB-M-B2", "FSB-M-B3"]
    for expected in reference["samples"]:
        sample_id = expected["sample_id"]
        response = next(
            event for event in events
            if event["event_type"] == "API_RESPONSE"
            and event["call_spec"]["role"] == "production_decision"
            and event["call_spec"]["case_id"] == sample_id
        )
        completed = next(
            event for event in events
            if event["event_type"] == "CALL_COMPLETED"
            and event["call_spec"]["role"] == "production_decision"
            and event["call_spec"]["case_id"] == sample_id
        )
        assert response["request_sha256"] == expected["request_sha256"]
        raw = json.loads(response["response_raw"]["choices"][0]["message"]["content"])
        selected = raw["selected_action"]
        assert _sha_value(selected) == expected["selected_action_sha256"]
        assert selected["decision_intent"] == "reflection"
        assert selected["primary_content"]["kind"] == "system_reflection"
        assert selected["response_targets"][0]["interaction"] == "evaluation"
        assert completed["normalized_result"]["arm"] == "treatment"
        assert completed["normalized_result"]["production_prompt_sha256"] == manifest["prompt"]["sha256"]

    params = manifest["production_parameters"]
    assert params == {
        "retry": 0,
        "smoke_calls": 0,
        "confirmations": 0,
        "adaptive_sampling": False,
        "evaluator_calls": 0,
        "selected_only": True,
        "api_family": "chat.completions.create",
        "strict_structured_output": True,
        "client_timeout_seconds": 120.0,
    }
    assert manifest["mini_arm"]["sample_order"] == [
        "FSB-M-mini-1", "FSB-M-mini-2", "FSB-M-mini-3"
    ]
    assert manifest["api_budget"] == {"production": 3, "evaluator": 0, "maximum_total": 3}

    print(json.dumps({
        "result": "READY",
        "head": head,
        "astra_reference": "PASS",
        "astra_distribution": "0/3 direct; 3/3 reflection/evaluation",
        "state_identity": "PASS",
        "input_serialization_identity_except_model": "PASS",
        "prompt_identity": "PASS",
        "wire_schema_contract_identity": "PASS",
        "production_parameters_identity_except_model": "PASS",
        "mini_samples": 3,
        "evaluator_calls": 0,
        "future_api_calls": 3,
        "api_calls": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
