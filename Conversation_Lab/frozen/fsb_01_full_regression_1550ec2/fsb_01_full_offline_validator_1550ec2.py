"""Offline validator for the full FSB-01 regression bound to HEAD 1550ec2."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HEAD = "1550ec2"
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
SOURCE = ROOT / "Conversation_Lab/frozen/fsb_01_672455a"
MANIFEST = PACKAGE / "fsb_01_full_manifest_1550ec2.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA
    from ai.cognitive_turn import CognitiveTurnResult
    from Conversation_Lab.fsb_01_live_runner_672455a import _load_production_input, _sha_value

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert head == HEAD == manifest["head"]

    for name, expected in manifest["source_artifact_sha256"].items():
        assert sha_file(SOURCE / name) == expected
    prompt = subprocess.run(
        ["git", "show", manifest["prompt"]["git_path"]], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout
    assert hashlib.sha256(prompt).hexdigest().upper() == manifest["prompt"]["sha256"]
    assert (ROOT / "ai/prompts/cognitive_core_system_prompt.txt").read_bytes() == prompt

    boundary = manifest["contract_boundary"]
    assert sha_file(ROOT / "ai/cognitive_core.py") == boundary["cognitive_core_file_sha256"]
    assert sha_file(ROOT / "ai/cognitive_turn.py") == boundary["cognitive_turn_file_sha256"]
    assert sha_file(ROOT / "Conversation_Lab/selected_action_evaluator.py") == boundary["selected_action_evaluator_file_sha256"]
    assert _sha_value(COGNITIVE_TURN_JSON_SCHEMA) == boundary["cognitive_core_wire_schema_sha256"]
    assert _sha_value(CognitiveTurnResult.model_json_schema()) == boundary["cognitive_turn_schema_sha256"]

    cases_doc = json.loads((SOURCE / "fsb_01_cases_672455a.json").read_text(encoding="utf-8"))
    challengers_doc = json.loads((SOURCE / "fsb_01_challengers_672455a.json").read_text(encoding="utf-8"))
    order = manifest["case_order"]
    assert order == ["FSB-H", "FSB-M", "FSB-R", "FSB-S", "FSB-T", "FSB-A"]
    assert [case["case_id"] for case in cases_doc["cases"]] == order
    assert [item["case_id"] for item in challengers_doc["challengers"]] == order
    for case in cases_doc["cases"]:
        case_id = case["case_id"]
        assert _sha_value(case) == manifest["case_sha256"][case_id]
        assert case["state_sha256"] == manifest["state_sha256"][case_id]
        _load_production_input(case)
    for item in challengers_doc["challengers"]:
        assert item["challenger_sha256"] == manifest["challenger_sha256"][item["case_id"]]

    source_manifest = json.loads((SOURCE / "fsb_01_manifest_672455a.json").read_text(encoding="utf-8"))
    assert manifest["production_order"] == source_manifest["baseline_production_order"]
    assert manifest["evaluator_order"] == source_manifest["baseline_evaluator_order"]
    for case_id in order:
        assert manifest["evaluator_orientations_selected_on_left"][case_id] == source_manifest["evaluator_orientations_selected_on_left"][case_id]["baseline"]

    assert manifest["production_model"] == "gpt-5-mini"
    assert manifest["evaluator_model"] == "gpt-6-astra"
    assert manifest["production_boundary"] == "SELECTED_ONLY"
    assert manifest["protocol"] == {
        "retry": 0, "smoke_calls": 0, "confirmations": 0,
        "adaptive_sampling": False, "challenger_generation": False,
        "candidate_materialization": False, "all_production_before_evaluator": True,
    }
    assert manifest["api_budget"] == {"production": 6, "evaluator": 18, "maximum_total": 24}

    print(json.dumps({
        "result": "READY", "head": head, "cases": order,
        "case_identity": "PASS", "state_reachability": "6/6",
        "challenger_identity": "PASS", "prompt_identity": "PASS",
        "wire_schema_identity": "PASS", "cognitive_turn_identity": "PASS",
        "projection_identity": "PASS", "evaluator_identity": "PASS",
        "orientations_identity": "PASS", "selected_only": "PASS",
        "production_model": "gpt-5-mini", "evaluator_model": "gpt-6-astra",
        "future_api_calls": 24, "api_calls": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
