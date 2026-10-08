"""Offline validator for the frozen FSB-M product-policy A/B experiment."""

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
MANIFEST = PACKAGE / "fsb_m_product_policy_manifest_1550ec2.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Conversation_Lab.fsb_01_live_runner_672455a import _load_production_input, _sha_value

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert head == HEAD == manifest["head"]

    for name, expected in manifest["source_artifact_sha256"].items():
        assert sha_file(SOURCE / name) == expected
    for name, expected in manifest["frozen_artifact_sha256"].items():
        assert sha_file(PACKAGE / name) == expected

    cases = json.loads((SOURCE / "fsb_01_cases_672455a.json").read_text(encoding="utf-8"))["cases"]
    case = next(item for item in cases if item["case_id"] == "FSB-M")
    assert _sha_value(case) == manifest["source_case"]["case_sha256"]
    assert case["state_sha256"] == manifest["source_case"]["state_sha256"]
    _load_production_input(case)

    prompt_binding = manifest["production_prompt_binding"]
    production_prompt = subprocess.run(
        ["git", "show", prompt_binding["git_path"]], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout
    assert hashlib.sha256(production_prompt).hexdigest().upper() == prompt_binding["sha256"]
    assert prompt_binding["used_for_generation"] is False

    pair = json.loads((PACKAGE / "fsb_m_product_policy_pair_1550ec2.json").read_text(encoding="utf-8"))
    assert pair["source_case_sha256"] == manifest["source_case"]["case_sha256"]
    assert pair["source_state_sha256"] == manifest["source_case"]["state_sha256"]
    assert _sha_value(pair["action_a"]) == manifest["action_sha256"]["A"]
    assert _sha_value(pair["action_b"]) == manifest["action_sha256"]["B"]
    assert pair["action_a"]["strategy"] == "direct_discrimination"
    assert pair["action_a"]["response_target_interaction"] == "open_response"
    assert pair["action_b"]["strategy"] == "testable_working_version"
    assert pair["action_b"]["response_target_interaction"] == "evaluation"

    rubric = (PACKAGE / "fsb_m_product_policy_evaluator_prompt_1550ec2.txt").read_text(encoding="utf-8")
    for criterion in (
        "EXPECTED HUMAN MODEL DELTA", "USER BURDEN", "GROUNDING", "CORRECTABILITY",
        "EPISTEMIC SAFETY", "FORWARD VALUE", "NATURAL CONSULTATION QUALITY",
        "PRODUCT BOUNDARY",
    ):
        assert criterion in rubric
    assert "Reduction of system uncertainty has no value by itself" in rubric
    assert "Neither strategy has default priority" in rubric
    assert "LEFT" in rubric and "RIGHT" in rubric
    assert "action_a" not in rubric.lower() and "action_b" not in rubric.lower()

    schema = json.loads((PACKAGE / "fsb_m_product_policy_evaluator_schema_1550ec2.json").read_text(encoding="utf-8"))
    assert schema["required"] == ["verdict"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["verdict"]["enum"]) == {
        "LEFT_HIGHER", "RIGHT_HIGHER", "EQUIVALENT_OR_BOTH_ACCEPTABLE", "INSUFFICIENT_BASIS"
    }

    guards = json.loads((PACKAGE / "fsb_m_product_policy_guards_1550ec2.json").read_text(encoding="utf-8"))
    assert len(guards["working_version_guards"]) == 7
    assert len(guards["discrimination_guards"]) == 4
    guard_ids = [item["id"] for group in guards.values() for item in group]
    assert len(guard_ids) == len(set(guard_ids))
    assert all(item["expected"] == "REJECT_VALUE" for group in guards.values() for item in group)

    orientations = manifest["orientation_a_on_left"]
    assert orientations == [True, False, True, False, True, False]
    assert sum(orientations) == 3
    assert manifest["protocol"] == {
        "production_calls": 0, "evaluator_calls": 6, "retry": 0,
        "smoke_calls": 0, "confirmations": 0, "adaptive_sampling": False,
        "action_generation": False, "blind_labels_only": True,
    }
    assert manifest["api_budget"] == {"production": 0, "evaluator": 6, "maximum_total": 6}

    print(json.dumps({
        "result": "READY", "head": head, "case": "FSB-M",
        "context_identity": "PASS", "state_reachability": "1/1",
        "action_a_frozen": "PASS", "action_b_frozen": "PASS",
        "rubric_symmetry": "PASS", "guard_cases": "11/11",
        "orientation_balance": "3/3", "production_calls": 0,
        "evaluator_calls": 6, "evaluator_model": manifest["evaluator_model"],
        "api_calls": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
