"""Offline validator for the frozen FSB-M prompt-clarification A/B."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
SOURCE = ROOT / "Conversation_Lab" / "frozen" / "fsb_01_mrs_policy_patch_1fe07e9"
MANIFEST_PATH = PACKAGE / "fsb_m_prompt_ab_manifest_1550ec2.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest().upper()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def git_bytes(spec: str) -> bytes:
    return subprocess.run(["git", "show", spec], cwd=ROOT, check=True, capture_output=True).stdout


def main() -> int:
    from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA
    from ai.cognitive_turn import CognitiveTurnResult
    from ai.discovery_data_model import (
        ActiveConversationState, DialogueHistory, DialogueMessage, DialogueRole,
        HumanModel, validate_discovery_memory,
    )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    if head != manifest["head"] or head != "1550ec2":
        raise RuntimeError(f"HEAD mismatch: {head}")

    prompt_hashes = {}
    for key in ("arm_a", "arm_b"):
        spec = manifest[key]["prompt_git_path"]
        actual = hashlib.sha256(git_bytes(spec)).hexdigest().upper()
        if actual != manifest[key]["prompt_sha256"]:
            raise RuntimeError(f"{key} prompt identity mismatch")
        prompt_hashes[key] = actual
    if sha_file(ROOT / "ai" / "prompts" / "cognitive_core_system_prompt.txt") != manifest["arm_b"]["prompt_sha256"]:
        raise RuntimeError("Working-tree treatment prompt is not HEAD treatment bytes")
    if prompt_hashes["arm_a"] == prompt_hashes["arm_b"]:
        raise RuntimeError("Control and treatment prompts are identical")

    for name, expected in manifest["source_artifact_sha256"].items():
        if sha_file(SOURCE / name) != expected:
            raise RuntimeError(f"Source artifact drift: {name}")
    for path, key in (
        (ROOT / "ai" / "cognitive_core.py", "cognitive_core_file_sha256"),
        (ROOT / "ai" / "cognitive_turn.py", "cognitive_turn_file_sha256"),
        (ROOT / "Conversation_Lab" / "selected_action_evaluator.py", "selected_action_evaluator_file_sha256"),
    ):
        if sha_file(path) != manifest["contract_boundary"][key]:
            raise RuntimeError(f"Contract/projection/evaluator drift: {path.name}")
    if sha_value(COGNITIVE_TURN_JSON_SCHEMA) != manifest["contract_boundary"]["cognitive_core_wire_schema_sha256"]:
        raise RuntimeError("Wire schema drift")
    if sha_value(CognitiveTurnResult.model_json_schema()) != manifest["contract_boundary"]["cognitive_turn_schema_sha256"]:
        raise RuntimeError("CognitiveTurnResult schema drift")

    cases = json.loads((SOURCE / "fsb_mrs_cases_1fe07e9.json").read_text(encoding="utf-8"))["cases"]
    case = next(item for item in cases if item["case_id"] == manifest["base_case_id"])
    challengers = json.loads((SOURCE / "fsb_mrs_challengers_1fe07e9.json").read_text(encoding="utf-8"))["challengers"]
    challenger = next(item for item in challengers if item["case_id"] == manifest["base_case_id"])
    if sha_value(case) != manifest["case_sha256"] or case["state_sha256"] != manifest["state_sha256"]:
        raise RuntimeError("FSB-M state/case identity mismatch")
    if challenger["challenger_sha256"] != manifest["challenger_sha256"]:
        raise RuntimeError("FSB-M challenger identity mismatch")
    frozen = case["frozen_input"]
    current = DialogueMessage.model_validate(frozen["current_user_message"])
    model = HumanModel.model_validate(frozen["human_model"])
    acs = ActiveConversationState.model_validate(frozen["previous_acs"])
    history = DialogueHistory.model_validate(frozen["dialogue_history"])
    validate_discovery_memory(model, acs, history)
    if current.role != DialogueRole.USER or current.sequence != history.messages[-1].sequence + 1:
        raise RuntimeError("Frozen FSB-M input is not reachable")

    order = manifest["production_order"]
    if order != manifest["evaluator_order"] or order != [
        "FSB-M-A1", "FSB-M-B1", "FSB-M-B2", "FSB-M-A2", "FSB-M-A3", "FSB-M-B3"
    ]:
        raise RuntimeError("Frozen balanced order drift")
    if manifest["samples_per_arm"] != 3 or manifest["production_model"] != "gpt-6-astra":
        raise RuntimeError("Sample/model drift")
    if manifest["api_budget"] != {"production": 6, "evaluator": 18, "maximum_total": 24}:
        raise RuntimeError("API budget drift")
    if manifest["retry"] != 0 or manifest["adaptive_sampling"] or manifest["smoke_calls"] != 0:
        raise RuntimeError("Forbidden adaptive/retry/smoke behavior")

    print(json.dumps({
        "result": "READY", "head": head, "api_calls": 0,
        "samples_per_arm": 3, "future_api_calls": 24,
        "control_prompt_identity": "PASS", "treatment_prompt_identity": "PASS",
        "state_identity": "PASS", "challenger_identity": "PASS",
        "schema_identity": "PASS", "evaluator_identity": "PASS",
        "projection_identity": "PASS", "only_variable": "production_prompt"
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
