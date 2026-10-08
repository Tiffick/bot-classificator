"""Offline validator for the two-arm focused FSB-M prompt regression."""

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
MANIFEST_PATH = PACKAGE / "fsb_m_prompt_regression_manifest_d34fdca.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest().upper()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA
    from ai.cognitive_turn import CognitiveTurnResult
    from ai.discovery_data_model import (
        ActiveConversationState,
        DialogueHistory,
        DialogueMessage,
        DialogueRole,
        HumanModel,
        validate_discovery_memory,
    )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    if head != manifest["head"] or head != "d34fdca":
        raise RuntimeError(f"HEAD mismatch: {head}")

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
    if sha_file(ROOT / "ai" / "prompts" / "cognitive_core_system_prompt.txt") != manifest["production_prompt_sha256_new"]:
        raise RuntimeError("Current production prompt SHA mismatch")
    if sha_value(COGNITIVE_TURN_JSON_SCHEMA) != manifest["contract_boundary"]["cognitive_core_wire_schema_sha256"]:
        raise RuntimeError("Wire schema drift")
    if sha_value(CognitiveTurnResult.model_json_schema()) != manifest["contract_boundary"]["cognitive_turn_schema_sha256"]:
        raise RuntimeError("CognitiveTurnResult schema drift")

    cases = json.loads((SOURCE / "fsb_mrs_cases_1fe07e9.json").read_text(encoding="utf-8"))["cases"]
    case = next(item for item in cases if item["case_id"] == manifest["base_case_id"])
    challengers = json.loads((SOURCE / "fsb_mrs_challengers_1fe07e9.json").read_text(encoding="utf-8"))["challengers"]
    challenger = next(item for item in challengers if item["case_id"] == manifest["base_case_id"])
    if sha_value(case) != manifest["source_case_sha256"] or case["state_sha256"] != manifest["state_sha256"]:
        raise RuntimeError("FSB-M case/state identity mismatch")
    if challenger["challenger_sha256"] != manifest["source_challenger_sha256"]:
        raise RuntimeError("FSB-M challenger identity mismatch")
    if set(manifest["case_sha256"].values()) != {manifest["source_case_sha256"]}:
        raise RuntimeError("Arm case identity mismatch")
    if set(manifest["challenger_sha256"].values()) != {manifest["source_challenger_sha256"]}:
        raise RuntimeError("Arm challenger identity mismatch")

    frozen = case["frozen_input"]
    current = DialogueMessage.model_validate(frozen["current_user_message"])
    human_model = HumanModel.model_validate(frozen["human_model"])
    previous_acs = ActiveConversationState.model_validate(frozen["previous_acs"])
    history = DialogueHistory.model_validate(frozen["dialogue_history"])
    validate_discovery_memory(human_model, previous_acs, history)
    if current.role != DialogueRole.USER or current.sequence != history.messages[-1].sequence + 1:
        raise RuntimeError("FSB-M current message is not reachable from frozen history")

    arms = manifest["arms"]
    if [(x["arm_id"], x["production_model"]) for x in arms] != [
        ("FSB-M-ASTRA", "gpt-6-astra"), ("FSB-M-MINI", "gpt-5-mini")
    ]:
        raise RuntimeError("Arm/model order drift")
    if manifest["api_budget"] != {"production": 2, "evaluator": 6, "maximum_total": 8}:
        raise RuntimeError("API budget drift")
    if manifest["retry"] != 0 or manifest["adaptive_confirmation"] or manifest["smoke_calls"] != 0:
        raise RuntimeError("Forbidden adaptive/retry/smoke behavior")

    print(json.dumps({
        "result": "READY", "head": head, "api_calls": 0,
        "state_identity": "PASS", "challenger_identity": "PASS",
        "prompt_change_only": True, "wire_schema_unchanged": True,
        "projection_unchanged": True, "evaluator_policy_unchanged": True,
        "arms": [x["arm_id"] for x in arms], "future_api_calls": 8
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
