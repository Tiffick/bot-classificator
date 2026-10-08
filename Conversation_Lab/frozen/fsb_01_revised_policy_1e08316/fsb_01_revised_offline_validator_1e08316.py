"""Offline-only validator for the FSB-01 revised-policy freeze.

This process installs a fail-closed socket audit hook, imports no OpenAI client,
and reports its API-call count as zero. It replays every frozen state through
the real Discovery contracts and deterministic StateApplier.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HEAD = "1e08316"
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
HISTORICAL = ROOT / "Conversation_Lab/frozen/fsb_01_672455a"
PRODUCT = ROOT / "Conversation_Lab/frozen/fsb_m_product_policy_ab_1550ec2"
MANIFEST = PACKAGE / "fsb_01_revised_manifest_1e08316.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA  # noqa: E402
from ai.cognitive_turn import CognitiveTurnResult  # noqa: E402


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest().upper()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_base_validator():
    path = HISTORICAL / "fsb_01_offline_validator_672455a.py"
    spec = importlib.util.spec_from_file_location("fsb01_base_validator", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Cannot load historical replay validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _validate_pair(pair: dict[str, Any], manifest: dict[str, Any]) -> None:
    if set(pair) != {"case_id", "direct_discrimination", "testable_working_version"}:
        raise AssertionError("FSB-D pair fields changed")
    if pair["case_id"] != "FSB-D":
        raise AssertionError("FSB-D pair case changed")
    required = {
        "strategy", "decision_intent", "action_content_kind", "semantic_content",
        "response_target_interaction", "reply_text",
    }
    direct = pair["direct_discrimination"]
    working = pair["testable_working_version"]
    if set(direct) != required or set(working) != required:
        raise AssertionError("FSB-D action projection fields changed")
    if direct["strategy"] != "direct_discrimination" or direct["response_target_interaction"] != "open_response":
        raise AssertionError("FSB-D direct action semantics changed")
    if working["strategy"] != "testable_working_version" or working["response_target_interaction"] != "evaluation":
        raise AssertionError("FSB-D working-version semantics changed")
    direct_text = canonical(direct).decode("utf-8").lower()
    working_text = canonical(working).decode("utf-8").lower()
    for token in ("до тяжёлого разговора", "после него", "не удаётся вспомнить"):
        if token not in direct_text:
            raise AssertionError(f"FSB-D direct action lost available observation: {token}")
    for token in ("пропущенного обеда", "тяжёлого разговора", "рабочая версия", "не установленная причина", "частично", "не знаете"):
        if token not in working_text:
            raise AssertionError(f"FSB-D working version lost a guard/value: {token}")
    if sha_value(direct) != manifest["fsb_d"]["direct_action_sha256"]:
        raise AssertionError("FSB-D direct action hash mismatch")
    if sha_value(working) != manifest["fsb_d"]["working_version_sha256"]:
        raise AssertionError("FSB-D working-version hash mismatch")


def validate() -> dict[str, Any]:
    network_events: list[str] = []

    def audit(event: str, _args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex"}:
            network_events.append(event)
            raise RuntimeError("Network access is forbidden in offline validation")

    sys.addaudithook(audit)
    actual_head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_head != HEAD:
        raise AssertionError(f"HEAD={actual_head}; expected {HEAD}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["head"] != HEAD:
        raise AssertionError("Manifest HEAD changed")
    for name, expected in manifest["historical_source"]["sha256"].items():
        if sha_file(HISTORICAL / name) != expected:
            raise AssertionError(f"Historical artifact changed: {name}")
    for name, expected in manifest["product_policy_source"]["sha256"].items():
        if sha_file(PRODUCT / name) != expected:
            raise AssertionError(f"Product-policy artifact changed: {name}")
    for name, expected in manifest["new_artifact_sha256"].items():
        if sha_file(PACKAGE / name) != expected:
            raise AssertionError(f"New frozen artifact changed: {name}")

    prompt = _git_bytes(manifest["production_prompt"]["git_path"])
    if hashlib.sha256(prompt).hexdigest().upper() != manifest["production_prompt"]["sha256"]:
        raise AssertionError("Production prompt binding changed")
    boundary = manifest["contract_boundary"]
    if sha_file(ROOT / "ai/cognitive_core.py") != boundary["cognitive_core_file_sha256"]:
        raise AssertionError("CognitiveCore implementation changed")
    if sha_file(ROOT / "ai/cognitive_turn.py") != boundary["cognitive_turn_file_sha256"]:
        raise AssertionError("CognitiveTurn implementation changed")
    if sha_value(COGNITIVE_TURN_JSON_SCHEMA) != boundary["cognitive_core_wire_schema_sha256"]:
        raise AssertionError("Wire schema changed")
    if sha_value(CognitiveTurnResult.model_json_schema()) != boundary["cognitive_turn_schema_sha256"]:
        raise AssertionError("CognitiveTurn schema changed")
    if sha_file(ROOT / "Conversation_Lab/selected_action_evaluator.py") != boundary["selected_action_evaluator_file_sha256"]:
        raise AssertionError("Selected-action evaluator changed")

    base = _load_base_validator()
    cases_doc = json.loads((HISTORICAL / "fsb_01_cases_672455a.json").read_text(encoding="utf-8"))
    expected_order = ["FSB-H", "FSB-M", "FSB-R", "FSB-S", "FSB-T", "FSB-A"]
    if [case["case_id"] for case in cases_doc["cases"]] != expected_order:
        raise AssertionError("Historical case order/meaning changed")
    replay_hashes: dict[str, str] = {}
    for case in cases_doc["cases"]:
        replayed = base.replay_serialized(case)
        if canonical(replayed) != canonical(case["frozen_input"]):
            raise AssertionError(f"{case['case_id']}: replay drift")
        state_hash = sha_value(replayed)
        if state_hash != case["state_sha256"]:
            raise AssertionError(f"{case['case_id']}: state hash drift")
        replay_hashes[case["case_id"]] = state_hash
    base.validate_ambiguity_probe(cases_doc["cases"][-1])

    d_case = json.loads((PACKAGE / "fsb_d_case_1e08316.json").read_text(encoding="utf-8"))
    replayed_d = base.replay_serialized(d_case)
    if canonical(replayed_d) != canonical(d_case["frozen_input"]):
        raise AssertionError("FSB-D replay drift")
    if sha_value(replayed_d) != manifest["fsb_d"]["state_sha256"]:
        raise AssertionError("FSB-D state hash mismatch")
    if sha_value(d_case) != manifest["fsb_d"]["case_sha256"]:
        raise AssertionError("FSB-D case hash mismatch")
    current = d_case["frozen_input"]["current_user_message"]["text"].lower()
    for token in ("пропустил обед", "тяжёлый разговор", "хорошо помню", "до разговора", "после него"):
        if token not in current:
            raise AssertionError(f"FSB-D context lost required grounding: {token}")

    pair = json.loads((PACKAGE / "fsb_d_policy_pair_1e08316.json").read_text(encoding="utf-8"))
    _validate_pair(pair, manifest)
    m_pair = json.loads((PRODUCT / "fsb_m_product_policy_pair_1550ec2.json").read_text(encoding="utf-8"))
    if sha_value(m_pair["action_b"]) != manifest["product_policy_source"]["fsb_m_working_version_sha256"]:
        raise AssertionError("Frozen FSB-M working version changed")

    if manifest["production_order"] != expected_order + ["FSB-D"]:
        raise AssertionError("Production order changed")
    if manifest["qualification"]["direct_on_left"] != [True, False, True, False, True, False]:
        raise AssertionError("Qualification orientations changed")
    if manifest["api_budget"] != {
        "qualification_evaluator": 6, "production": 7,
        "posthoc_evaluator": 21, "maximum_evaluator": 27,
        "maximum_total": 34, "qualification_only_maximum_total": 6,
    }:
        raise AssertionError("API budget changed")
    protocol = manifest["protocol"]
    if protocol["retry"] != 0 or protocol["smoke_calls"] != 0 or protocol["confirmations"] != 0 or protocol["adaptive_sampling"] or protocol["astra_production"]:
        raise AssertionError("Forbidden live behavior enabled")
    if not protocol["qualification_only_phase_required"] or not protocol["production_requires_explicit_resume_production"]:
        raise AssertionError("Qualification review authorization boundary changed")
    if network_events:
        raise AssertionError(f"Network events occurred: {network_events}")
    return {
        "result": "READY", "api_calls": 0, "network_events": 0,
        "head": actual_head, "historical_replay": "6/6",
        "historical_state_sha256": replay_hashes,
        "fsb_d_replay": "PASS", "fsb_d_state_sha256": sha_value(replayed_d),
        "qualification_calls": 6, "qualification_only_maximum_calls": 6,
        "maximum_live_calls": 34,
    }


def main() -> int:
    # StateApplier allocates deterministic UUIDs, but allocation order can pass
    # through set iteration.  Re-exec with a frozen hash seed so independent
    # validator processes reproduce the same byte-identical snapshots.
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode
    print(json.dumps(validate(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
