"""Build and validate the offline FSB-M/R/S policy-patch regression freeze.

The three semantic cases and challengers are imported from the immutable FSB-01
freeze at 672455a.  State snapshots are rebuilt through that freeze's real
Discovery-contract/StateApplier replay code.  This module performs no API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HEAD = "1fe07e9"
SOURCE_HEAD = "672455a"
CASE_ORDER = ("FSB-M", "FSB-R", "FSB-S")
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
SOURCE_PACKAGE = PACKAGE.parent / "fsb_01_672455a"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES_PATH = PACKAGE / f"fsb_mrs_cases_{HEAD}.json"
CHALLENGERS_PATH = PACKAGE / f"fsb_mrs_challengers_{HEAD}.json"
PROMPT_PATH = PACKAGE / f"fsb_mrs_evaluator_prompt_{HEAD}.txt"
SCHEMA_PATH = PACKAGE / f"fsb_mrs_evaluator_schema_{HEAD}.json"
PLAN_PATH = PACKAGE / f"fsb_mrs_measurement_plan_{HEAD}.md"
MANIFEST_PATH = PACKAGE / f"fsb_mrs_manifest_{HEAD}.json"
REPORT_PATH = PACKAGE / f"fsb_mrs_freeze_report_{HEAD}.md"

SOURCE_SHA256 = {
    "fsb_01_cases_672455a.json": "E9CB63B9D54F19EBCEF06A592C83DF70C3E75561C934C0D453BD9D40E914CEA1",
    "fsb_01_challengers_672455a.json": "23F38F097ED4B2A30902D22EEE128CB88EAE9B446EDA5E8C9912F1752960AF51",
    "fsb_01_evaluator_prompt_672455a.txt": "C5ACFB34717B6F548CE7F0F65349E7F0C56DA3203580242E1B40A9BD3DB1FCF6",
    "fsb_01_evaluator_schema_672455a.json": "6D9B6D568D3F58C1D622F18B93FB3D428CA9631AB0B3C8029C134E9270FCB54B",
    "fsb_01_manifest_672455a.json": "08A4CB3EC1AD94D2DB2A286F4698DDF3EA5504B7105FBEA72CD39EB8A309506A",
    "fsb_01_offline_validator_672455a.py": "B8F71D2DC24631D363E61FF0AABB41BC2806E499EF3979CC066B7FBC4EB837F1",
}
OLD_PRODUCTION_PROMPT_SHA256 = (
    "9F71228617730A13116B02E2A900035D210B327AF359F16F3BF36B3617477BA2"
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest().upper()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_source_builder():
    path = SOURCE_PACKAGE / "fsb_01_offline_validator_672455a.py"
    spec = importlib.util.spec_from_file_location("fsb01_source_builder_672455a", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Cannot load source FSB-01 replay builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source_files() -> None:
    for name, expected in SOURCE_SHA256.items():
        actual = sha_file(SOURCE_PACKAGE / name)
        if actual != expected:
            raise AssertionError(f"Source freeze SHA mismatch: {name}")


def build_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    source = load_source_builder()
    old_cases = read_json(SOURCE_PACKAGE / "fsb_01_cases_672455a.json")
    old_challengers = read_json(
        SOURCE_PACKAGE / "fsb_01_challengers_672455a.json"
    )
    old_case_map = {case["case_id"]: case for case in old_cases["cases"]}
    old_challenger_map = {
        entry["case_id"]: entry for entry in old_challengers["challengers"]
    }

    cases = []
    challengers = []
    for case_id in CASE_ORDER:
        rebuilt = source.build_case(case_id)
        replayed = source.replay_serialized(rebuilt)
        if canonical(replayed) != canonical(rebuilt["frozen_input"]):
            raise AssertionError(f"{case_id}: deterministic replay mismatch")
        if canonical(rebuilt) != canonical(old_case_map[case_id]):
            raise AssertionError(f"{case_id}: semantic case drift")
        challenger = source.resolve_challenger(rebuilt)
        if canonical(challenger) != canonical(old_challenger_map[case_id]):
            raise AssertionError(f"{case_id}: challenger drift")
        cases.append(rebuilt)
        challengers.append(challenger)

    return (
        {"head": HEAD, "source_freeze_head": SOURCE_HEAD, "cases": cases},
        {
            "head": HEAD,
            "source_freeze_head": SOURCE_HEAD,
            "freeze_timing": old_challengers["freeze_timing"],
            "challengers": challengers,
        },
    )


MEASUREMENT_PLAN = """# FSB-M/R/S focused policy-patch regression

## Question

Does the Decision-policy patch at HEAD 1fe07e9 remove the three previously
reproducible MISSED HIGH-VALUE ALTERNATIVE results without technical,
intent-semantic, or Product Boundary regression?

No inference about latent consideration, hidden cognition, candidate generation,
ranking, rationale, or chain-of-thought is permitted.

## Frozen execution

1. Verify HEAD, every package SHA, and the offline validator before any API call.
2. Run exactly one selected-only `CognitiveCore.propose()` call for FSB-M, FSB-R,
   and FSB-S in that order with `gpt-5-mini`, retry 0.
3. Only after all three production outputs are structurally valid, run exactly
   three blind `gpt-6-astra` judgments per case in frozen case/orientation order.
4. Use the existing `selected_action_evaluator` normalization and aggregation.
5. No adaptive confirmation, smoke, challenger generation, retry, or extra call.

Exact future budget: 3 production + 9 evaluator = 12 API calls.

## Frozen interpretation

- **PATCH REGRESSION PASS**: no case aggregates to `CHALLENGER_HIGHER`; every
  production output is valid; there is no semantic intent mismatch or new
  Product Boundary violation.
- **PATCH REGRESSION FAIL**: at least one valid comparison aggregates to
  `CHALLENGER_HIGHER`.
- **INCONCLUSIVE**: technical/structural failure, critical `ABSTAIN`, state or
  challenger drift, or evaluator protocol violation.

DecisionIntent family is not gold. Comparative quality against the independently
frozen challenger is primary.
"""


def make_manifest(cases_doc: dict[str, Any], challengers_doc: dict[str, Any]) -> dict[str, Any]:
    from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA
    from ai.cognitive_turn import CognitiveTurnResult

    prompt = ROOT / "ai" / "prompts" / "cognitive_core_system_prompt.txt"
    core = ROOT / "ai" / "cognitive_core.py"
    turn = ROOT / "ai" / "cognitive_turn.py"
    evaluator = ROOT / "Conversation_Lab" / "selected_action_evaluator.py"
    orientations = {
        "FSB-M": [False, True, False],
        "FSB-R": [True, False, True],
        "FSB-S": [False, True, False],
    }
    return {
        "experiment": "FSB-M/R/S FOCUSED DECISION-POLICY PATCH REGRESSION",
        "head": HEAD,
        "source_freeze_head": SOURCE_HEAD,
        "production_boundary": "SELECTED_ONLY",
        "production_model": "gpt-5-mini",
        "evaluator_model": "gpt-6-astra",
        "retry": 0,
        "smoke_calls": 0,
        "adaptive_confirmation": False,
        "case_order": list(CASE_ORDER),
        "production_order": list(CASE_ORDER),
        "evaluator_order": list(CASE_ORDER),
        "evaluator_orientations_selected_on_left": orientations,
        "api_budget": {"production": 3, "evaluator": 9, "maximum_total": 12},
        "contract_boundary": {
            "cognitive_turn_schema_sha256": sha_value(
                CognitiveTurnResult.model_json_schema()
            ),
            "cognitive_core_wire_schema_sha256": sha_value(
                COGNITIVE_TURN_JSON_SCHEMA
            ),
            "cognitive_turn_file_sha256": sha_file(turn),
            "cognitive_core_file_sha256": sha_file(core),
            "production_prompt_sha256_old_672455a": OLD_PRODUCTION_PROMPT_SHA256,
            "production_prompt_sha256": sha_file(prompt),
            "selected_action_evaluator_file_sha256": sha_file(evaluator),
        },
        "case_sha256": {
            case["case_id"]: sha_value(case) for case in cases_doc["cases"]
        },
        "state_sha256": {
            case["case_id"]: case["state_sha256"] for case in cases_doc["cases"]
        },
        "challenger_sha256": {
            entry["case_id"]: entry["challenger_sha256"]
            for entry in challengers_doc["challengers"]
        },
        "source_artifact_sha256": SOURCE_SHA256,
        "artifact_sha256": {
            CASES_PATH.name: sha_file(CASES_PATH),
            CHALLENGERS_PATH.name: sha_file(CHALLENGERS_PATH),
            PROMPT_PATH.name: sha_file(PROMPT_PATH),
            SCHEMA_PATH.name: sha_file(SCHEMA_PATH),
            PLAN_PATH.name: sha_file(PLAN_PATH),
            Path(__file__).name: sha_file(Path(__file__)),
        },
        "stop_rules": [
            "preflight_head_hash_or_offline_validation_failure_stops_with_zero_api_calls",
            "transport_billing_structural_order_or_mapping_failure_stops_without_retry",
            "no_unfrozen_prompt_contract_state_challenger_evaluator_or_policy_mutation",
            "no_adaptive_confirmation_or_extra_call",
        ],
    }


def write_freeze() -> None:
    verify_source_files()
    cases_doc, challengers_doc = build_documents()
    write_json(CASES_PATH, cases_doc)
    write_json(CHALLENGERS_PATH, challengers_doc)
    PROMPT_PATH.write_bytes(
        (SOURCE_PACKAGE / "fsb_01_evaluator_prompt_672455a.txt").read_bytes()
    )
    SCHEMA_PATH.write_bytes(
        (SOURCE_PACKAGE / "fsb_01_evaluator_schema_672455a.json").read_bytes()
    )
    PLAN_PATH.write_text(MEASUREMENT_PLAN, encoding="utf-8")
    manifest = make_manifest(cases_doc, challengers_doc)
    write_json(MANIFEST_PATH, manifest)
    report = [
        "# FSB-M/R/S focused offline freeze report",
        "",
        f"- HEAD: `{HEAD}`",
        f"- Source freeze: `{SOURCE_HEAD}`",
        "- Reachability: `3/3`",
        "- Semantic states unchanged: `YES`",
        "- Challengers unchanged: `YES`",
        "- Evaluator prompt/schema unchanged: `YES`",
        f"- Production prompt SHA old: `{OLD_PRODUCTION_PROMPT_SHA256}`",
        f"- Production prompt SHA new: `{manifest['contract_boundary']['production_prompt_sha256']}`",
        f"- Wire schema SHA: `{manifest['contract_boundary']['cognitive_core_wire_schema_sha256']}`",
        "- Future live budget: `12`",
        "- Adaptive confirmation: `NO`",
        "- Result: `READY`",
        "",
        "## State hashes",
        "",
    ]
    report.extend(
        f"- `{case_id}`: `{state_hash}`"
        for case_id, state_hash in manifest["state_sha256"].items()
    )
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def validate_package() -> dict[str, Any]:
    verify_source_files()
    if current_head() != HEAD:
        raise AssertionError(f"HEAD must be {HEAD}")
    cases_doc = read_json(CASES_PATH)
    challengers_doc = read_json(CHALLENGERS_PATH)
    manifest = read_json(MANIFEST_PATH)
    rebuilt_cases, rebuilt_challengers = build_documents()
    if canonical(cases_doc) != canonical(rebuilt_cases):
        raise AssertionError("Cases differ from deterministic rebuild")
    if canonical(challengers_doc) != canonical(rebuilt_challengers):
        raise AssertionError("Challengers differ from deterministic rebuild")
    if PROMPT_PATH.read_bytes() != (
        SOURCE_PACKAGE / "fsb_01_evaluator_prompt_672455a.txt"
    ).read_bytes():
        raise AssertionError("Evaluator prompt drift")
    if SCHEMA_PATH.read_bytes() != (
        SOURCE_PACKAGE / "fsb_01_evaluator_schema_672455a.json"
    ).read_bytes():
        raise AssertionError("Evaluator schema drift")
    if PLAN_PATH.read_text(encoding="utf-8") != MEASUREMENT_PLAN:
        raise AssertionError("Measurement plan drift")
    expected_manifest = make_manifest(cases_doc, challengers_doc)
    if canonical(manifest) != canonical(expected_manifest):
        raise AssertionError("Manifest drift")
    if manifest["contract_boundary"]["production_prompt_sha256"] == (
        OLD_PRODUCTION_PROMPT_SHA256
    ):
        raise AssertionError("Production prompt patch is not represented")
    if manifest["api_budget"] != {
        "production": 3,
        "evaluator": 9,
        "maximum_total": 12,
    }:
        raise AssertionError("Call budget drift")
    if manifest["adaptive_confirmation"] is not False:
        raise AssertionError("Adaptive confirmation must be disabled")
    return {
        "result": "READY",
        "head": HEAD,
        "reachability": "3/3",
        "cases": list(CASE_ORDER),
        "state_sha256": manifest["state_sha256"],
        "challenger_sha256": manifest["challenger_sha256"],
        "semantic_states_unchanged": "YES",
        "challengers_unchanged": "YES",
        "evaluator_policy_unchanged": "YES",
        "production_prompt_sha256_old": OLD_PRODUCTION_PROMPT_SHA256,
        "production_prompt_sha256_new": manifest["contract_boundary"][
            "production_prompt_sha256"
        ],
        "wire_schema_sha256": manifest["contract_boundary"][
            "cognitive_core_wire_schema_sha256"
        ],
        "future_api_calls": 12,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-freeze", action="store_true")
    args = parser.parse_args()
    if args.write_freeze:
        write_freeze()
    print(json.dumps(validate_package(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
