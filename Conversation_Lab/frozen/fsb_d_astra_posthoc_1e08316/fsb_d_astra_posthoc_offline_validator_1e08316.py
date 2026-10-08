"""Offline validator for the frozen FSB-D Astra post-hoc comparison."""

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
MANIFEST = PACKAGE / "fsb_d_astra_posthoc_manifest_1e08316.json"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sha_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(message)


def evaluator_request(
    *, model: str, prompt: str, schema: dict[str, Any], context: dict[str, Any],
    left: dict[str, Any], right: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(
                {"frozen_context": context, "left_action": left, "right_action": right},
                ensure_ascii=False, sort_keys=True,
            )},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "fsb_d_astra_posthoc_verdict",
                "strict": True,
                "schema": schema,
            },
        },
    }


def main() -> int:
    manifest = read_json(MANIFEST)
    actual_head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_head != HEAD or manifest["head"] != HEAD:
        fail(f"HEAD mismatch: {actual_head}")

    paths = {
        "checkpoint": manifest["source_run"]["checkpoint_path"],
        "run_json": manifest["source_run"]["run_json_path"],
        "source_runner": manifest["source_run"]["runner_path"],
        "case": manifest["frozen_context"]["case_path"],
        "pair": manifest["direct_benchmark"]["pair_path"],
        "prompt": manifest["evaluator_contract"]["prompt_path"],
        "schema": manifest["evaluator_contract"]["schema_path"],
    }
    expected = {
        "checkpoint": manifest["source_run"]["checkpoint_sha256"],
        "run_json": manifest["source_run"]["run_json_sha256"],
        "source_runner": manifest["source_run"]["runner_sha256"],
        "case": manifest["frozen_context"]["case_file_sha256"],
        "pair": manifest["direct_benchmark"]["pair_file_sha256"],
        "prompt": manifest["evaluator_contract"]["prompt_sha256"],
        "schema": manifest["evaluator_contract"]["schema_sha256"],
    }
    for name, relative in paths.items():
        actual = sha_file(ROOT / relative)
        if actual != expected[name]:
            fail(f"SHA mismatch: {name}: {actual}")

    from Conversation_Lab.fsb_01_live_runner_672455a import Checkpoint
    import Conversation_Lab.fsb_d_model_swap_runner_1e08316 as source

    source_frozen = source.preflight()
    run_dir = (ROOT / manifest["source_run"]["checkpoint_path"]).parent
    metadata = read_json(run_dir / "run.json")
    events = Checkpoint(run_dir / "checkpoint.jsonl").read()
    completed = source.validate_checkpoint(
        metadata, events, source_frozen, for_resume=False
    )
    if len(completed) != 3 or events[-1]["event_type"] != "RUN_COMPLETED":
        fail("Source run is not a valid completed three-call run")

    actions: dict[str, dict[str, Any]] = {}
    for sample_id in manifest["sample_order"]:
        matches = [
            event for event in completed
            if event["call_spec"]["case_id"] == sample_id
        ]
        if len(matches) != 1:
            fail(f"Missing unique source action: {sample_id}")
        event = matches[0]
        binding = manifest["saved_actions"][sample_id]
        if event["event_sha256"] != binding["completed_event_sha256"]:
            fail(f"Completed event drift: {sample_id}")
        if event["normalized_result"].get("observable_class") != "OTHER":
            fail(f"Source action is not frozen as OTHER: {sample_id}")
        action = event["normalized_result"]["observable_action"]
        if sha_value(action) != binding["observable_action_sha256"]:
            fail(f"Observable action drift: {sample_id}")
        actions[sample_id] = action

    case = read_json(ROOT / paths["case"])
    pair = read_json(ROOT / paths["pair"])
    direct = {key: value for key, value in pair["direct_discrimination"].items() if key != "strategy"}
    if sha_value(pair["direct_discrimination"]) != manifest["direct_benchmark"]["direct_action_sha256"]:
        fail("Direct action drift")
    if sha_value(direct) != manifest["direct_benchmark"]["direct_projection_sha256"]:
        fail("Direct projection drift")

    prompt = (ROOT / paths["prompt"]).read_text(encoding="utf-8")
    schema = read_json(ROOT / paths["schema"])
    request_hashes: dict[str, list[str]] = {}
    for sample_id in manifest["sample_order"]:
        hashes = []
        for astra_left in manifest["orientation_astra_on_left"][sample_id]:
            left, right = (
                (actions[sample_id], direct) if astra_left else (direct, actions[sample_id])
            )
            hashes.append(sha_value(evaluator_request(
                model=manifest["evaluator_model"], prompt=prompt, schema=schema,
                context=case["frozen_input"], left=left, right=right,
            )))
        if hashes != manifest["expected_request_sha256"][sample_id]:
            fail(f"Evaluator request drift: {sample_id}")
        request_hashes[sample_id] = hashes

    if not manifest["evaluator_contract"]["rubric_reused_byte_identically"]:
        fail("Rubric must be byte-identical")
    if manifest["evaluator_contract"]["semantic_modification"]:
        fail("Semantic rubric modification is forbidden")
    if manifest["api_budget"] != {"production": 0, "evaluator": 9, "maximum_total": 9}:
        fail("Budget mismatch")
    protocol = manifest["protocol"]
    if protocol != {
        "production_calls": 0, "evaluator_calls": 9, "retry": 0,
        "smoke_calls": 0, "confirmations": 0, "adaptive_sampling": False,
        "challenger_generation": False, "all_calls_fixed_before_live": True,
    }:
        fail("Protocol drift")

    print(json.dumps({
        "result": "READY",
        "api_calls": 0,
        "head": actual_head,
        "source_run_validation": "PASS",
        "saved_action_identity": "PASS",
        "direct_identity": "PASS",
        "context_identity": "PASS",
        "rubric_schema_identity": "PASS",
        "request_hashes": request_hashes,
        "budget": manifest["api_budget"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
