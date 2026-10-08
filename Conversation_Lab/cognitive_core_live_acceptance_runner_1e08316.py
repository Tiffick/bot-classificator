"""Exactly-five-call live acceptance runner for the uncommitted policy patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEAD = "1e08316"
MODEL = "gpt-5-mini"
ORDER = ["FSB-M", "FSB-R", "FSB-S", "FSB-T", "FSB-D"]
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_01_live_runner_672455a as base  # noqa: E402
from ai.cognitive_core import (  # noqa: E402
    COGNITIVE_TURN_JSON_SCHEMA,
    CognitiveCore,
)


MANIFEST_PATH = (
    ROOT / "Conversation_Lab/frozen/cognitive_core_live_acceptance_1e08316/manifest.json"
)
HISTORICAL_CASES = (
    ROOT / "Conversation_Lab/frozen/fsb_01_672455a/fsb_01_cases_672455a.json"
)
D_CASE = (
    ROOT / "Conversation_Lab/frozen/fsb_01_revised_policy_1e08316/fsb_d_case_1e08316.json"
)
RUNS_ROOT = ROOT / "Conversation_Lab/traces/cognitive_core_live_acceptance_1e08316_runs"


def sha_value(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def load_frozen() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != HEAD:
        raise RuntimeError(f"HEAD={head}; expected {HEAD}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["order"] != ORDER or manifest["budget"] != {
        "production_calls": 5, "evaluator_calls": 0, "retries": 0,
        "confirmations": 0, "smoke_calls": 0, "adaptive_calls": 0,
        "maximum_total": 5,
    }:
        raise RuntimeError("Frozen order or budget changed")
    prompt_sha = hashlib.sha256(CognitiveCore._system_prompt().encode("utf-8")).hexdigest().upper()
    if prompt_sha != manifest["production_prompt_sha256"]:
        raise RuntimeError("Production prompt changed")
    if sha_value(COGNITIVE_TURN_JSON_SCHEMA) != manifest["wire_schema_sha256"]:
        raise RuntimeError("Wire schema changed")

    payload = json.loads(HISTORICAL_CASES.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in payload["cases"]}
    cases["FSB-D"] = json.loads(D_CASE.read_text(encoding="utf-8"))
    requests: dict[str, Any] = {}
    for case_id in ORDER:
        case = cases[case_id]
        expected = manifest["cases"][case_id]
        if sha_value(case) != expected["case_sha256"]:
            raise RuntimeError(f"Case drift: {case_id}")
        if case.get("state_sha256") != expected["state_sha256"]:
            raise RuntimeError(f"State drift: {case_id}")
        current, human_model, previous_acs, history = base._load_production_input(case)
        request = {
            "model": MODEL,
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
        if sha_value(request) != expected["request_sha256"]:
            raise RuntimeError(f"Request drift: {case_id}")
        requests[case_id] = request
    return manifest, cases, requests


def run() -> Path:
    from dotenv import load_dotenv
    from openai import OpenAI

    manifest, _, requests = load_frozen()
    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")
    run_id = (
        "ccaccept-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_dir = RUNS_ROOT / run_id
    metadata = {
        "run_id": run_id,
        "head": HEAD,
        "model": MODEL,
        "order": ORDER,
        "manifest_sha256": base._sha_file(MANIFEST_PATH),
        "prompt_sha256": manifest["production_prompt_sha256"],
        "wire_schema_sha256": manifest["wire_schema_sha256"],
        "maximum_api_calls": 5,
        "production_calls": 5,
        "evaluator_calls": 0,
        "retry": 0,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append("RUN_STARTED", metadata=metadata)
    client = OpenAI(max_retries=0, timeout=120.0)

    for index, case_id in enumerate(ORDER, 1):
        request = requests[case_id]
        checkpoint.append(
            "CALL_PREPARED", call_index=index, case_id=case_id,
            model=MODEL, request_sha256=sha_value(request),
            acceptance=manifest["cases"][case_id]["acceptance"],
        )
        try:
            response = client.chat.completions.create(**request)
        except Exception as error:
            checkpoint.append(
                "STOP", category="api_failure", call_index=index,
                case_id=case_id, error_type=type(error).__name__,
                error_message=str(error),
            )
            raise
        base._append_response(
            checkpoint,
            base.CallSpec(
                key=f"production.{case_id}", phase="production",
                case_id=case_id, role="production_decision", model=MODEL,
            ),
            index,
            response,
            request,
        )
        checkpoint.append(
            "CALL_RECORDED", call_index=index, case_id=case_id,
            raw_structured_content=response.choices[0].message.content,
        )

    checkpoint.append(
        "RUN_COMPLETED", api_calls=5, production_calls=5,
        evaluator_calls=0,
    )
    print(f"RUN_ID={run_id}")
    print(f"CHECKPOINT={checkpoint.path}")
    return checkpoint.path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
