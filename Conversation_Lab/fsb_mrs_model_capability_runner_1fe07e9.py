"""Resume-safe runner for the matched FSB-M/R/S model-capability isolation.

This thin lab-only runner reuses the already validated checkpoint state machine
from the focused regression runner.  It binds the same frozen inputs and
evaluator artifacts to a separate manifest whose sole experimental change is
the production model, gpt-6-astra.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


HEAD = "1fe07e9"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_mrs_live_runner_1fe07e9 as base  # noqa: E402


FREEZE = ROOT / "Conversation_Lab" / "frozen" / "fsb_mrs_model_capability_1fe07e9"
SOURCE = ROOT / "Conversation_Lab" / "frozen" / "fsb_01_mrs_policy_patch_1fe07e9"
RUNS_ROOT = ROOT / "Conversation_Lab" / "traces" / "fsb_mrs_model_capability_1fe07e9_runs"
MANIFEST_NAME = "fsb_mrs_model_capability_manifest_1fe07e9.json"
VALIDATOR_NAME = "fsb_mrs_model_capability_offline_validator_1fe07e9.py"
SELF = Path(__file__).resolve()

FREEZE_SHA256 = {
    "fsb_mrs_model_capability_freeze_report_1fe07e9.md": "6547588EC9B30CF7DD52B7601F298996A156F4C0D6E43B02E0219E66DAAD19D1",
    "fsb_mrs_model_capability_manifest_1fe07e9.json": "5DBD0FA5A58E3B2CE9339952454557A5BD13F0D5009830DEFD1F6519EC862477",
    "fsb_mrs_model_capability_offline_validator_1fe07e9.py": "F4F1DF2D27031C7DED8FEB736461D626005A5168704AA70C765B790516B9BF36",
}


def preflight() -> dict[str, Any]:
    actual_head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_head != HEAD:
        raise base.RunnerError(
            "head_failure", f"HEAD={actual_head}; expected {HEAD}"
        )

    actual_names = {path.name for path in FREEZE.iterdir() if path.is_file()}
    if actual_names != set(FREEZE_SHA256):
        raise base.RunnerError(
            "hash_failure", f"Isolation freeze file set mismatch: {sorted(actual_names)}"
        )
    freeze_hashes: dict[str, str] = {}
    for name, expected in FREEZE_SHA256.items():
        actual = base._sha_file(FREEZE / name)
        if actual != expected:
            raise base.RunnerError(
                "hash_failure", f"Isolation freeze SHA mismatch for {name}"
            )
        freeze_hashes[name] = actual

    manifest = base._read_json(FREEZE / MANIFEST_NAME)
    source_hashes = manifest["source_artifact_sha256"]
    for name, expected in source_hashes.items():
        actual = base._sha_file(SOURCE / name)
        if actual != expected:
            raise base.RunnerError(
                "hash_failure", f"Source freeze SHA mismatch for {name}"
            )

    validator = subprocess.run(
        [sys.executable, str(FREEZE / VALIDATOR_NAME)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if validator.returncode != 0:
        raise base.RunnerError(
            "offline_validation_failure",
            f"Isolation validator exit={validator.returncode}: {validator.stderr}",
        )
    try:
        validation = json.loads(validator.stdout)
    except json.JSONDecodeError as error:
        raise base.RunnerError(
            "offline_validation_failure", "Isolation validator returned non-JSON"
        ) from error
    if validation.get("result") != "READY" or validation.get("api_calls") != 0:
        raise base.RunnerError(
            "offline_validation_failure", f"Isolation result={validation!r}"
        )

    if manifest.get("production_model") != "gpt-6-astra":
        raise base.RunnerError("protocol_failure", "Production model drift")
    if manifest.get("baseline_production_model") != "gpt-5-mini":
        raise base.RunnerError("protocol_failure", "Baseline model drift")
    if manifest.get("evaluator_model") != "gpt-6-astra":
        raise base.RunnerError("protocol_failure", "Evaluator model drift")
    if manifest.get("api_budget") != {
        "production": 3,
        "evaluator": 9,
        "maximum_total": 12,
    }:
        raise base.RunnerError("protocol_failure", "Call budget drift")

    cases = base._read_json(SOURCE / "fsb_mrs_cases_1fe07e9.json")
    challengers = base._read_json(SOURCE / "fsb_mrs_challengers_1fe07e9.json")
    return {
        "head": actual_head,
        "manifest_sha256": freeze_hashes[MANIFEST_NAME],
        "freeze_sha256": {
            "isolation": freeze_hashes,
            "source": source_hashes,
        },
        "manifest": manifest,
        "cases": cases,
        "challengers": challengers,
        "offline_validation": validation,
        "runner_dependency_sha256": base._sha_file(SELF),
        "evaluator_implementation_sha256": base._sha_file(
            ROOT / "Conversation_Lab" / "selected_action_evaluator.py"
        ),
    }


# Rebind only lab-runner locations and preflight. Production modules and frozen
# source artifacts are never modified.
base.PACKAGE = SOURCE
base.RUNS_ROOT = RUNS_ROOT
base.EVALUATOR_PROMPT_NAME = "fsb_mrs_evaluator_prompt_1fe07e9.txt"
base.EVALUATOR_SCHEMA_NAME = "fsb_mrs_evaluator_schema_1fe07e9.json"
base.preflight = preflight
_original_new_run = base._new_run


def _new_run(frozen: dict[str, Any], run_root: Optional[Path] = None):
    return _original_new_run(frozen, RUNS_ROOT if run_root is None else run_root)


base._new_run = _new_run


if __name__ == "__main__":
    raise SystemExit(base.main())
