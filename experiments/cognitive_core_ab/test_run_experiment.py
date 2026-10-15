"""Offline regression tests for the lab-only A/B/C/D harness."""

from copy import deepcopy

import pytest

from experiments.cognitive_core_ab import run_experiment as experiment


def test_reconstruct_model_replaces_incompatible_frozendict_deepcopy() -> None:
    scenario = next(
        row for row in experiment.scenarios() if row.scenario_id == "S3"
    )
    assert scenario.previous_acs is not None

    with pytest.raises(TypeError, match="immutable"):
        deepcopy(scenario.previous_acs)

    reconstructed = experiment.reconstruct_model(scenario.previous_acs)

    assert reconstructed is not scenario.previous_acs
    assert reconstructed.active_content is not scenario.previous_acs.active_content
    assert reconstructed.response_targets is not scenario.previous_acs.response_targets
    assert reconstructed.model_dump(mode="json") == scenario.previous_acs.model_dump(
        mode="json"
    )


@pytest.mark.parametrize("scenario_id", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_each_frozen_scenario_round_trips_and_adapts_offline(
    scenario_id: str,
) -> None:
    scenario = next(
        row for row in experiment.scenarios() if row.scenario_id == scenario_id
    )

    result = experiment.adapt_semantic_turn(
        experiment._offline_probe_payload(),
        current_user_text=experiment.reconstruct_model(scenario.current).text,
        human_model=experiment.reconstruct_model(scenario.human_model),
        previous_acs=experiment.reconstruct_model(scenario.previous_acs),
    )

    assert result.decision_intent.value == "reflection"
    assert "".join(segment.text for segment in result.reply_segments) == (
        "Техническая проверка."
    )


def test_fixture_payloads_and_request_settings_are_deterministic() -> None:
    first = {
        row.scenario_id: experiment._sha(
            experiment._json_bytes(experiment._input_payload(row))
        )
        for row in experiment.scenarios()
    }
    second = {
        row.scenario_id: experiment._sha(
            experiment._json_bytes(experiment._input_payload(row))
        )
        for row in experiment.scenarios()
    }
    assert first == second

    production_prompt = experiment._production_system_prompt()
    scenario = experiment.scenarios()[0]
    for condition in ("A", "B", "C", "D"):
        request = experiment.request_for(
            {"condition": condition}, scenario, production_prompt
        )
        assert request["model"] == "gpt-5-mini-2025-08-07"
        assert request["reasoning_effort"] == "minimal"
        assert request["max_completion_tokens"] == 4000
        assert ("response_format" in request) is (condition != "C")


def test_all_fixture_probe_hashes_are_available_without_api_calls() -> None:
    hashes = experiment.validate_scenario_fixtures()

    assert set(hashes) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert all(len(value) == 64 for value in hashes.values())


def test_failed_pilot_is_not_a_resumable_run() -> None:
    failed = (
        experiment.RESULTS_ROOT
        / "cc-abcd-v2-20261011T021544Z-9f1be97e"
    )

    assert (failed / "PILOT_FAILED.json").exists()
    assert not (
        (failed / "manifest.json").exists()
        and not (failed / "RUN_COMPLETED").exists()
        and not (failed / "PILOT_FAILED").exists()
        and not (failed / "PILOT_FAILED.json").exists()
    )
