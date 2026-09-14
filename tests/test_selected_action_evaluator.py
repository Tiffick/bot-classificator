import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai.cognitive_turn import (
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    DecisionIntent,
    ReplySegment,
    SystemAction,
)
from ai.discovery_data_model import (
    ActiveContentKind,
    TargetInteractionKind,
    TargetSubjectKind,
)
from Conversation_Lab.selected_action_evaluator import (
    AgreementKind,
    BlindVerdict,
    EvaluationCase,
    ObservableAction,
    ObservableVerdict,
    aggregate_judgments,
    evaluate_pair,
    selected_action_from_cognitive_turn,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "selected_action_evaluator_frozen_7674220.json"
)


def _frozen_cases():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    state_hashes = payload["source"]["state_sha256"]
    return payload, [
        (
            raw,
            EvaluationCase.from_context(
                case_id=raw["case_id"],
                frozen_context={
                    "snapshot": raw["snapshot"],
                    "state_sha256": state_hashes[raw["snapshot"]],
                },
                selected_action=ObservableAction(**raw["selected_action"]),
                challenger_action=ObservableAction(**raw["challenger_action"]),
            ),
        )
        for raw in payload["cases"]
    ]


def _recorded_judge(verdicts):
    remaining = iter(BlindVerdict(verdict) for verdict in verdicts)

    def judge(_comparison):
        return next(remaining)

    return judge


def test_fixture_identifies_the_original_frozen_pairwise_source():
    payload, _ = _frozen_cases()

    assert payload["source"] == {
        "head": "7674220",
        "pairwise_artifact": "three_condition_matched_live_7674220_pairwise.json",
        "pairwise_sha256": "AFDBAEAC1663E3569C99476FC396AE25C588C966BFE97DEA59C23D4EDC8EE0C0",
        "state_sha256": {
            "T6": "E9F54D5806D61AF3952388A53DE4807E21BA970182B4285E4515000CC12C6CD6",
            "T12": "C93D283943D135AD3CF740181BD684A42640D2B757A375663CF206E79AAC4225",
            "T14": "90E2ECE68C97E793A27449CA9F4195EE0BB1CF8682EBF8C996F0B3C7196B021D",
        },
    }


@pytest.mark.parametrize(
    "case_id, expected",
    [
        ("T6-S2-B", ObservableVerdict.CHALLENGER_HIGHER),
        ("T6-S1-A", ObservableVerdict.SELECTED_HIGHER),
        ("T12-S2-B", ObservableVerdict.SELECTED_HIGHER),
        ("T14-S2-A", ObservableVerdict.SELECTED_HIGHER),
        ("T12-S1-A", ObservableVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE),
        ("T14-S1-B", ObservableVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE),
    ],
)
def test_saved_frozen_pairwise_judgments_replay_deterministically(case_id, expected):
    _, cases = _frozen_cases()
    raw, case = next(entry for entry in cases if entry[0]["case_id"] == case_id)

    result = evaluate_pair(
        case,
        _recorded_judge(raw["blind_verdicts"]),
        selected_on_left=raw["selected_on_left"],
    )

    assert result.verdict == expected == ObservableVerdict(raw["expected"])
    assert result.evidence.agreement == AgreementKind.UNANIMOUS
    assert result.missed_high_value_alternative is (
        expected == ObservableVerdict.CHALLENGER_HIGHER
    )


def test_blind_judge_never_receives_selected_or_challenger_labels():
    _, cases = _frozen_cases()
    _, case = cases[0]
    comparisons = []

    def judge(comparison):
        comparisons.append(comparison)
        return BlindVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE

    evaluate_pair(case, judge)

    assert len(comparisons) == 3
    assert all(not hasattr(comparison, "selected_action") for comparison in comparisons)
    assert all(not hasattr(comparison, "challenger_action") for comparison in comparisons)
    assert comparisons[0].left_action == case.selected_action
    assert comparisons[1].right_action == case.selected_action


def test_two_of_three_directional_judgments_produce_a_majority():
    result = aggregate_judgments(
        (
            ObservableVerdict.CHALLENGER_HIGHER,
            ObservableVerdict.SELECTED_HIGHER,
            ObservableVerdict.CHALLENGER_HIGHER,
        )
    )

    assert result.verdict == ObservableVerdict.CHALLENGER_HIGHER
    assert result.evidence.agreement == AgreementKind.DIRECTIONAL_MAJORITY


def test_frozen_insufficient_basis_verdict_maps_to_observable_abstain():
    _, cases = _frozen_cases()
    _, case = cases[0]

    result = evaluate_pair(
        case,
        lambda _: BlindVerdict.INSUFFICIENT_BASIS,
    )

    assert result.verdict == ObservableVerdict.ABSTAIN
    assert result.evidence.agreement == AgreementKind.UNANIMOUS


@pytest.mark.parametrize(
    "judgments",
    [
        (
            ObservableVerdict.SELECTED_HIGHER,
            ObservableVerdict.CHALLENGER_HIGHER,
            ObservableVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE,
        ),
        (
            ObservableVerdict.ABSTAIN,
            ObservableVerdict.ABSTAIN,
            ObservableVerdict.SELECTED_HIGHER,
        ),
        (
            ObservableVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE,
            ObservableVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE,
            ObservableVerdict.SELECTED_HIGHER,
        ),
    ],
)
def test_substantial_disagreement_or_uncertainty_aggregates_to_abstain(judgments):
    result = aggregate_judgments(judgments)

    assert result.verdict == ObservableVerdict.ABSTAIN
    assert result.evidence.agreement == AgreementKind.INDETERMINATE
    assert result.missed_high_value_alternative is False


def test_evaluator_does_not_mutate_the_production_turn_or_context():
    content = ActionContent(
        local_id="content:question",
        kind=ActiveContentKind.SYSTEM_QUESTION,
        semantic_content="Что обычно происходит после вечернего желания поесть?",
    )
    target = ActionTarget(
        local_id="target:question",
        active_content_local_id=content.local_id,
        subject=ActionSubjectReference(
            kind=TargetSubjectKind.ACTIVE_CONTENT,
            active_content_local_id=content.local_id,
        ),
        interaction=TargetInteractionKind.OPEN_RESPONSE,
    )
    turn = CognitiveTurnResult(
        decision_intent=DecisionIntent.MECHANISM_DISCOVERY,
        system_action=SystemAction(contents=(content,), response_targets=(target,)),
        reply_segments=(
            ReplySegment(
                local_id="segment:reply",
                text="Что обычно происходит после вечернего желания поесть?",
                realizes_action_content_ids=(content.local_id,),
            ),
        ),
    )
    context = {"history": [{"role": "user", "text": "Да, всё так."}]}
    turn_before = turn.model_dump_json()
    context_before = deepcopy(context)
    case = EvaluationCase.from_context(
        case_id="non-intervention",
        frozen_context=context,
        selected_action=selected_action_from_cognitive_turn(turn),
        challenger_action=ObservableAction(
            decision_intent="human_discovery",
            semantic_focus="Explore personal meaning.",
        ),
    )

    evaluate_pair(
        case,
        lambda comparison: (
            comparison.context_value()
            and BlindVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE
        ),
    )

    assert turn.model_dump_json() == turn_before
    assert context == context_before
    assert selected_action_from_cognitive_turn(turn) == case.selected_action


def test_evaluator_requires_exactly_three_judgments_with_both_orientations():
    _, cases = _frozen_cases()
    _, case = cases[0]

    with pytest.raises(ValueError, match="Exactly three"):
        evaluate_pair(
            case,
            lambda _: BlindVerdict.INSUFFICIENT_BASIS,
            selected_on_left=(True,),
        )
    with pytest.raises(ValueError, match="both orientations"):
        evaluate_pair(
            case,
            lambda _: BlindVerdict.INSUFFICIENT_BASIS,
            selected_on_left=(True, True, True),
        )
