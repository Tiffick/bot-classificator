"""Lab-only, post-hoc evaluation of a selected action against one challenger.

This module is deliberately outside the production ``ai`` package.  It neither
calls Cognitive Core nor writes Discovery state.  A caller supplies an already
selected action, an independently frozen challenger, and a blind judge.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Sequence


class BlindVerdict(str, Enum):
    """Verdict available to a judge that only sees anonymous LEFT/RIGHT actions."""

    LEFT_HIGHER = "LEFT_HIGHER"
    RIGHT_HIGHER = "RIGHT_HIGHER"
    EQUIVALENT_OR_BOTH_ACCEPTABLE = "EQUIVALENT_OR_BOTH_ACCEPTABLE"
    INSUFFICIENT_BASIS = "INSUFFICIENT_BASIS"


class ObservableVerdict(str, Enum):
    """Post-hoc verdict expressed relative to the production selection."""

    SELECTED_HIGHER = "SELECTED_HIGHER"
    CHALLENGER_HIGHER = "CHALLENGER_HIGHER"
    EQUIVALENT_OR_BOTH_ACCEPTABLE = "EQUIVALENT_OR_BOTH_ACCEPTABLE"
    ABSTAIN = "ABSTAIN"


class AgreementKind(str, Enum):
    UNANIMOUS = "UNANIMOUS"
    DIRECTIONAL_MAJORITY = "DIRECTIONAL_MAJORITY"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class ObservableAction:
    """The minimum observable action needed for semantic comparison."""

    decision_intent: str
    semantic_focus: str
    basis_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis_evidence_refs", tuple(self.basis_evidence_refs))
        if not self.decision_intent.strip():
            raise ValueError("decision_intent must be non-empty.")
        if not self.semantic_focus.strip():
            raise ValueError("semantic_focus must be non-empty.")
        if any(not reference.strip() for reference in self.basis_evidence_refs):
            raise ValueError("basis_evidence_refs must contain non-empty strings.")


@dataclass(frozen=True)
class EvaluationCase:
    """An immutable pair and canonical frozen context for post-hoc evaluation."""

    case_id: str
    frozen_context_json: str
    selected_action: ObservableAction
    challenger_action: ObservableAction

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty.")
        try:
            json.loads(self.frozen_context_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("frozen_context_json must be valid JSON.") from error

    @classmethod
    def from_context(
        cls,
        *,
        case_id: str,
        frozen_context: Any,
        selected_action: ObservableAction,
        challenger_action: ObservableAction,
    ) -> "EvaluationCase":
        context_json = json.dumps(
            frozen_context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(
            case_id=case_id,
            frozen_context_json=context_json,
            selected_action=selected_action,
            challenger_action=challenger_action,
        )


@dataclass(frozen=True)
class BlindComparison:
    """One orientation-blinded judgment request."""

    frozen_context_json: str
    left_action: ObservableAction
    right_action: ObservableAction

    def context_value(self) -> Any:
        """Return a fresh JSON value so a judge cannot mutate the frozen input."""

        return json.loads(self.frozen_context_json)


@dataclass(frozen=True)
class EvaluationEvidence:
    """Minimal machine-readable aggregation evidence; never free-form reasoning."""

    judgments: tuple[ObservableVerdict, ...]
    vote_counts: tuple[tuple[ObservableVerdict, int], ...]
    agreement: AgreementKind


@dataclass(frozen=True)
class EvaluationResult:
    verdict: ObservableVerdict
    evidence: EvaluationEvidence

    @property
    def missed_high_value_alternative(self) -> bool:
        return self.verdict == ObservableVerdict.CHALLENGER_HIGHER


BlindJudge = Callable[[BlindComparison], BlindVerdict]


def selected_action_from_cognitive_turn(turn_result: Any) -> ObservableAction:
    """Create a read-only lab projection without modifying the production DTO."""

    intent = getattr(turn_result.decision_intent, "value", turn_result.decision_intent)
    semantic_parts = tuple(
        content.semantic_content for content in turn_result.system_action.contents
    )
    if not semantic_parts:
        semantic_parts = tuple(segment.text for segment in turn_result.reply_segments)
    return ObservableAction(
        decision_intent=str(intent),
        semantic_focus="\n".join(semantic_parts),
    )


def evaluate_pair(
    case: EvaluationCase,
    judge: BlindJudge,
    *,
    selected_on_left: Sequence[bool] = (True, False, True),
) -> EvaluationResult:
    """Collect exactly three blind judgments and aggregate them deterministically."""

    if len(selected_on_left) != 3 or set(selected_on_left) != {True, False}:
        raise ValueError("Exactly three judgments with both orientations are required.")

    normalized: list[ObservableVerdict] = []
    for is_selected_left in selected_on_left:
        comparison = BlindComparison(
            frozen_context_json=case.frozen_context_json,
            left_action=(
                case.selected_action if is_selected_left else case.challenger_action
            ),
            right_action=(
                case.challenger_action if is_selected_left else case.selected_action
            ),
        )
        blind_verdict = judge(comparison)
        if not isinstance(blind_verdict, BlindVerdict):
            raise TypeError("judge must return BlindVerdict.")
        normalized.append(_normalize(blind_verdict, is_selected_left))

    return aggregate_judgments(normalized)


def aggregate_judgments(
    judgments: Iterable[ObservableVerdict],
) -> EvaluationResult:
    """Apply the frozen three-repeat policy without confidence scores."""

    values = tuple(judgments)
    if len(values) != 3 or any(
        not isinstance(verdict, ObservableVerdict) for verdict in values
    ):
        raise ValueError("Exactly three ObservableVerdict judgments are required.")

    counts = Counter(values)
    if len(counts) == 1:
        verdict = values[0]
        agreement = AgreementKind.UNANIMOUS
    elif counts[ObservableVerdict.SELECTED_HIGHER] >= 2:
        verdict = ObservableVerdict.SELECTED_HIGHER
        agreement = AgreementKind.DIRECTIONAL_MAJORITY
    elif counts[ObservableVerdict.CHALLENGER_HIGHER] >= 2:
        verdict = ObservableVerdict.CHALLENGER_HIGHER
        agreement = AgreementKind.DIRECTIONAL_MAJORITY
    else:
        verdict = ObservableVerdict.ABSTAIN
        agreement = AgreementKind.INDETERMINATE

    vote_counts = tuple((member, counts[member]) for member in ObservableVerdict)
    return EvaluationResult(
        verdict=verdict,
        evidence=EvaluationEvidence(
            judgments=values,
            vote_counts=vote_counts,
            agreement=agreement,
        ),
    )


def _normalize(
    verdict: BlindVerdict,
    selected_on_left: bool,
) -> ObservableVerdict:
    if verdict == BlindVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE:
        return ObservableVerdict.EQUIVALENT_OR_BOTH_ACCEPTABLE
    if verdict == BlindVerdict.INSUFFICIENT_BASIS:
        return ObservableVerdict.ABSTAIN
    selected_higher = (
        verdict == BlindVerdict.LEFT_HIGHER
        if selected_on_left
        else verdict == BlindVerdict.RIGHT_HIGHER
    )
    return (
        ObservableVerdict.SELECTED_HIGHER
        if selected_higher
        else ObservableVerdict.CHALLENGER_HIGHER
    )
