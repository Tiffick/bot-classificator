from dataclasses import dataclass, field


@dataclass
class ImpactContext:
    """Ожидаемое полезное изменение после следующего ответа."""

    main_goal: str = "unknown"
    expected_understanding_change: str = ""
    expected_emotional_change: str = ""
    expected_attitude_change: str = ""
    expected_readiness_change: str = ""
    success_criterion: str = ""
    confidence: float = 0.0

    # Временная совместимость с текущими точками интеграции.
    expected_effect: str = ""
    risks: list = field(default_factory=list)
    follow_up: str = ""


class ImpactEngine:
    """Преобразует принятое решение в цель коммуникационного воздействия."""

    def evaluate(
        self,
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        memory,
    ) -> ImpactContext:
        if decision_context.next_goal == "clarify_contradiction":
            return ImpactContext(
                main_goal="reduce_internal_conflict",
                expected_understanding_change="user_recognizes_the_contradiction",
                expected_emotional_change="reduced_tension",
                expected_attitude_change="more_open_to_exploration",
                expected_readiness_change="ready_to_clarify_position",
                success_criterion="contradiction_is_acknowledged_without_pressure",
                confidence=decision_context.confidence,
                expected_effect="clarify_contradiction",
                follow_up="continue_exploration",
            )

        if decision_context.next_goal.startswith("clarify_"):
            area = decision_context.next_goal.removeprefix("clarify_")
            return ImpactContext(
                main_goal="increase_self_understanding",
                expected_understanding_change=f"understanding_of_{area}_increases",
                expected_emotional_change="calm_attention",
                expected_attitude_change="more_open_to_reflection",
                expected_readiness_change="ready_to_share_relevant_information",
                success_criterion=f"user_provides_or_recognizes_{area}",
                confidence=decision_context.confidence,
                expected_effect=f"clarify_{area}",
                follow_up="continue_exploration",
            )

        if decision_context.next_goal == "verify_hypothesis":
            return ImpactContext(
                main_goal="support_shared_investigation",
                expected_understanding_change="hypothesis_becomes_clearer",
                expected_emotional_change="psychological_safety",
                expected_attitude_change="curiosity_about_own_experience",
                expected_readiness_change="ready_to_confirm_or_refute_hypothesis",
                success_criterion="user_can_relate_to_or_correct_the_hypothesis",
                confidence=decision_context.confidence,
                expected_effect="verify_hypothesis",
                follow_up="update_understanding",
            )

        return ImpactContext(
            main_goal="maintain_consultative_progress",
            expected_understanding_change="current_situation_remains_clear",
            expected_emotional_change="stable_support",
            expected_attitude_change="continued_engagement",
            expected_readiness_change="ready_for_next_consultation_step",
            success_criterion="conversation_progresses_without_pressure",
            confidence=decision_context.confidence,
            expected_effect="continue_consultation",
            follow_up="continue_consultation",
        )
