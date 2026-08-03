from dataclasses import dataclass


@dataclass
class DecisionContext:
    """Одно объяснимое решение о следующем действии консультанта."""

    next_goal: str = "unknown"
    reason: str = ""
    priority: str = "normal"
    needs_additional_information: bool = False
    ready_for_next_stage: bool = False
    expected_outcome: str = ""
    confidence: float = 0.0

    # Временная совместимость с текущими точками интеграции.
    strategy: str = ""
    response_type: str = ""


class DecisionEngine:
    """Выбирает действие на основе результатов рассуждения."""

    def decide(
        self,
        semantic_context,
        human_model,
        reasoning_context,
        memory,
    ) -> DecisionContext:
        if reasoning_context.contradictions:
            contradiction = reasoning_context.contradictions[0]
            return DecisionContext(
                next_goal="clarify_contradiction",
                reason=contradiction["name"],
                priority="high",
                needs_additional_information=True,
                expected_outcome="contradiction_is_clarified",
                confidence=0.8,
                strategy="clarification",
                response_type="question",
            )

        if reasoning_context.hypotheses:
            hypothesis_name = next(iter(reasoning_context.hypotheses))
            return DecisionContext(
                next_goal="verify_hypothesis",
                reason=f"working_hypothesis:{hypothesis_name}",
                priority="normal",
                needs_additional_information=True,
                expected_outcome="hypothesis_confidence_is_updated",
                confidence=reasoning_context.confidence.get(hypothesis_name, 0.5),
                strategy="verification",
                response_type="question",
            )

        if "duration" in semantic_context.facts:
            return DecisionContext(
                next_goal="clarify_desired_change",
                reason="fact:duration; missing_information:desired_change",
                priority="normal",
                needs_additional_information=True,
                expected_outcome="understanding_of_desired_change_increases",
                confidence=semantic_context.confidence,
                strategy="exploration",
                response_type="question",
            )

        if "weight" in semantic_context.topics:
            return DecisionContext(
                next_goal="clarify_weight_impact",
                reason="topic:weight; missing_information:weight_impact",
                priority="normal",
                needs_additional_information=True,
                expected_outcome="understanding_of_weight_impact_increases",
                confidence=semantic_context.confidence,
                strategy="exploration",
                response_type="question",
            )

        if any(topic in semantic_context.topics for topic in ("energy", "health")):
            return DecisionContext(
                next_goal="clarify_problem_duration",
                reason="topic:energy_or_health; missing_information:problem_duration",
                priority="normal",
                needs_additional_information=True,
                expected_outcome="understanding_of_problem_duration_increases",
                confidence=semantic_context.confidence,
                strategy="exploration",
                response_type="question",
            )

        if reasoning_context.priorities:
            area = reasoning_context.priorities[0]
            return DecisionContext(
                next_goal=f"clarify_{area}",
                reason=f"missing_information:{area}",
                priority="normal",
                needs_additional_information=True,
                expected_outcome=f"understanding_of_{area}_increases",
                confidence=0.6,
                strategy="exploration",
                response_type="question",
            )

        return DecisionContext(
            next_goal="continue_consultation",
            reason="no_critical_uncertainty_or_hypothesis",
            priority="normal",
            ready_for_next_stage=bool(human_model.goals),
            expected_outcome="consultation_progresses",
            confidence=semantic_context.confidence,
            strategy="continuation",
            response_type="statement",
        )
