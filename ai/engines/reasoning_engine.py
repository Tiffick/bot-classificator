from dataclasses import dataclass, field


@dataclass
class ReasoningContext:
    """Объяснимые результаты анализа без принятия решения."""

    conclusions: list = field(default_factory=list)
    hypotheses: dict = field(default_factory=dict)
    confirmed_hypotheses: list = field(default_factory=list)
    refuted_hypotheses: list = field(default_factory=list)
    uncertainties: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    missing_information: list = field(default_factory=list)
    priorities: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    foundations: dict = field(default_factory=dict)


class ReasoningEngine:
    """Анализирует Human Model, не меняя её и не выбирая действие."""

    def reason(self, semantic_context, human_model, memory) -> ReasoningContext:
        context = ReasoningContext()

        if human_model.facts:
            context.conclusions.append(
                {
                    "name": "confirmed_facts_available",
                    "fields": sorted(human_model.facts),
                }
            )
            context.confidence["confirmed_facts_available"] = 1.0
            context.foundations["confirmed_facts_available"] = "human_model.facts"

        if semantic_context.goal and semantic_context.goal in human_model.goals:
            context.conclusions.append(
                {
                    "name": "explicit_goal",
                    "value": semantic_context.goal,
                }
            )
            context.confidence["explicit_goal"] = semantic_context.confidence
            context.foundations["explicit_goal"] = "semantic_context.goal"

        if human_model.barriers and "frustration" in human_model.emotional_features:
            hypothesis_name = "change_difficulty_requires_clarification"
            context.hypotheses[hypothesis_name] = {
                "status": "working",
                "basis": ["human_model.barriers", "human_model.emotional_features"],
            }
            context.confidence[hypothesis_name] = 0.6
            context.foundations[hypothesis_name] = [
                "barrier_present",
                "frustration_signal_present",
            ]

        if semantic_context.goal == "lose_weight" and any(
            "не хочу" in barrier for barrier in human_model.barriers
        ):
            context.contradictions.append(
                {
                    "name": "stated_goal_conflicts_with_stated_barrier",
                    "goal": semantic_context.goal,
                }
            )

        areas = {
            "goals": human_model.goals,
            "motivation": human_model.motivation,
            "barriers": human_model.barriers,
            "trust": [human_model.trust] if human_model.trust else [],
        }
        for area, values in areas.items():
            if not values:
                context.missing_information.append(area)

        if not semantic_context.facts and semantic_context.confidence < 1.0:
            context.uncertainties.append(
                "current_message_contains_no_confirmed_facts"
            )

        if memory.get("facts") and not human_model.facts:
            context.uncertainties.append(
                "memory_facts_not_reflected_in_human_model"
            )

        context.priorities = context.missing_information.copy()

        return context
