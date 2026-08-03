from dataclasses import dataclass, field


@dataclass
class EmotionalContext:
    """Эмоциональная форма ответа без его текстового содержания."""

    tone: str = "calm"
    empathy_level: str = "medium"
    speech_confidence: str = "calm"
    explanation_depth: str = "medium"
    initiative_level: str = "balanced"
    directness: str = "gentle"
    response_length: str = "medium"
    user_specific_notes: list = field(default_factory=list)


class EmotionalEngine:
    """Выбирает форму общения на основе уже принятых Context."""

    def choose(
        self,
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        memory,
    ) -> EmotionalContext:
        if impact_context.main_goal == "reduce_internal_conflict":
            return EmotionalContext(
                tone="calm_supportive",
                empathy_level="high",
                speech_confidence="calm",
                explanation_depth="brief",
                initiative_level="guided",
                directness="gentle",
                response_length="short",
                user_specific_notes=["avoid_pressure"],
            )

        if impact_context.main_goal == "increase_self_understanding":
            return EmotionalContext(
                tone="reflective",
                empathy_level="medium",
                speech_confidence="calm",
                explanation_depth="moderate",
                initiative_level="inviting",
                directness="gentle",
                response_length="medium",
                user_specific_notes=["invite_self_observation"],
            )

        if impact_context.main_goal == "support_shared_investigation":
            return EmotionalContext(
                tone="curious_supportive",
                empathy_level="medium",
                speech_confidence="tentative",
                explanation_depth="brief",
                initiative_level="collaborative",
                directness="tentative",
                response_length="medium",
                user_specific_notes=["allow_correction"],
            )

        return EmotionalContext(
            tone="calm",
            empathy_level="medium",
            speech_confidence="calm",
            explanation_depth="brief",
            initiative_level="balanced",
            directness="gentle",
            response_length="short",
            user_specific_notes=["maintain_dialogue_momentum"],
        )
