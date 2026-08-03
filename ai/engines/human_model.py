from dataclasses import dataclass, field


@dataclass
class HumanModel:
    """Текущее понимание человека, отдельное от истории и состояния диалога."""

    facts: dict = field(default_factory=dict)
    goals: list = field(default_factory=list)
    needs: list = field(default_factory=list)
    motivation: list = field(default_factory=list)
    values: list = field(default_factory=list)
    lifestyle: list = field(default_factory=list)
    life_situation: list = field(default_factory=list)
    habits: list = field(default_factory=list)
    emotional_features: list = field(default_factory=list)
    barriers: list = field(default_factory=list)
    fears: list = field(default_factory=list)
    doubts: list = field(default_factory=list)
    beliefs: list = field(default_factory=list)
    experience: list = field(default_factory=list)
    communication_style: str = "unknown"
    trust: float = 0.0
    hypotheses: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    pain_map: dict = field(default_factory=dict)
    consultation_stage: str = "unknown"
    understanding_score: dict = field(
        default_factory=lambda: {
            "problem": 0.0,
            "motivation": 0.0,
            "pain": 0.0,
            "fear": 0.0,
            "self_perception": 0.0,
            "limitations": 0.0,
            "trust": 0.0,
        }
    )

    # Временная совместимость с текущим маршрутом генерации ответа.
    known: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
