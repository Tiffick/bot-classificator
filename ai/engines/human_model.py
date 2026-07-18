from dataclasses import dataclass, field


@dataclass
class HumanModel:
    """
    Текущее понимание человека.

    Пока содержит только структуру данных.
    Логика наполнения будет появляться постепенно.
    """

    facts: dict = field(default_factory=dict)
    unknown: dict = field(default_factory=dict)
    hypotheses: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    pain_map: dict = field(default_factory=dict)
    motivation: dict = field(default_factory=dict)
    consultation_stage: str = "unknown"