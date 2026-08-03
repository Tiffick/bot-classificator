from dataclasses import dataclass, field


@dataclass
class ImpactContext:
    expected_effect: str = ""
    risks: list = field(default_factory=list)
    follow_up: str = ""
    confidence: dict = field(default_factory=dict)


class ImpactEngine:

    def evaluate(
        self,
        human_model,
        reasoning_context,
        decision_context,
    ) -> ImpactContext:
        return ImpactContext()
