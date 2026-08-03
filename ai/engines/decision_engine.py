from dataclasses import dataclass, field


@dataclass
class DecisionContext:
    next_goal: str = ""
    strategy: str = ""
    response_type: str = ""
    confidence: dict = field(default_factory=dict)


class DecisionEngine:

    def decide(self, human_model, reasoning_context) -> DecisionContext:
        return DecisionContext()
