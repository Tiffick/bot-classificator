from dataclasses import dataclass, field


@dataclass
class ReasoningContext:
    hypotheses: dict = field(default_factory=dict)
    contradictions: list = field(default_factory=list)
    priorities: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)


class ReasoningEngine:

    def reason(self, human_model) -> ReasoningContext:
        return ReasoningContext()
