from copy import deepcopy

from ai.engines.human_model import HumanModel


class HumanModelEngine:
    """Создаёт и обновляет только Human Model."""

    def build(
        self,
        semantic_context,
        previous_human_model,
        memory,
    ) -> HumanModel:
        model = deepcopy(previous_human_model) if previous_human_model else HumanModel()

        memory_facts = memory.get("facts", {})
        technical_fields = {
            "history",
            "human_model",
            "last_question",
            "discovery_complete",
            "conversation_state",
        }
        confirmed_facts = {
            key: value
            for key, value in memory_facts.items()
            if (
                key not in technical_fields
                and value is not None
                and value is not False
            )
        }
        model.facts.update(confirmed_facts)
        model.facts.update(semantic_context.facts)

        if semantic_context.goal and semantic_context.goal not in model.goals:
            model.goals.append(semantic_context.goal)

        for constraint in semantic_context.constraints:
            if constraint not in model.barriers:
                model.barriers.append(constraint)

        for signal in semantic_context.emotional_signals:
            if signal not in model.emotional_features:
                model.emotional_features.append(signal)

        for preference in semantic_context.preferences:
            if preference not in model.communication_style:
                model.communication_style = preference

        reported_topics = model.pain_map.setdefault("reported_topics", [])
        for topic in semantic_context.topics:
            if topic not in reported_topics:
                reported_topics.append(topic)

        model.understanding_score["problem"] = float(bool(reported_topics))
        model.understanding_score["motivation"] = float(
            bool(model.goals or model.motivation)
        )
        model.understanding_score["pain"] = float(
            bool(model.facts.get("duration"))
        )
        model.understanding_score["limitations"] = float(bool(model.barriers))
        model.consultation_stage = (
            "discovery_complete"
            if self.is_discovery_complete(model)
            else "discovery"
        )

        model.known = [
            f"{key}: {value}"
            for key, value in sorted(model.facts.items())
        ]
        model.unknown = []

        return model

    def apply_update(self, profile: dict, update: dict) -> dict:
        safe_update = {}
        immutable_fields = {"name", "gender"}

        for key, value in update.items():
            if value is None:
                continue

            if key in immutable_fields and profile.get(key) is not None:
                continue

            safe_update[key] = value

        return safe_update

    def is_discovery_complete(self, model_or_profile) -> bool:
        if isinstance(model_or_profile, HumanModel):
            score = model_or_profile.understanding_score
            required_areas = ("problem", "motivation", "pain", "limitations")
            return all(score.get(area, 0.0) >= 1.0 for area in required_areas)

        profile = model_or_profile
        required = ["main_problem", "duration"]
        optional = [
            "age",
            "current_weight",
            "target_weight",
            "previous_attempts",
            "failure_reason",
        ]

        if any(not profile.get(field) for field in required):
            return False

        return sum(bool(profile.get(field)) for field in optional) >= 2
