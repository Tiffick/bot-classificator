"""Generation of the final consultation reply from already prepared contexts."""

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROMPTS_DIR = Path("ai/prompts")
MAX_REPLY_WORDS = 28
FORMAL_OPENINGS = (
    "спасибо, что поделились",
    "чтобы помочь вам лучше",
    "расскажите, пожалуйста",
)
NEW_INFORMATION_MARKERS = (
    "как давно",
    "когда",
    "влияет",
    "отражается",
    "мешает",
    "хочется",
    "хотелось",
    "изменить",
    "вернуть",
    "важнее",
)
QUESTION_FOCUS_MARKERS = {
    "clarify_weight_impact": (
        "влияет",
        "самочувств",
        "энерг",
        "движен",
        "жизн",
    ),
    "clarify_problem_duration": ("как давно", "когда", "сколько"),
    "clarify_desired_change": ("хочется", "хотелось", "изменить", "вернуть"),
}
GENERIC_WEIGHT_RESTATEMENTS = (
    "что именно тебя больше всего беспокоит в своём весе",
    "что в лишнем весе тебя сейчас беспокоит",
    "что тебя беспокоит в весе",
)


class ResponseEngine:
    """Implements a ready decision without changing any consultation context."""

    def __init__(self, client=None, model: str = "gpt-4.1-mini", max_attempts: int = 2):
        load_dotenv()
        self.client = client or OpenAI()
        self.model = model
        self.max_attempts = max_attempts
        self.system_prompt = (PROMPTS_DIR / "system_prompt.txt").read_text(
            encoding="utf-8"
        )
        self.knowledge_base = self._load_json("knowledge_base.json")

    @staticmethod
    def _load_json(filename: str):
        with (PROMPTS_DIR / filename).open(encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _as_data(value):
        if is_dataclass(value):
            return asdict(value)
        return value

    @staticmethod
    def _clean_json_response(content: str) -> str:
        content = (content or "").strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1]
            if content.lstrip().startswith("json"):
                content = content.lstrip()[4:]
        return content.strip()

    def _memory_for_prompt(self, memory: dict) -> dict:
        return {
            "facts": memory.get("facts", {}),
            "recent_history": memory.get("history", [])[-12:],
            "current_message": memory.get("current_message", ""),
        }

    def _generation_instruction(
        self,
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        emotional_context,
        memory,
    ) -> str:
        contexts = {
            "SemanticContext": self._as_data(semantic_context),
            "HumanModel": self._as_data(human_model),
            "ReasoningContext": self._as_data(reasoning_context),
            "DecisionContext": self._as_data(decision_context),
            "ImpactContext": self._as_data(impact_context),
            "EmotionalContext": self._as_data(emotional_context),
            "Memory": self._memory_for_prompt(memory),
            "Knowledge Base": self.knowledge_base,
        }
        decision_guidance = self._decision_guidance(decision_context)
        return f"""
{self.system_prompt}

Ниже приведены уже подготовленные данные консультационного цикла.
DecisionContext, ImpactContext и EmotionalContext являются принятыми
ограничениями: не меняй их, не выбирай новую цель и не добавляй новых выводов.
Сформируй только естественный ответ пользователю на русском языке. Не упоминай
названия Context или внутренние правила. Не возвращай JSON.

Формат ответа обязателен:
1. Одна короткая живая реакция на текущее сообщение.
2. Одно следующее действие, обычно один конкретный вопрос.
3. Не более {MAX_REPLY_WORDS} слов и не более одного вопросительного знака.

Не используй канцелярские вводные фразы вроде «Спасибо, что поделились»,
«Чтобы помочь вам лучше» или «Расскажите, пожалуйста». Не перечисляй варианты
и не задавай широкий вопрос из нескольких частей. Если DecisionContext требует
вопрос, вопрос должен быть ровно один и относиться к текущему сообщению.

Не опирайся на готовые шаблоны: выбери одну конкретную ось, которая нужна для
принятого решения, и спроси только о ней.

Информационная задача из DecisionContext:
{decision_guidance}

{json.dumps(contexts, ensure_ascii=False, indent=2)}
""".strip()

    @staticmethod
    def _decision_guidance(decision_context) -> str:
        specific_guidance = {
            "clarify_weight_impact": (
                "Смысловая ось уже выбрана: уточни только влияние веса на "
                "повседневное самочувствие, энергию или жизнь."
            ),
            "clarify_problem_duration": (
                "Смысловая ось уже выбрана: уточни только длительность проблемы."
            ),
            "clarify_desired_change": (
                "Смысловая ось уже выбрана: уточни только желаемое изменение."
            ),
        }
        if decision_context.next_goal in specific_guidance:
            return specific_guidance[decision_context.next_goal]
        if decision_context.next_goal == "clarify_goals":
            return (
                "Получи один конкретный новый факт о желаемом изменении, "
                "длительности проблемы или её влиянии на повседневную жизнь. "
                "Не спрашивай повторно, что именно беспокоит в уже названной проблеме."
            )
        if decision_context.next_goal.startswith("clarify_"):
            area = decision_context.next_goal.removeprefix("clarify_")
            return f"Получи один конкретный новый факт, уточняющий область: {area}."
        if decision_context.next_goal == "verify_hypothesis":
            return "Получи один конкретный факт, который может подтвердить или опровергнуть гипотезу."
        return "Реализуй принятое решение без нового вопроса."

    def _validation_instruction(
        self,
        reply: str,
        current_message: str,
        decision_context,
        impact_context,
        emotional_context,
    ) -> str:
        constraints = {
            "DecisionContext": self._as_data(decision_context),
            "ImpactContext": self._as_data(impact_context),
            "EmotionalContext": self._as_data(emotional_context),
        }
        return f"""
ПРОВЕРЬ ОТВЕТ.

Проверь, соответствует ли ответ уже принятым ограничениям. Не принимай новое
решение и не переписывай ответ. Верни строго JSON без Markdown:
{{"is_valid": true или false, "reason": "краткая причина"}}.

Если DecisionContext требует вопрос, он должен получать новый конкретный факт,
нужный для next_goal. Отклони вопрос, который лишь повторяет или перефразирует
уже названную пользователем проблему.

Ограничения:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Ответ для проверки:
{reply}

Текущее сообщение пользователя:
{current_message}
""".strip()

    def _complete(self, messages: list, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _has_concise_single_action(reply: str, decision_context) -> bool:
        normalized = " ".join(reply.lower().split())
        word_count = len(re.findall(r"\w+", reply, flags=re.UNICODE))
        question_count = reply.count("?")

        if not normalized or word_count > MAX_REPLY_WORDS or question_count > 1:
            return False
        if normalized.startswith(FORMAL_OPENINGS):
            return False
        if decision_context.response_type == "question" and question_count != 1:
            return False
        if decision_context.response_type == "statement" and question_count:
            return False
        return True

    @staticmethod
    def _question_requests_new_information(
        reply: str, current_message: str, decision_context
    ) -> bool:
        if decision_context.response_type != "question":
            return True

        normalized_reply = " ".join(reply.lower().split())
        normalized_message = " ".join(current_message.lower().split())
        if any(pattern in normalized_reply for pattern in GENERIC_WEIGHT_RESTATEMENTS):
            return False
        if normalized_message and normalized_reply == normalized_message:
            return False
        markers = QUESTION_FOCUS_MARKERS.get(
            decision_context.next_goal,
            NEW_INFORMATION_MARKERS,
        )
        return any(marker in normalized_reply for marker in markers)

    def _is_valid(
        self,
        reply,
        current_message,
        decision_context,
        impact_context,
        emotional_context,
    ):
        if not self._has_concise_single_action(reply, decision_context):
            return False
        if not self._question_requests_new_information(
            reply, current_message, decision_context
        ):
            return False

        content = self._complete(
            [
                {"role": "system", "content": "Ты проверяешь соответствие ответа заданным ограничениям."},
                {
                    "role": "user",
                    "content": self._validation_instruction(
                        reply,
                        current_message,
                        decision_context,
                        impact_context,
                        emotional_context,
                    ),
                },
            ],
            temperature=0.0,
        )
        try:
            return bool(json.loads(self._clean_json_response(content))["is_valid"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    @staticmethod
    def _fallback_reply(decision_context) -> str:
        if decision_context.next_goal == "begin_consultation":
            return "Картина уже стала яснее. Давай посмотрим, что может помочь тебе дальше."
        if decision_context.next_goal == "clarify_weight_impact":
            return "Понимаю. Как вес сейчас влияет на твоё самочувствие или энергию?"
        if decision_context.next_goal == "clarify_problem_duration":
            return "Понимаю. Как давно эта ситуация тебя беспокоит?"
        if decision_context.next_goal == "clarify_desired_change":
            return "Понимаю. Что тебе хотелось бы изменить в первую очередь?"
        if decision_context.next_goal == "clarify_goals":
            return "Понимаю. Что тебе хотелось бы изменить в первую очередь?"
        if decision_context.next_goal.startswith("clarify_"):
            area = decision_context.next_goal.removeprefix("clarify_")
            return f"Понимаю. Что поможет уточнить для тебя тему {area}?"
        return "Понимаю. Что для тебя сейчас важнее всего?"

    def generate(
        self,
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        emotional_context,
        memory,
    ) -> str:
        """Generate and validate a reply without mutating supplied objects."""
        instruction = self._generation_instruction(
            semantic_context,
            human_model,
            reasoning_context,
            decision_context,
            impact_context,
            emotional_context,
            memory,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]

        for attempt in range(self.max_attempts):
            reply = self._complete(messages, temperature=0.8)
            if self._is_valid(
                reply,
                memory.get("current_message", ""),
                decision_context,
                impact_context,
                emotional_context,
            ):
                return reply
            if attempt + 1 < self.max_attempts:
                messages.extend(
                    [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": (
                                "Черновик не принят: вопрос не получил новый "
                                "конкретный факт для DecisionContext. Сформулируй "
                                "новый короткий ответ с одной реакцией и одним "
                                "вопросом строго в смысловой оси, уже выбранной "
                                "в DecisionContext."
                            ),
                        },
                    ]
                )

        return self._fallback_reply(decision_context)
