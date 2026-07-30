import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ai.engines.human_model_engine import HumanModelEngine

load_dotenv()

PROMPTS_DIR = Path("ai/prompts")


def load_text(filename: str) -> str:
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def load_json(filename: str):
    path = PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SYSTEM_PROMPT = load_text("system_prompt.txt")
DECISION_LAYER = load_text("decision_layer.txt")
EMOTIONAL_ENGINE = load_text("emotional_engine.txt")
SEMANTIC_RULES = load_text("semantic_rules.txt")
IMPACT_ENGINE = load_text("impact_engine.txt")

KNOWLEDGE_BASE = load_json("knowledge_base.json")
SEMANTIC_MAP = load_json("semantic_map.json")
HUMAN_EXPERIENCE = load_json("human_experience_layer.json")
HUMAN_TYPES = load_json("human_types.json")


def build_system_prompt() -> str:
    return f"""
{SYSTEM_PROMPT}

---

# DECISION LAYER

{DECISION_LAYER}

---

# EMOTIONAL ENGINE

{EMOTIONAL_ENGINE}

---

# IMPACT ENGINE

{IMPACT_ENGINE}

---

# SEMANTIC RULES

{SEMANTIC_RULES}
"""


SYSTEM_CONTEXT = build_system_prompt()


def build_messages(history: list) -> list:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_CONTEXT
        }
    ]

    messages.extend(history[-12:])

    return messages


def append_history(history: list, user_text: str, reply: str) -> None:

    history.append(
        {
            "role": "user",
            "content": user_text
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": reply
        }
    )


def clean_json_response(content: str) -> str:

    content = content.strip()

    if content.startswith("```"):
        parts = content.split("```")

        if len(parts) > 1:
            content = parts[1]

        if content.startswith("json"):
            content = content[4:]

    return content.strip()


async def run_dialog_engine(user_text: str, profile: dict):

    client = OpenAI()

    history = profile.get("history", [])

    engine = HumanModelEngine()

    human_model = engine.build(profile)

    # TODO V2:
    # Дальнейшие Engine будут работать через HumanModel,
    # а не напрямую с profile.

    messages = build_messages(history)

    extraction_prompt = f"""
ТЕКУЩИЙ ПРОФИЛЬ:

{json.dumps(profile, ensure_ascii=False, indent=2)}

---

УЖЕ ИЗВЕСТНО:

{chr(10).join(human_model.known) if human_model.known else "ничего"}

---

НЕ ХВАТАЕТ:

{chr(10).join(human_model.unknown) if human_model.unknown else "ничего"}

---

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:

{user_text}

---

Извлеки новую информацию, если она появилась.

Никогда не придумывай значения.

Если данных нет — не заполняй поле.

Верни строго JSON:

{{
    "update": {{
        "age": null,
        "current_weight": null,
        "target_weight": null,
        "duration": null,
        "main_problem": null,
        "previous_attempts": null,
        "failure_reason": null
    }},
    "reply": "живой ответ пользователю"
}}
"""

    messages.append(
        {
            "role": "user",
            "content": extraction_prompt
        }
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.8
    )

    content = response.choices[0].message.content

    print("\nRAW GPT RESPONSE:")
    print(content)

    content = clean_json_response(content)

    try:
        parsed = json.loads(content)

    except Exception:

        parsed = {
            "update": {},
            "reply": "Можешь чуть подробнее рассказать?"
        }

    update = parsed.get("update", {})
    reply = parsed.get("reply", "Можешь чуть подробнее рассказать?")

    safe_update = engine.apply_update(profile, update)

    append_history(history, user_text, reply)

    temp_profile = profile.copy()
    temp_profile.update(safe_update)

    safe_update["history"] = history[-20:]
    safe_update["discovery_complete"] = (
        engine.is_discovery_complete(temp_profile)
    )

    return {
        "reply": reply,
        "update": safe_update
    }