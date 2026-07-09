import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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


def clean_json_response(content: str) -> str:

    content = content.strip()

    if content.startswith("```"):
        parts = content.split("```")

        if len(parts) > 1:
            content = parts[1]

        if content.startswith("json"):
            content = content[4:]

    return content.strip()


def calculate_discovery_complete(profile: dict) -> bool:

    required_fields = [
        "age",
        "current_weight",
        "target_weight",
        "duration",
        "main_problem",
        "previous_attempts",
        "failure_reason"
    ]

    for field in required_fields:
        if not profile.get(field):
            return False

    return True


def build_known_unknown(profile: dict):

    slots = [
        "age",
        "current_weight",
        "target_weight",
        "duration",
        "main_problem",
        "previous_attempts",
        "failure_reason"
    ]

    known = []
    unknown = []

    for slot in slots:

        value = profile.get(slot)

        if value:
            known.append(f"{slot}: {value}")
        else:
            unknown.append(slot)

    return known, unknown


async def run_dialog_engine(user_text: str, profile: dict):

    client = OpenAI()

    history = profile.get("history", [])

    known, unknown = build_known_unknown(profile)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_CONTEXT
        }
    ]

    for item in history[-12:]:
        messages.append(item)

    extraction_prompt = f"""
ТЕКУЩИЙ ПРОФИЛЬ:

{json.dumps(profile, ensure_ascii=False, indent=2)}

---

УЖЕ ИЗВЕСТНО:

{chr(10).join(known) if known else "ничего"}

---

НЕ ХВАТАЕТ:

{chr(10).join(unknown) if unknown else "ничего"}

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

    safe_update = {}

    for key, value in update.items():

        if value is None:
            continue

        if profile.get(key) is None:
            safe_update[key] = value

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

    temp_profile = profile.copy()
    temp_profile.update(safe_update)

    safe_update["history"] = history[-20:]
    safe_update["discovery_complete"] = calculate_discovery_complete(
        temp_profile
    )

    return {
        "reply": reply,
        "update": safe_update
    }