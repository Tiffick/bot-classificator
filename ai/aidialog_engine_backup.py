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


# TEXT PROMPTS
SYSTEM_PROMPT = load_text("system_prompt.txt")
DECISION_LAYER = load_text("decision_layer.txt")
EMOTIONAL_ENGINE = load_text("emotional_engine.txt")
SEMANTIC_RULES = load_text("semantic_rules.txt")
IMPACT_ENGINE = load_text("impact_engine.txt")

# JSON LAYERS
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

---

# KNOWLEDGE BASE

{json.dumps(KNOWLEDGE_BASE, ensure_ascii=False, indent=2)}

---

# SEMANTIC MAP

{json.dumps(SEMANTIC_MAP, ensure_ascii=False, indent=2)}

---

# HUMAN EXPERIENCE LAYER

{json.dumps(HUMAN_EXPERIENCE, ensure_ascii=False, indent=2)}

---

# HUMAN TYPES

{json.dumps(HUMAN_TYPES, ensure_ascii=False, indent=2)}
"""


SYSTEM_CONTEXT = build_system_prompt()


def generate_reply(user_message: str, history: list = None) -> str:

    client = OpenAI()

    if history is None:
        history = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_CONTEXT
        }
    ]

    for item in history[-12:]:
        role = item.get("role")
        content = item.get("content")

        if role and content:
            messages.append({
                "role": role,
                "content": content
            })

    messages.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.9,
    )

    return response.choices[0].message.content.strip()


async def run_dialog_engine(user_text: str, profile: dict):
    history = profile.get("history", [])

    reply = generate_reply(
        user_message=user_text,
        history=history
    )

    history.append({
        "role": "user",
        "content": user_text
    })

    history.append({
        "role": "assistant",
        "content": reply
    })

    return {
        "reply": reply,
        "update": {
            "history": history[-20:]
        }
    }