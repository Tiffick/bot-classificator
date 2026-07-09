from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def analyze_user_message(text: str, context: dict):

    last_question = context.get("last_question")
    state = context.get("state")

    prompt = f"""
Ты анализируешь сообщения пользователя в диалоге про самочувствие.

Контекст:

Состояние диалога: {state}
Последний вопрос бота: {last_question}

ВАЖНО:

1. Короткие ответы:
"да", "есть", "ага", "ок"

— могут быть ответом на вопрос

2. Если вопрос был про энергию:
"есть", "да", "ага" → has_energy_issue = true
"нет" → has_energy_issue = false

3. Если вопрос был про важность:
"да", "очень", "конечно" → readiness = 7-10

4. Если пользователь явно пишет число → это readiness

5. Если:
"лишний вес", "толстый", "хочу похудеть" → has_weight_issue = true

6. Если не уверен → null

НЕ ВЫДУМЫВАЙ.

Сообщение:
{text}

Верни строго JSON:

{{
    "problem": string or null,
    "has_energy_issue": true/false/null,
    "has_weight_issue": true/false/null,
    "readiness": number or null
}}
"""

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Ты возвращаешь только JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    return content.strip()