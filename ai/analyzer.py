from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def analyze_user_message(text: str):

    prompt = f"""
Ты анализируешь сообщения пользователя в боте про самочувствие и энергию.

ВАЖНО:

1. Короткие ответы типа:
"да", "есть", "давай", "ок", "ну", "ага"
— НЕ являются проблемой

2. Если пользователь отвечает на вопрос:
"насколько важно" → это readiness

3. Если пользователь говорит:
"есть усталость", "нет энергии" → has_energy_issue = true

4. Если:
"хочу похудеть", "лишний вес", "толстый" → has_weight_issue = true

5. Если не уверен → НЕ придумывай, ставь null

Сообщение:
{text}

Верни строго JSON:

{{
    "problem": строка или null,
    "has_energy_issue": true/false/null,
    "has_weight_issue": true/false/null,
    "readiness": число 0-10 или null
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