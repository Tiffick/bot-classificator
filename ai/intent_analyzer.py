from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def detect_intent(text: str, context: dict):

    last_question = context.get("last_question")
    state = context.get("state")

    prompt = f"""
Ты анализируешь сообщение пользователя в диалоге.

Контекст:
Состояние: {state}
Последний вопрос: {last_question}

Твоя задача — определить НАМЕРЕНИЕ (intent) пользователя.

Варианты intent:

- clear_problem → пользователь чётко описывает проблему
- vague → ответ размытый, не даёт ясности
- answer → отвечает на вопрос
- doubt → сомневается / не уверен
- question → задаёт вопрос
- confirmation → согласие двигаться дальше
- off_topic → уходит от темы
- short → короткий неинформативный ответ

Сообщение:
{text}

Верни строго JSON:

{{
  "intent": "<one_of_intents>"
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

    try:
        return json.loads(content)
    except Exception:
        return {"intent": "vague"}