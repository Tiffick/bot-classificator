import json

from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram import Router

from ai.analyzer import analyze_user_message
from logic.dialog_manager import generate_reply
from memory.user_memory import get_user_profile, update_user_profile


router = Router()


@router.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! 👋\n\n"
        "Я помогу разобрать ваше самочувствие и уровень энергии.\n\n"
        "Расскажите, что вас больше всего беспокоит?"
    )


@router.message()
async def message_handler(message: Message):

    user_text = message.text
    user_id = message.from_user.id

    print("\n--- NEW MESSAGE ---")
    print("USER:", user_text)

    await message.answer("Анализирую ваше сообщение...")

    analysis = await analyze_user_message(user_text)

    print("RAW AI RESPONSE:", analysis)

    try:
        analysis_data = json.loads(analysis)
    except Exception:
        await message.answer("Ошибка анализа. Попробуйте еще раз.")
        print("JSON ERROR:", analysis)
        return

    print("PARSED AI:", analysis_data)

    profile = update_user_profile(user_id, analysis_data)

    print("UPDATED PROFILE:", profile)

    reply = generate_reply(profile)

    print("BOT REPLY:", reply)

    await message.answer(reply)