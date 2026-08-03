from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram import Router

from ai.dialog_engine import run_dialog_engine

from memory.user_memory import (
    get_user_profile,
    update_user_profile,
    set_last_question,
    reset_user_profile
)

from utils.logger import (
    clear_log,
    log_user,
    log_bot
)

router = Router()


@router.message(CommandStart())
async def start_handler(message: Message):

    user_id = message.from_user.id

    reset_user_profile(user_id)

    clear_log()

    text = (
        "Привет 🙂\n\n"
        "Если ты здесь, значит что-то уже начало напрягать - "
        "вес, энергия или просто самочувствие в целом."
    )

    set_last_question(user_id, text)

    log_bot(text)

    await message.answer(text)


@router.message()
async def message_handler(message: Message):

    user_text = message.text
    user_id = message.from_user.id

    print("\n--- NEW MESSAGE ---")
    print("USER:", user_text)

    log_user(user_text)

    profile = get_user_profile(user_id)

    result = await run_dialog_engine(user_text, profile, user_id)

    print("ENGINE RESULT:", result)

    update_data = result.get("update", {})
    reply = result.get(
        "reply",
        "Можешь чуть подробнее рассказать?"
    )

    profile = update_user_profile(
        user_id,
        update_data
    )

    set_last_question(
        user_id,
        reply
    )

    log_bot(reply)

    print("UPDATED PROFILE:", profile)
    print("BOT REPLY:", reply)

    await message.answer(reply)
