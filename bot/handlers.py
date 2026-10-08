from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.filters import CommandStart
from aiogram import Router
from aiogram.utils.chat_action import ChatActionSender

from ai.dialog_engine import run_dialog_engine
from ai.telegram_consultant_handoff import CONTACT_BUTTON_TEXT

from memory.user_memory import (
    get_telegram_consultant_handoff,
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
TYPING_REFRESH_SECONDS = 4.0


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

    contact_phone = message.contact.phone_number if message.contact is not None else None
    user_text = message.text or ""
    user_id = message.from_user.id

    print("\n--- NEW MESSAGE ---")
    print("USER:", user_text)

    log_user(user_text if contact_phone is None else "[Контакт Telegram предоставлен]")

    profile = get_user_profile(user_id)

    async def consultant_sender(chat_id, text):
        await message.bot.send_message(chat_id=chat_id, text=text)

    engine_kwargs = {
        "contact_phone": contact_phone,
        "consultant_sender": consultant_sender,
    }
    if get_telegram_consultant_handoff(user_id) is None:
        async with ChatActionSender.typing(
            chat_id=message.chat.id,
            bot=message.bot,
            interval=TYPING_REFRESH_SECONDS,
        ):
            result = await run_dialog_engine(
                user_text, profile, user_id, **engine_kwargs
            )
    else:
        result = await run_dialog_engine(
            user_text, profile, user_id, **engine_kwargs
        )

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

    reply_markup = None
    if result.get("request_contact"):
        reply_markup = ReplyKeyboardMarkup(
            keyboard=[[
                KeyboardButton(
                    text=CONTACT_BUTTON_TEXT,
                    request_contact=True,
                )
            ]],
            resize_keyboard=True,
            one_time_keyboard=True,
        )
    elif result.get("remove_contact_keyboard"):
        reply_markup = ReplyKeyboardRemove()

    await message.answer(reply, reply_markup=reply_markup)
