from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.filters import CommandStart
from aiogram import Router
from aiogram.utils.chat_action import ChatActionSender

from ai.dialog_engine import run_dialog_engine
from ai.cognitive_diagnostics import (
    conversation_identifier,
    record_telegram_delivery,
    start_diagnostic_session,
)
from ai.telegram_consultant_handoff import CONTACT_BUTTON_TEXT

from memory.user_memory import (
    get_telegram_consultant_handoff,
    get_user_profile,
    update_user_profile,
    set_last_question,
    reset_user_profile
)

router = Router()
TYPING_REFRESH_SECONDS = 4.0


def _telegram_diagnostic_metadata(message, sent_message=None):
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    return {
        "incoming_message_id": getattr(message, "message_id", None),
        "outgoing_message_id": getattr(sent_message, "message_id", None),
        "chat_identifier": (
            conversation_identifier(chat_id) if chat_id is not None else None
        ),
    }


@router.message(CommandStart())
async def start_handler(message: Message):

    user_id = message.from_user.id

    reset_user_profile(user_id)

    text = (
        "Привет 🙂\n\n"
        "Если ты здесь, значит что-то уже начало напрягать - "
        "вес, энергия или просто самочувствие в целом."
    )

    set_last_question(user_id, text)

    sent_message = await message.answer(text)
    start_diagnostic_session(
        user_id,
        start_exchange={
            "user_command": "/start",
            "assistant_reply": text,
            "telegram_delivery": "delivered",
            **_telegram_diagnostic_metadata(message, sent_message),
        },
    )


@router.message()
async def message_handler(message: Message):

    contact_phone = message.contact.phone_number if message.contact is not None else None
    user_text = message.text or ""
    user_id = message.from_user.id

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

    try:
        sent_message = await message.answer(reply, reply_markup=reply_markup)
    except Exception as error:
        record_telegram_delivery(
            user_id,
            result.get("_diagnostic_turn_id"),
            reply=reply,
            delivered=False,
            error=f"{type(error).__name__}: {error}",
            telegram_metadata=_telegram_diagnostic_metadata(message),
        )
        raise
    record_telegram_delivery(
        user_id,
        result.get("_diagnostic_turn_id"),
        reply=reply,
        delivered=True,
        telegram_metadata=_telegram_diagnostic_metadata(message, sent_message),
    )
