import asyncio
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from bot.handlers import router

load_dotenv(".env")

TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()

dp.include_router(router)

async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())