from pathlib import Path

LOG_FILE = Path("conversation_log.txt")


def clear_log():

    LOG_FILE.write_text("", encoding="utf-8")


def log_user(text: str):

    with open(LOG_FILE, "a", encoding="utf-8") as f:

        f.write(
            "\n"
            "==========================\n"
            "USER\n"
            "==========================\n\n"
        )

        f.write(text)
        f.write("\n")


def log_bot(text: str):

    with open(LOG_FILE, "a", encoding="utf-8") as f:

        f.write(
            "\n"
            "==========================\n"
            "BOT\n"
            "==========================\n\n"
        )

        f.write(text)
        f.write("\n")