def normalize(text: str) -> str:
    return text.strip().lower()


def is_empty_message(text: str) -> bool:

    if not text:
        return True

    text = normalize(text)

    garbage = [
        "?", "??", "???",
        "ну", "нуу", "ну?",
        "мм", "эм",
        "..."
    ]

    return text in garbage


def is_continue_signal(text: str) -> bool:
    text = normalize(text)

    return text in [
        "ок",
        "ага"
    ]


def is_confirmation(text: str) -> bool:
    text = normalize(text)

    # ❗ УБРАЛИ "да"
    return text in [
        "давай",
        "хочу",
        "разбери",
        "ок давай",
        "давай попробуем"
    ]
