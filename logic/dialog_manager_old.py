from memory.user_memory import set_last_question, set_concern


def generate_reply(user_id: int, profile: dict, intent: str):

    problem = profile.get("problem")
    readiness = profile.get("readiness")
    has_energy = profile.get("has_energy_issue")
    is_concerned = profile.get("is_concerned")
    last_q_type = profile.get("last_question_type")

    # 🔥 1. фиксируем ответ на конкретный вопрос
    if intent == "answer":

        if last_q_type == "concern":
            set_concern(user_id, True)
            is_concerned = True

        if last_q_type == "readiness" and readiness is None:
            # пока просто считаем, что ответ был → двигаемся дальше
            readiness = 1

    # 🔥 2. вопрос пользователя
    if intent == "question":
        return (
            "Поясню чуть позже, это важно. "
            "Сначала хочу понять твою ситуацию, чтобы не говорить в общем.\n\n"
            "Расскажи, что тебя сейчас больше всего беспокоит?"
        )

    # 🔥 3. нет проблемы
    if not problem:
        q = "Что тебя сейчас больше всего беспокоит?"
        set_last_question(user_id, q, "problem")
        return q

    # 🔥 4. спросить про напряжение
    if is_concerned is None:
        q = "Скажи, а это уже начинает тебя напрягать или пока просто наблюдение?"
        set_last_question(user_id, q, "concern")
        return q

    # 🔥 5. readiness
    if readiness is None:
        q = "Скажи, это уже стало чем-то, что хочется изменить, или пока просто замечаешь?"
        set_last_question(user_id, q, "readiness")
        return q

    # 🔥 6. энергия
    if has_energy is None:
        q = "А как у тебя сейчас с энергией — есть усталость?"
        set_last_question(user_id, q, "energy")
        return q

    # 🔥 7. дальше
    return "Давай разберём это глубже. Ты уже пробовал что-то менять?"