def generate_reply(profile: dict):

    has_weight = profile.get("has_weight_issue")
    has_energy = profile.get("has_energy_issue")
    readiness = profile.get("readiness")

    # 1. Нет проблемы вообще
    if not has_weight and not has_energy:
        return "Расскажите, что именно вас беспокоит?"

    # 2. Есть вес, но нет readiness
    if has_weight and readiness is None:
        return "Понял. А насколько для вас это сейчас важно по шкале от 0 до 10?"

    # 3. Есть вес + есть readiness → двигаемся дальше
    if has_weight and readiness is not None:

        # если энергии еще нет → спросить
        if has_energy is None:
            return "Понял. А как у вас с уровнем энергии? Есть усталость?"

        # если есть энергия → следующий шаг
        if has_energy:
            return "Похоже, причина может быть связана с питанием. Хотите, я подскажу, с чего начать?"

        # если энергии нет → тоже идем дальше
        if has_energy is False:
            return "Хорошо. Тогда давайте посмотрим, что можно скорректировать в питании."

    # fallback (почти не должен срабатывать)
    return "Расскажите чуть подробнее о вашей ситуации."