user_profiles = {}


def get_user_profile(user_id):
    return user_profiles.get(user_id, {})


def update_user_profile(user_id, new_data):

    profile = user_profiles.get(user_id, {})

    for key, value in new_data.items():

        # НЕ перезаписываем problem пустыми или мусорными значениями
        if key == "problem":
            if value and len(value) > 5:
                profile[key] = value
            continue

        if value is True:
            profile[key] = True

        elif value is False:
            if key not in profile:
                profile[key] = False

        elif value is not None:
            profile[key] = value

    user_profiles[user_id] = profile

    return profile