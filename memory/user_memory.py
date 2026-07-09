user_profiles = {}


def get_user_profile(user_id):

    return user_profiles.get(user_id, {
        "last_question": None,

        # BASIC INFO
        "name": None,
        "gender": None,
        "age": None,
        "height": None,
        "current_weight": None,
        "target_weight": None,

        # DISCOVERY
        "duration": None,
        "main_problem": None,
        "previous_attempts": None,
        "failure_reason": None,

        # PAIN MAP
        "appearance": False,
        "clothes": False,
        "breathing": False,
        "energy": False,
        "mobility": False,
        "sleep": False,
        "health": False,

        # STATE
        "discovery_complete": False,

        # CHAT HISTORY
        "history": []
    })


def update_user_profile(user_id, new_data):

    profile = get_user_profile(user_id)

    for key, value in new_data.items():

        if value is None:
            continue

        profile[key] = value

    user_profiles[user_id] = profile

    return profile


def set_last_question(user_id, question):

    profile = get_user_profile(user_id)

    profile["last_question"] = question

    user_profiles[user_id] = profile


def reset_user_profile(user_id):

    user_profiles[user_id] = {
        "last_question": None,

        # BASIC INFO
        "name": None,
        "gender": None,
        "age": None,
        "height": None,
        "current_weight": None,
        "target_weight": None,

        # DISCOVERY
        "duration": None,
        "main_problem": None,
        "previous_attempts": None,
        "failure_reason": None,

        # PAIN MAP
        "appearance": False,
        "clothes": False,
        "breathing": False,
        "energy": False,
        "mobility": False,
        "sleep": False,
        "health": False,

        # STATE
        "discovery_complete": False,

        # CHAT HISTORY
        "history": []
    }