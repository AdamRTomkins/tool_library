import os

import anvil.users
import anvil.server


anvil.server.connect(os.getenv("ANVIL_UPLINK_KEY", ""))


def auth_user(username: str, password: str):
    try:
        user = anvil.users.login_with_email(
            username,
            password)
        return user
    except Exception as e:
        print(str(e))
    return None
