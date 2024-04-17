import os

import anvil.server
import anvil.users

anvil_key = os.getenv("ANVIL_UPLINK_KEY", "")
anvil.server.connect(anvil_key)


def auth_user(username: str, password: str):
    try:
        user = anvil.users.login_with_email(username, password)
        return user
    except Exception as e:
        print(str(e))
    return None
