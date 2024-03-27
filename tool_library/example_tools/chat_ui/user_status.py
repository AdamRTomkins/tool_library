import os
import time
from pydantic import BaseModel

class UserStatus(BaseModel):
    username: str
    registered: bool
    currently_sharing: bool
    total_share_time: int
    last_updated: float


def get_user_status(username):
    # This is a placeholder function to get the user status from the API

    return UserStatus(
        username=username,
        registered=os.environ.get("REGISTERED", "True").lower() in ["true", "1"],
        currently_sharing=os.environ.get("CURRENTLY_SHARING", "True").lower() in ["True", "1"],
        total_share_time=os.environ.get("TOTAL_SHARE_TIME", 100),
        last_updated=time.time(),
    )