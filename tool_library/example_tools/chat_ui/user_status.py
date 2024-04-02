import os
import time

import requests
from pydantic import BaseModel

USER_PING_API_KEY = os.getenv("USER_PING_API_KEY")

if not USER_PING_API_KEY:
    raise ValueError("USER_PING_API_KEY environment variable is not set")

class UserStatus(BaseModel):
    username: str
    registered: bool
    currently_sharing: bool
    #total_share_time: int = 0
    last_updated: float


def get_user_status(username):
    # This is a placeholder function to get the user status from the API

    url = "https://platform.backend.test.k8s.mvp.kalavai.net/get_user_pings"

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': USER_PING_API_KEY
    }

    payload = {
        "username": username,
        "delta_minutes": 0,
    }   
    
    all_response = requests.post(url, json=payload, headers=headers)
    all_response.raise_for_status()

    registered = all_response.json().get("pings", False) > 0 

    payload["delta_minutes"] = 5
    recent_response = requests.post(url, json=payload, headers=headers)
    currently_sharing = recent_response.json().get("pings", False) > 0

    return UserStatus(
        username=username,
        registered=os.environ.get("REGISTERED", str(registered)).lower() in ["true", "1"],
        currently_sharing=os.environ.get("CURRENTLY_SHARING", str(currently_sharing)).lower() in ["true", "1"],
        #total_share_time=os.environ.get("TOTAL_SHARE_TIME", 100),
        last_updated=time.time(),
    )