import os
import time

import requests
from pydantic import BaseModel

PLATFORM_BACKEND_API_KEY = os.getenv("PLATFORM_BACKEND_API_KEY")
PLATFORM_BACKEND_ENDPOINT = os.getenv("PLATFORM_BACKEND_ENDPOINT", "https://platform.backend.test.k8s.mvp.kalavai.net")

if not PLATFORM_BACKEND_API_KEY or not PLATFORM_BACKEND_ENDPOINT:
    raise ValueError("PLATFORM_BACKEND_API_KEY or PLATFORM_BACKEND_ENDPOINT environment variable is not set")

class UserStatus(BaseModel):
    username: str
    registered: bool
    currently_sharing: bool
    #total_share_time: int = 0
    last_updated: float


def get_user_status(username):
    # This is a placeholder function to get the user status from the API

    url = f"{PLATFORM_BACKEND_ENDPOINT}/get_user_pings"

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': PLATFORM_BACKEND_API_KEY
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