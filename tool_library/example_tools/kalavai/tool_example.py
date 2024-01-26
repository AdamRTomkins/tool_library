from fastapi import FastAPI, HTTPException, Depends
from starlette.requests import Request

import os
import logging
from typing import Dict, List, Any

# Configure logging at the application's entry point
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration handling
# Conservatively assume Auth is enabled if anything other than these false values are set.
USE_AUTH = not os.getenv("TOOL_USE_AUTH", "True").lower() in ("false", "0", "f", "no")
MASTER_KEY = os.getenv("TOOL_MASTER_KEY")

if USE_AUTH:
    assert MASTER_KEY is not None, "If you are using auth, you must set a master key using the 'TL_MASTER_KEY' environment variable."
else:
    logger.warning("Warning: Authentication is disabled. This should only be used for testing.")

# API Key Management (Consider using a more secure approach for production)
VALID_API_KEYS = {MASTER_KEY}

IN_CLUSTER = os.getenv("IN_CLUSTER", "False").lower() in ("true", "1", "t", "yes")
logger.warning(f"IN_CLUSTER: {IN_CLUSTER}")

tags_metadata = [
    {
        "name": "Example Section",
        "description": "Example Tools we could give to users.",
    },
]

# FastAPI instance
app = FastAPI(openapi_tags=tags_metadata)

# API Key Validation
async def verify_api_key(request: Request):
    if not USE_AUTH:
        return
    api_key = request.headers.get("X-API-KEY")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# Endpoint to check health
@app.get("/health/")
async def health():
    return HTTPException(status_code=200, detail="OK")

# Routes
#1. Delete a MT for a User (KW)
@app.get("/who_is_kalavai", tags=["Example Section"])
async def who_is_kalavai():
    return "Kalavai is a distributed compute platform, democratizing AI, founded by Carlos, Annie, Adam and Srevin (CaaS), which stands for Carlos as a Service, because it mostly runs on Carlos' computer Today."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
