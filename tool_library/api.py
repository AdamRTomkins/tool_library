from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
from tool_library.library import ToolLibrary
from tool_library.models import *
from fastapi import FastAPI, HTTPException, Depends
from starlette.requests import Request
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


USE_AUTH = os.environ.get("TL_USE_AUTH", True)
if USE_AUTH == "False" or USE_AUTH == False:
    USE_AUTH = False
else:
    USE_AUTH = True

logger.warning(f"USE_AUTH { USE_AUTH}")

MASTER_KEY = os.environ.get("TL_MASTER_KEY", "adam")
VALID_API_KEYS = {MASTER_KEY}


def validate_api_key(request: Request):
    if not USE_AUTH:
        return None

    api_key = request.headers.get("X-API-KEY")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


app = FastAPI()
tool_library = ToolLibrary()
# tool_library.register_api_tool()


@app.post("/register-api-tool/")
async def register_api_tool(
    request: APIToolRegistrationRequest, api_key: str = Depends(validate_api_key)
):
    # This will need to be adapted based on how you want to handle the dynamic function registration
    logger.info(request)
    try:
        tool_library.register_api_tool(request.service_url, request.add_routes)
        return {"message": "Tool registered successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint to execute a tool
@app.post("/execute-tool/")
async def execute_tool(
    request: ToolExecutionRequest, api_key: str = Depends(validate_api_key)
):
    logger.info(request)
    try:
        logger.info(request)
        result = tool_library.execute_tool(request.tool_name, request.params)
        logger.info(result)
        ret = {"result": result}
        return ret
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Endpoint to get tool statistics
@app.get("/tool-stats/{tool_name}", response_model=ToolStatsResponse)
async def get_tool_stats(tool_name: str, api_key: str = Depends(validate_api_key)):
    logger.info(request)
    try:
        stats = tool_library.get_tool_stats(tool_name)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Endpoint to find tools
@app.post("/find-tools/")
async def find_tools(
    request: ToolSearchRequest, api_key: str = Depends(validate_api_key)
):
    logger.info(request)
    tools = tool_library.find_tools(request.query)
    return {"tools": tools}


# Endpoint to find tools
@app.post("/get_new_events/")
async def get_events(request: EventsRequest, api_key: str = Depends(validate_api_key)):
    logger.info(request)
    events = tool_library.get_events(request.minutes_ago)
    return {"events": events}


# Endpoint to get all tools
@app.get("/get-tools/")
async def get_tools(api_key: str = Depends(validate_api_key)):
    logger.info("Get Tools Called")
    tools = tool_library.get_tools()
    return {"tools": [t.to_json() for t in tools.values()]}


# Endpoint to remove a tool
@app.delete("/remove-tool/{tool_name}")
async def remove_tool(tool_name: str, api_key: str = Depends(validate_api_key)):
    logger.info(f"Remove Tool called {tool_name}")
    success = tool_library.remove_tool(tool_name)
    if success:
        return {"message": "Tool removed successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")


# Endpoint to get all tools
@app.get("/health/")
async def health():
    return HTTPException(status_code=200, detail="OK")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.1", port=8000)
