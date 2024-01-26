from pydantic import BaseModel
from typing import Dict, List, Any, Optional


# Pydantic models for request and response
class ToolRegistrationRequest(BaseModel):
    name: str
    description: str
    function: str  # Assuming the function is a string representing the function name


class APIToolRegistrationRequest(BaseModel):
    tool_url: str
    tool_routes: Optional[List[str]] = None
    api_key: Optional[str] = None # Currently Unsed


class ToolExecutionRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any] = {}


class ToolStatsResponse(BaseModel):
    creation_time: str
    call_count: int
    average_execution_time: float


class ToolSearchRequest(BaseModel):
    query: str


class EventsRequest(BaseModel):
    minutes_ago: Optional[int] = 0
