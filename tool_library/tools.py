import requests
import logging
import time

from typing import Callable, get_type_hints, Dict, List, Any, Type
from pydantic import BaseModel
from prance import ResolvingParser
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolStats:
    def __init__(self):
        self.creation_time = datetime.now()
        self.call_count = 0
        self.total_execution_time = 0

    def update_stats(self, execution_time):
        self.call_count += 1
        self.total_execution_time += execution_time

    @property
    def average_execution_time(self):
        return self.total_execution_time / self.call_count if self.call_count > 0 else 0


class Tool:
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
        self.params = self._introspect_params(function)
        self.stats = ToolStats()

    def to_json(self):
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
        }

    def get_type_hints(self, func):
        return get_type_hints(func)

    def _introspect_params(self, func: Callable) -> List[Dict[str, Any]]:
        type_hints = self.get_type_hints(func)
        parameters = []

        for param_name, param_type in type_hints.items():
            if issubclass(param_type, BaseModel):
                model_schema = param_type.schema()
                for field_name, field_info in model_schema["properties"].items():
                    param_info = {
                        "name": field_name,
                        "in": "body",
                        "required": field_name in model_schema.get("required", []),
                        "schema": field_info,
                    }
                    parameters.append(param_info)
            else:
                # Handling simple types (extend this as per your needs)
                param_info = {
                    "name": param_name,
                    "in": "query",  # Assuming query parameters for simple types
                    "required": True,  # Assuming all simple type parameters are required
                    "schema": {"type": str(param_type)},
                }
                parameters.append(param_info)

    def execute(self, params):
        start_time = time.time()
        result = self._execute(params)
        execution_time = time.time() - start_time
        self.stats.update_stats(execution_time)
        return result

    def _execute(self, params):
        return self.function(**params)


class FastApiRouteTool(Tool):
    def __init__(self, name, description, method, endpoint_url, params, api_key=None):
        self.name = name
        self.description = description
        self.method = method.upper()
        self.endpoint_url = endpoint_url
        self.params = params
        self.stats = ToolStats()

        self.api_key = api_key  
        self.headers = {}
        if api_key is not None:
            self.headers = {'X-API-KEY': api_key}

    def _execute(self, payload: Dict) -> Any:
        
        try:
            if self.method == "GET":
                response = requests.get(
                    self.endpoint_url, params=payload, headers=self.headers
                )
            elif self.method == "POST":
                response = requests.post(
                    self.endpoint_url, json=payload, headers=self.headers
                )
            # Add other HTTP methods as necessary
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "response": e.response.text}


try:
    import ray

    class RayTool(Tool):
        def __init__(self, name: str, description: str, function: Callable):
            # Initialize the base Tool class
            super().__init__(name=name, description=description, function=function)

        @classmethod
        def is_ray_remote_function(func):
            return hasattr(func, "remote") and callable(getattr(func, "remote"))

        def get_type_hints(self, func):
            if hasattr(func, "_function"):
                # Access the original function
                func = func._function
                return get_type_hints(func)

        def _execute(self, params: Dict) -> Any:
            # Submit a Ray task
            future = self.function.remote(params)

            # Retrieve and return the result
            return ray.get(future)

except:
    logger.warning("Cannot import ray, to use ray functions, install tool-library[ray]")
