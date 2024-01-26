import requests
from typing import Dict, Any, List
from prance import ResolvingParser
from tool_library.tools import FastApiRouteTool


class FastApiToolFactory:
    def __init__(self, service_url: str, tool_routes: List[str] = None, schema: Dict[str, Any] = None):
        self.service_url = service_url
        self.tools = []
        self.schema = schema
        self.tool_routes = tool_routes
        if not self.schema:
            self.introspect_service()

    def fetch_openapi_schema(self) -> None:
        try:
            # Use prance to load and resolve the OpenAPI specification
            parser = ResolvingParser(f"{self.service_url}/openapi.json")
            self.schema = parser.specification
        except Exception as e:
            raise ConnectionError(f"Error resolving OpenAPI schema: {e}")

    def introspect_service(self) -> None:
        if not self.schema:
            self.fetch_openapi_schema()

        for path, methods in self.schema.get("paths", {}).items():
            if self.tool_routes and path not in self.tool_routes:
                continue

            for method, details in methods.items():
                params = details
                self._create_tool_from_endpoint(path, method, details, params)

    def _create_tool_from_endpoint(
        self,
        path: str,
        method: str,
        details: Dict[str, Any],
        params: List[Dict[str, Any]],
    ) -> None:
        tool_name = f"{method.upper()} {path}"
        description = details.get("summary", "No description available")
        endpoint_url = f"{self.service_url}{path}"
        self.tools.append(
            FastApiRouteTool(tool_name, description, method, endpoint_url, params)
        )
