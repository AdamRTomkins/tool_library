import requests
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.getLevelName(os.environ.get("TOOL_LIBRARY_LOG_LEVEL", "INFO")))

class ToolLibraryClient:
    def __init__(self, service_url: str, api_key: str=None):

        logger.debug(f"Initializing Tool Library for service url {service_url}")
        logger.debug(f"Using API Key: {api_key is None}")
        self.service_url = service_url
        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key}

        # Check health on initialization          
          
        assert self.health()["status_code"] == 200

    def register_api_tool(self, tool_url, tool_routes=None, tool_api_key=None):


        if tool_api_key is not None:
            logger.warning("Tool APIS not yet supported")

        # TODO: Add API KEY to Headers

        data = {"tool_url": tool_url, "tool_routes": tool_routes, "api_key": tool_api_key}
        response = requests.post(f"{self.service_url}/register-api-tool/", headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def execute_tool(self, tool_name, params):
        data = {"tool_name": tool_name, "params": params}
        response = requests.post(f"{self.service_url}/execute-tool/", headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def get_tool_stats(self, tool_name):
        response = requests.get(f"{self.service_url}/tool-stats/{tool_name}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def find_tools(self, query):
        data = {"query": query}
        response = requests.post(f"{self.service_url}/find-tools/", headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def get_events(self, minutes_ago=0):
        data = {"minutes_ago": minutes_ago}
        response = requests.post(f"{self.service_url}/get_new_events/", headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def get_tools(self):
        response = requests.get(f"{self.service_url}/get-tools/", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def remove_tool(self, tool_name):
        response = requests.delete(f"{self.service_url}/remove-tool/{tool_name}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def health(self):
        logger.debug(f"Checking Health on {self.service_url}/health/ from the Client")
        response = requests.get(f"{self.service_url}/health/", headers=self.headers)
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":

    # Example Usage:
    client = ToolLibraryClient("http://localhost:8000", "your_api_key")
    print(client.get_tools())
