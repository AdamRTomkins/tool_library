import requests
from typing import List, Dict, Any

class FAISSClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"X-API-KEY": self.api_key}

    def search(self, query: str) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_url}/search/", params={"query": query}, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def add_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/add/", json=items, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def clear_index(self) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/clear/", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delete_items(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/delete/", json=criteria, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def update_items(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/update/", json=updates, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def index_info(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/info/", headers=self.headers)
        response.raise_for_status()
        return response.json()

## Integrationsfrom langchain_core.retrievers import BaseRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from your_faiss_client_module import FAISSClient  # Ensure this is correctly imported

class KalavaiRetriever(BaseRetriever):
    
    def __init__(self, base_url: str, api_key: str):
        self.faiss_client = FAISSClient(base_url, api_key)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        search_results = self.faiss_client.search(query)
        documents = [Document(page_content=result['text'], metadata=result['attributes']) for result in search_results]
        return documents

###