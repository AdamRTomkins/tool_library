import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel


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
USE_AUTH = not os.getenv("RAG_USE_AUTH", "True").lower() in ("false", "0", "f", "no")
MASTER_KEY = os.getenv("RAG_MASTER_KEY")

if USE_AUTH:
    assert MASTER_KEY is not None, "If you are using auth, you must set a master key using the 'RAG_MASTER_KEY' environment variable."
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


class TextItem(BaseModel):
    text: str
    attributes: Dict[str, Any] = {}

class FaissManager:
    def __init__(self, vector_dim: int, model: SentenceTransformer):
        self.vector_dim = vector_dim
        self.model = model
        self.index = faiss.IndexFlatL2(vector_dim)  # Using L2 distance for simplicity
        self.id_to_data = {}  # Mapping from FAISS index IDs to data
        self.next_id = 0  # To keep track of the next available ID

    def _text_to_embedding(self, text: str) -> np.ndarray:
        # Using SentenceTransformer model for embedding generation
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype("float32")

    def search(self, text: str) -> List[Dict[str, Any]]:
        query_vector = self._text_to_embedding(text)
        distances, indices = self.index.search(query_vector.reshape(1, -1), 10)  # Example: 10 nearest neighbors
        return [
            self.id_to_data[idx] for idx, distance in zip(indices[0], distances[0]) if idx in self.id_to_data
        ]

    def add_items(self, items: List[TextItem]):
        for item in items:
            embedding = self._text_to_embedding(item.text)
            self.index.add(embedding.reshape(1, -1))
            self.id_to_data[self.next_id] = {"text": item.text, "attributes": item.attributes, "id": self.next_id}
            self.next_id += 1

    def clear_index(self):
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.id_to_data.clear()
        self.next_id = 0

    def delete_items(self, criteria: Dict[str, Any]):
        # Example criteria: {"id": [list_of_ids_to_delete]}
        if "id" in criteria:
            for idx in criteria["id"]:
                if idx in self.id_to_data:
                    del self.id_to_data[idx]

        # Rebuild index without deleted items
        all_embeddings = []
        for idx, data in self.id_to_data.items():
            embedding = self._text_to_embedding(data["text"])
            all_embeddings.append(embedding)

        self.index.reset()
        if all_embeddings:
            self.index.add(np.array(all_embeddings))

    def update_items(self, updates: Dict[str, Any]):
        # Example update: {"id": id_to_update, "text": new_text}
        id_to_update = updates.get("id")
        new_text = updates.get("text")

        if id_to_update in self.id_to_data and new_text:
            self.id_to_data[id_to_update]["text"] = new_text
            new_embedding = self._text_to_embedding(new_text)
            self.index.reset()
            for idx, data in self.id_to_data.items():
                self.index.add(self._text_to_embedding(data["text"]).reshape(1, -1))

    def get_info(self):
        return {
            "number_of_items": len(self.id_to_data),
            "next_id": self.next_id
        }


model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_manager = FaissManager(vector_dim=384, model=model)

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


# Endpoint to search the index
@app.get("/search/", tags=["Search"])
async def search_index(query: str = "", api_key: str = Depends(verify_api_key)):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    results = faiss_manager.search(query)
    return results

# Endpoint to add items to the index
@app.post("/add/", tags=["Index Management"])
async def add_items(items: List[TextItem], api_key: str = Depends(verify_api_key)):
    faiss_manager.add_items(items)
    return {"detail": "Items added successfully"}

# Endpoint to clear and recreate the index
@app.post("/clear/", tags=["Index Management"])
async def clear_index(api_key: str = Depends(verify_api_key)):
    faiss_manager.clear_index()
    return {"detail": "Index cleared successfully"}

# Endpoint to delete specific elements from the index
@app.post("/delete/", tags=["Index Management"])
async def delete_items(criteria: Dict[str, Any], api_key: str = Depends(verify_api_key)):
    faiss_manager.delete_items(criteria)
    return {"detail": "Items deleted successfully"}

# Endpoint to update existing items in the index
@app.post("/update/", tags=["Index Management"])
async def update_items(updates: Dict[str, Any], api_key: str = Depends(verify_api_key)):
    faiss_manager.update_items(updates)
    return {"detail": "Items updated successfully"}

# Endpoint to get index information
@app.get("/info/", tags=["Index Information"])
async def index_info(api_key: str = Depends(verify_api_key)):
    info = faiss_manager.get_info()
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
