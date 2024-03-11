from typing import List, Dict, Any
from fastapi import UploadFile, File

from fastapi import FastAPI, HTTPException, Depends
from starlette.requests import Request

import time
import os
import logging

from models import EmbeddingConfig
from utils import (
    save_upload_file,
    index_documents,
    load_document_retriever,
    process_retriever_results
)

from langchain_community.document_loaders import PyPDFium2Loader


# Configure logging at the application's entry point
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration handling
# Conservatively assume Auth is enabled if anything other than these false values are set.
USE_AUTH = not os.getenv("KB_USE_AUTH", "True").lower() in ("false", "0", "f", "no")
MASTER_KEY = os.getenv("KB_MASTER_KEY")
DB_BASE_FOLDER = os.getenv("DB_BASE_FOLDER", "./data")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8080") #"https://embedding.test.k8s.mvp.kalavai.net")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "None")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small") #"BAAI/bge-large-en-v1.5")


if USE_AUTH:
    assert MASTER_KEY is not None, "If you are using auth, you must set a master key using the 'RAG_MASTER_KEY' environment variable."
else:
    logger.warning("Warning: Authentication is disabled. This should only be used for testing.")

# API Key Management (Consider using a more secure approach for production)
VALID_API_KEYS = {MASTER_KEY}

tags_metadata = [
    {
        "name": "Example Section",
        "description": "Example Tools we could give to users.",
    },
]

# FastAPI instance
app = FastAPI(openapi_tags=tags_metadata)

embedder = EmbeddingConfig(
    api_url=EMBEDDING_API_URL,
    api_key=EMBEDDING_API_KEY,
    embedding_id=EMBEDDING_MODEL_ID
)


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
async def search_index(query: str, index_name: str, api_key: str = Depends(verify_api_key)):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    if not index_name:
        raise HTTPException(status_code=400, detail="Index name is required")
    retriever = load_document_retriever(
        index_name=index_name,
        embedder=embedder,
        top_k=10,
        similarity_threshold=0.5,
        base_path=DB_BASE_FOLDER)
    results = retriever.invoke(query)

    return process_retriever_results(results)


# Endpoint to add pdf files to the index
@app.post("/add")
async def upload_file(file: UploadFile = File(...), username: str = None, api_key: str = Depends(verify_api_key)):
    # Do here your stuff with the file
    if not file:
        raise HTTPException(status_code=400, detail="File is required")
    t = time.time()
    docs = []
    file_path = save_upload_file(
        upload_file=file,
        base_folder=DB_BASE_FOLDER)
    loader = PyPDFium2Loader(file_path)
    docs.extend(loader.load())
    
    index_documents(docs=docs, index_name=username, base_folder=DB_BASE_FOLDER, embedder=embedder)   
    print(f"TOTAL time: {time.time()-t:.2f} seconds") 
    
    return {"detail": "File added successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
