from typing import List, Dict, Any
from fastapi import UploadFile, File, Form

from fastapi import FastAPI, HTTPException, Depends
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles

import time
import os
import logging

from models import EmbeddingConfig
from utils import (
    save_upload_file,
    index_documents,
    load_embedding,
    load_document_retriever,
    process_retriever_results,
    load_documents_from_file
)


# Configure logging at the application's entry point
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration handling
# Conservatively assume Auth is enabled if anything other than these false values are set.
USE_AUTH = not os.getenv("KB_USE_AUTH", "True").lower() in ("false", "0", "f", "no")
MASTER_KEY = os.getenv("KB_MASTER_KEY")
DB_BASE_FOLDER = os.getenv("DB_BASE_FOLDER", "./data")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings") #"https://embedding.test.k8s.mvp.kalavai.net")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "None")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small") #"BAAI/bge-large-en-v1.5")

if USE_AUTH:
    assert MASTER_KEY is not None, "If you are using auth, you must set a master key using the 'KB_MASTER_KEY' environment variable."
else:
    logger.warning("Warning: Authentication is disabled. This should only be used for testing.")

# check if the base folder exists, if not create it
os.makedirs(DB_BASE_FOLDER, exist_ok=True)

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

# Mount the static directory to be served by FastAPI
app.mount("/static", StaticFiles(directory=DB_BASE_FOLDER), name="static")

embedding = load_embedding(
    embedder_config=EmbeddingConfig(
        api_url=EMBEDDING_API_URL,
        api_key=EMBEDDING_API_KEY,
        embedding_id=EMBEDDING_MODEL_ID
    )
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
async def search_index(query: str, index_name: str, top_k: int, similarity_threshold: float, api_key: str = Depends(verify_api_key)):
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")
    if not index_name:
        raise HTTPException(status_code=400, detail="Index name is required")
    if not top_k or not similarity_threshold:
        raise HTTPException(status_code=400, detail="Top_k and similarity_thresholds are required")
    retriever = load_document_retriever(
        index_name=index_name,
        embedding=embedding,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        base_path=DB_BASE_FOLDER)
    import time
    t = time.time()
    results = retriever.invoke(query)
    print(f"Search time: {time.time()-t:.2f}")
    
    return process_retriever_results(results)


# Endpoint to add pdf files to the index
@app.post("/add")
async def upload_file(file: UploadFile = File(...), username: str = Form(None), api_key: str = Depends(verify_api_key)):
    """Accepts files of the following formats: .txt, .docx, .pdf"""
    if not file:
        raise HTTPException(status_code=400, detail="File is required")
    t = time.time()


    print("USERNAME", username)

    file_path, file_name = save_upload_file(
        upload_file=file,
        base_folder=DB_BASE_FOLDER,
        username=username
    )
    documents = load_documents_from_file(file_path)
    
    index_documents(docs=documents, index_name=username, base_folder=DB_BASE_FOLDER, embedding=embedding)   
    print(f"TOTAL time: {time.time()-t:.2f} seconds") 
    
    return {"detail": "File added successfully", "filename": file_name, "url": f"/static/{file_name }", "username":username, "original_filename":file.filename}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
