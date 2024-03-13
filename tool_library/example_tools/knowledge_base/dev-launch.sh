export KB_USE_AUTH="False"
export KB_MASTER_KEY="CARLOS_ANNIE_ADAM_KALAVAI"
export DB_BASE_FOLDER="./kb_data"
export EMBEDDING_API_URL="https://api.openai.com/v1/embeddings"
export EMBEDDING_MODEL_ID="text-embedding-3-small"

uvicorn api:app --reload --port 8001