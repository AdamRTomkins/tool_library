from pydantic import BaseModel

class EmbeddingConfig(BaseModel):
    api_url: str
    api_key: str
    embedding_id: str