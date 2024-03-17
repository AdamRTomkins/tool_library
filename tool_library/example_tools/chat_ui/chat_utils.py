import time
import requests
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
import chainlit as cl
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


PROMPT_TEMPLATE = """You are a helpful academic assistant designed to help MBA students in their coursework. Your job is to get answers to user queries using only the information within the following context:
```{context}```

Query: {question}
Answer: """


# Move these too a specific REPO for the RAG when we have it
class KnowledgeClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"X-API-KEY": self.api_key, "accept": "application/json"}

    def search(self, query: str, namespace: str) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/search/",
            params={
                "query": query,
                "index_name": namespace,
                "top_k": 5,
                "similarity_threshold": 0.35,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def add_items(self, username: str, files: List) -> Dict[str, Any]:
        # Validate input
        if not isinstance(files, list):
            files = [files]

        # Prepare files for multipart/form-data
        # Map file extensions to MIME types
        mime_types = {
            "pdf": "application/pdf",
            "txt": "text/plain",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }

        # Prepare files for multipart/form-data
        multipart_files = [
            (
                "file",
                (
                    f.name,
                    f,
                    mime_types.get(
                        f.name.rsplit(".", 1)[-1], "application/octet-stream"
                    ),
                ),
            )
            for f in files
            if f.name.rsplit(".", 1)[-1] in mime_types
        ]

        data = {"username": username}
        response = requests.post(
            f"{self.base_url}/add",
            files=multipart_files,
            data=data,
            headers=self.headers,
        )

        # Close the file objects
        for f in files:
            f.close()

        # Return the response
        return response.json()  # Assuming the response is in JSON format

    def index_info(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/info/", headers=self.headers)
        response.raise_for_status()
        return response.json()


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.msg = cl.Message(content="")

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.msg.stream_token(token)

    async def on_llm_end(self, response: str, **kwargs):
        await self.msg.send()
        self.msg = cl.Message(content="")


async def typed_answer(message):
    tokens = message.split()
    msg = cl.Message(content="")
    time.sleep(0.5)
    for token in tokens:
        if "<line>" in token:
            await msg.stream_token("\n")
        else:
            await msg.stream_token(f"{token} ")
        time.sleep(0.05)
    await msg.send()
    

def format_docs(docs):
    if len(docs) > 0:
        reorder = LongContextReorder()
        docs = reorder.transform_documents(docs)
        return "\n* ".join(doc["page_content"] for doc in docs)
    else:
        return "Context is empty"


def create_rag_chain(llm):
    
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def load_llm(openai_api_key, openai_model, openai_base_url=None):
    if openai_base_url:
        return ChatOpenAI(
            api_key=openai_api_key,
            model=openai_model,
            temperature=0.0,
            base_url=openai_base_url)
    else:
        return ChatOpenAI(
            api_key=openai_api_key,
            model=openai_model,
            temperature=0.0)