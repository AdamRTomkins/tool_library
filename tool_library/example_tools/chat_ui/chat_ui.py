import os
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)   
 
# Pull in all our environment variables
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "phi-2")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
RETRIEVER_BASE_URL = os.environ.get("RETRIEVER_BASE_URL")
RETRIEVER_API_KEY = os.environ.get("RETRIEVER_API_KEY")

USE_AUTH = os.environ.get("CHAT_USE_AUTH", "False").lower() in ("true", "1", "t", "yes")
USERNAME = os.environ.get("USERNAME", "admin")
PASSWORD = os.environ.get("PASSWORD", "kalavai")


logger.info(f"LLM_API_KEY: {str(LLM_API_KEY)[:4]}...")
logger.info(f"LLM_MODEL: {LLM_MODEL}")
logger.info(f"LLM_BASE_URL: {LLM_BASE_URL}")
logger.info(f"RETRIEVER_BASE_URL: {RETRIEVER_BASE_URL}")
logger.info(f"RETRIEVER_API_KEY: {str(RETRIEVER_API_KEY)[:4]}...")

os.environ["OPENAI_API_KEY"] = LLM_API_KEY


from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
from typing import List

from chainlit.input_widget import *
from typing import Optional

import chainlit as cl
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

class KalavaiRetriever(BaseRetriever):
    faiss_client: Optional[FAISSClient] = None

    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self.faiss_client = FAISSClient(base_url, api_key)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            search_results = self.faiss_client.search(query)
            documents = [Document(page_content=result['text'], metadata=result['attributes']) for result in search_results]
        except Exception as e:
            documents = []
            logger.error(f"Error retrieving documents: {e}")
        return documents


if USE_AUTH:
    @cl.password_auth_callback
    def auth_callback(username: str, password: str):
        # Fetch the user matching username from your database
        # and compare the hashed password with the value stored in the database
        if (username, password) == (USERNAME, PASSWORD):
            return cl.User(
                identifier="admin", metadata={"role": "admin", "provider": "credentials"}
            )
        else:
            return None



@cl.on_chat_start
async def main():
    pass

@cl.on_chat_start
async def on_chat_start():  
    template = """Answer the question based only on the following context:
    
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(base_url=LLM_BASE_URL, model=LLM_MODEL)
    retriever = KalavaiRetriever(base_url=RETRIEVER_BASE_URL, api_key=RETRIEVER_API_KEY)

    def format_docs(docs):
        print(docs)
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("runnable", chain)


@cl.on_message
async def on_message(message: cl.Message):

    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
