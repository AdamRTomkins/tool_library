import os
import logging

from auth import auth_user


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)   
 
# Pull in all our environment variables
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-0125")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)
RETRIEVER_BASE_URL = os.environ.get("RETRIEVER_BASE_URL")
RETRIEVER_API_KEY = os.environ.get("RETRIEVER_API_KEY")

USE_AUTH = os.environ.get("CHAT_USE_AUTH", "False").lower() in ("true", "1", "t", "yes")


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

# Move these too a specific REPO for the RAG when we have it
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
        user = auth_user(username, password)
        if user is not None:
            api_key = user["api_key"]
            return cl.User(
                identifier=username, metadata={"role": "user", "provider": "credentials", "api_key": api_key}
            )
        else:
            return None

@cl.on_chat_start
async def on_chat_start():  
    template = """Answer the question based only on the following context:
    
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    if LLM_BASE_URL:
        llm = ChatOpenAI(base_url=LLM_BASE_URL, model=LLM_MODEL)
    else:
        llm = ChatOpenAI(model=LLM_MODEL)
 
    retriever = KalavaiRetriever(base_url=RETRIEVER_BASE_URL, api_key=RETRIEVER_API_KEY)

    cl.user_session.set("retriever", retriever)

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


async def knowledge_describe(**args):

    # get the faiss client
    retriever = cl.user_session.get("retriever")
    faiss_client = retriever.faiss_client
    url = faiss_client.base_url
    api_key = faiss_client.api_key

    info = faiss_client.index_info()
    num = info["number_of_items"]
    
    await cl.Message(content=f"Your Knowledge Base: \n- URL: {url} \n- API_KEY {api_key}\n Number of Documents: {num}").send()


async def knowledge_update(url=None, api_key=None):

    # get the faiss client
    retriever = cl.user_session.get("retriever")
    faiss_client = retriever.faiss_client
    updates = []
    if url:
        faiss_client.base_url = url
        updates.append("URL")
    if api_key:
        faiss_client.api_key = api_key
        updates.append("API KEY")
    url = faiss_client.base_url
    api_key = faiss_client.api_key

    await cl.Message(content=f"Updating the Knowledge Base: \n- URL: {url} \n- API_KEY {api_key}").send()

async def knowledge_search(text:str):
    # get the faiss client
    retriever = cl.user_session.get("retriever")
    faiss_client = retriever.faiss_client
    docs = faiss_client.search(text)

    for doc in docs:
        await cl.Message(content=doc).send()

cli = {
    "knowledge": {
        "description": "This tool allows you to search for knowledge",
        "functions": {
            "describe": knowledge_describe,
            "update": knowledge_update,
        }
    }
}

async def triage(message: cl.Message):
    # if its about 
    if message.content.startswith("/"):
        args = message.content[1:].split(" ")
        command = args[0]
        
        if len(args) < 2:
            function = "describe"
            kwargs = {}
        else:
            function = args[1]
            kwargs = args[2:]
            kwargs = {k.lower(): v for k, v in [kw.split("=") for kw in kwargs]}

        if command in cli:
            if function in cli[command]["functions"]:
                await cli[command]["functions"][function](**kwargs)
            else:
                await cl.Message(content=f"I don't understand that function {function}. Please use one of {cli[command]['functions'].keys()}").send()
        else:
            await cl.Message(content=f"I don't understand that command {command}. Please use one of {cli.keys()}").send()
        return True
    
    return False


@cl.on_message
async def on_message(message: cl.Message):

    triaged = await triage(message)
    if triaged: return
    
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
