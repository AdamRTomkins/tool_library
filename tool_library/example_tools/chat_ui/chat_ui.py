import os
import logging

from auth import auth_user


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)   
 
# Pull in all our environment variables
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-0125")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)

USE_AUTH = os.environ.get("CHAT_USE_AUTH", "False").lower() in ("true", "1", "t", "yes")


logger.info(f"LLM_API_KEY: {str(LLM_API_KEY)[:4]}...")
logger.info(f"LLM_MODEL: {LLM_MODEL}")
logger.info(f"LLM_BASE_URL: {LLM_BASE_URL}")
#logger.info(f"RETRIEVER_BASE_URL: {RETRIEVER_BASE_URL}")
#logger.info(f"RETRIEVER_API_KEY: {str(RETRIEVER_API_KEY)[:4]}...")

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
class KnowledgeClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "X-API-KEY": self.api_key,
            'accept': 'application/json'
        }

    def search(self, query: str, namespace:str) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_url}/search/", 
            params={
                "query": query,
                "index_name":namespace,
                "top_k":5,
                "similarity_threshold":0.1
                },
            headers=self.headers)
        response.raise_for_status()
        return response.json()

    def add_items(self, files: List) -> Dict[str, Any]:
        # Prepare multipart/form-data files
        # Note: 'files' in requests.post() can be a list of tuples for multiple files
        # TODO: ALIGN WITH CARLOS API
        
        if type(files) != list:
            files = [files]
        multipart_files = [('file', (f.name, f, 'application/pdf')) for f in files]

        # Send the POST request to upload files
        response = requests.post(f"{self.base_url}/index/", files=multipart_files, headers=self.headers)

        # Close the file objects
        for f in files:
            f.close()

        return response



    def index_info(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/info/", headers=self.headers)
        response.raise_for_status()
        return response.json()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    user = auth_user(username, password)

    print(user)

    if user is not None:
        api_key = user["api_key"]

        knowledge_base_url = user["knowledge_base_url"]
        knowledge_base_api = user["api_key"]

        return cl.User(
            identifier=username, metadata={"role": "user", "provider": "credentials", "api_key": api_key, "knowledge_base_url":knowledge_base_url, "knowledge_base_api":knowledge_base_api}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():  

    #cl.user_session.set("kb_client", kb_client)

    template = """Answer the question based only on the following context:
    
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    if LLM_BASE_URL:
        llm = ChatOpenAI(base_url=LLM_BASE_URL, model=LLM_MODEL)
    else:
        llm = ChatOpenAI(model=LLM_MODEL)

    RETRIEVER_BASE_URL = cl.user_session.get("user").metadata.get("knowledge_base_url")
    
    RETRIEVER_API_KEY = cl.user_session.get("user").metadata.get("api_key")

    RETRIEVER_BASE_URL = "https://knowledgebase.test.k8s.mvp.kalavai.net"
    RETRIEVER_API_KEY = None

    client = KnowledgeClient(
        base_url=RETRIEVER_BASE_URL, 
        api_key= RETRIEVER_API_KEY
    )

    cl.user_session.set("client", client)
    cl.user_session.set("prompt", prompt)
    cl.user_session.set("namespace", "carlosfm")
    cl.user_session.set("llm", llm)
    
    msg = await cl.Message(content="Welcome", actions = forever_actions()).send()



@cl.action_callback("index_files")
async def on_action(action: cl.Action):

    file = await cl.AskFileMessage(
            content="Please upload a PDF to begin!", accept={"application/pdf": [".pdf"]}
        ).send()


    retriever = cl.user_session.get("retriever")
    kb_client = retriever.kb_client
    address = kb_client.base_url

    files = [
        open(f.path, "rb")
        for f in file
    ]

    outputs = []

    for file in files:
        async with cl.Step(name="Test") as step:
        # Step is sent as soon as the context manager is entered
            step.input = f"Upfloading {f.name}"
            res = kb_client.add_items(files=file)
            step.output = res





    elements = [
                    cl.Text(name="Result", content=str(res.json()), display="inline", language="json"),
                    cl.Pdf(name="pdf1", display="inline", url=address+res.json()["url"])
    ]
    await cl.Message(content=f"Uploaded Files", elements=elements).send()
     

def forever_actions():
    # Generate your forever actions
    actions = [
        cl.Action(
            name="index_files", 
            label = f'Add New Files to the Chat',
            description=f'Upload A new document',
            collapsed=True,
            value=""
        ),
    ]

    return actions

@cl.on_message
async def on_message(message: cl.Message):
        
    # Pull the session vars
    client = cl.user_session.get("client")
    prompt = cl.user_session.get("prompt")
    namespace = cl.user_session.get("namespace")
    llm = cl.user_session.get("llm")

    msg = cl.Message(content="")
    await msg.send()

    # Search The Rag Instance
    with cl.Step(name="Search") as step:
        step.input = message.content

        try:
            context = client.search(message.content, namespace=namespace)
        except Exception as e:
            step.output = str(e)
            msg.content = str(e)
            await msg.update()
            return
        elements = [
            cl.Text(name="Result", content=str(p["page_content"]), display="inline") for p in context
        ]

        element_names = "Relavent Documents: \n " + "\n".join([f'{p["metadata"]["source"]} page {p["metadata"]["page"]}' for p in context])

        step.elements = elements
        step.output = element_names

        if len(context) == 0:
            # This checks if we have found any context in the retrieval
            msg.content = "No context found"
            await msg.update()
            return
        
        context_str  = "\n".join([p["page_content"] for p in context])

        # Limit the context in tesgint:
        context_str = context_str[:1000]

        await msg.update()

    messages = prompt.format_messages(context=context_str, question=message.content)

    with cl.Step(name="Generate") as step:
        step.input = message.content
        reply = await llm.ainvoke(messages)
        step.output = reply.content
        msg.content = reply.content
        await msg.update()
        

    msg.actions = forever_actions()
    await msg.update()
    await msg.send()
