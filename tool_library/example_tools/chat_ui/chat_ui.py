import logging
import time
import os

from chainlit.input_widget import Switch

from auth import auth_user
from chat_utils import (
    StreamHandler,
    typed_answer,
    format_docs,
    create_rag_chain,
    load_llm,
    KnowledgeClient
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Pull in all our environment variables
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-0125")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)
DEFAULT_NAMESPACE = os.environ.get("DEFAULT_NAMESPACE", "cjbs")
NO_CONTEXT_MESSAGE = "I'm sorry, I don't know how to respond to that. I could not find any relevant information in the documents you've shared with me. Maybe try again, this time adding a bit more context about the problem or a few keywords."

# Temporary Override Functions
OVERRIDE_KNOWLEDGE_BASE_URL = os.environ["OVERRIDE_KNOWLEDGE_BASE_URL"]

logger.info(f"LLM_API_KEY: {str(LLM_API_KEY)[:4]}...")
logger.info(f"LLM_MODEL: {LLM_MODEL}")
logger.info(f"LLM_BASE_URL: {LLM_BASE_URL}")
logger.info(f"DEFAULT_NAMESPACE: {DEFAULT_NAMESPACE}")
logger.info(f"OVERRIDE_KNOWLEDGE_BASE_URL: {OVERRIDE_KNOWLEDGE_BASE_URL}")


os.environ["OPENAI_API_KEY"] = LLM_API_KEY


from typing import Any, Dict, List

import chainlit as cl
import requests
from chainlit.input_widget import *
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    user = auth_user(username, password)

    if user is not None:
        api_key = user["api_key"]

        #knowledge_base_url = user["knowledge_base_url"]
        #knowledge_base_api = user["api_key"]

        return cl.User(
            identifier=username,
            metadata={
                "username": user["username"],
                "role": "user",
                "provider": "credentials",
                "api_key": api_key,
                #"knowledge_base_url": knowledge_base_url,
                #"knowledge_base_api": knowledge_base_api,
                "namespace": user["username"]
            },
        )
    else:
        return None


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

    search_namespaces = [
        k.split("_")[1] for k, v in settings.items() if k.startswith("ns_") and v
    ]
    cl.user_session.set("search_namespaces", search_namespaces)


@cl.on_chat_start
async def on_chat_start():

    # get the namespace for this user:
    namespace = cl.user_session.get("user").metadata.get("namespace")
    if namespace is None:
        logger.warning(
            "No namespace found for user, defaulting to extracting from the identifier. This needs to be passed in from the login."
        )
        namespace = cl.user_session.get("user").identifier.split("@")[0]

    namespaces = [namespace] #[DEFAULT_NAMESPACE, namespace]
    cl.user_session.set("namespace", namespace)
    cl.user_session.set("search_namespaces", namespaces)

    settings = await cl.ChatSettings(
        [
            Switch(
                id=f"ns_{namespace}",
                label=namespace.replace("_", " ").title(),
                initial=True,
            )
            for namespace in namespaces
        ]
    ).send()

    # cl.user_session.set("kb_client", kb_client)

    llm = load_llm(openai_api_key=LLM_API_KEY, openai_model=LLM_MODEL, openai_base_url=LLM_BASE_URL)
    chain = create_rag_chain(llm)

    knowledge_base_url = (
        cl.user_session.get("user").metadata.get("knowledge_base_url"),
    )
    api_key = cl.user_session.get("user").metadata.get(
        "api_key"
    )

    #######################################
    #
    # WARNING: OVERRIDE TEH SUPPLIED API KEY and URL
    #
    # THIS SHOULD BE GIVEN FROM THE MAIN SYSTEM
    #
    #######################################

    if OVERRIDE_KNOWLEDGE_BASE_URL:
        knowledge_base_url = OVERRIDE_KNOWLEDGE_BASE_URL

    client = KnowledgeClient(
        base_url=knowledge_base_url, api_key=api_key
    )

    cl.user_session.set("client", client)
    cl.user_session.set("chain", chain)


async def index_files(file_objects):
    namespace = cl.user_session.get("namespace")
    kb_client = cl.user_session.get("client")

    files = [open(f.path, "rb") for f in file_objects]

    outputs = []

    for file in files:
        async with cl.Step(name=f"Uploading File {file.name}") as step:
            # Step is sent as soon as the context manager is entered
            step.input = f"Uploading {file.name}"
            res = kb_client.add_items(files=file, username=namespace)
            step.output = res
            outputs.append(res)

    elements = []
    for res in outputs:
        elements.extend(
            [
                cl.Text(
                    name="Result", content=str(res), display="inline", language="json"
                ),
                cl.Pdf(
                    name=res["filename"],
                    display="side",
                    url=kb_client.base_url + res["url"],
                ),
            ]
        )
    
    response = "Uploaded Files \n" + "\n".join(
        [f"{res['filename']}" for res in outputs]
    )
    await cl.Message(content=response, elements=elements).send()


def forever_actions():
    # Generate your forever actions
    actions = [
        cl.Action(
            name="index_files",
            label=f"Add New Files to the Chat",
            description=f"Upload A new document",
            collapsed=True,
            value="",
        ),
    ]

    return actions


# @cl.action_callback("index_files")
# async def on_action(action: cl.Action):

#     file = await cl.AskFileMessage(
#             content="Please upload a PDF to begin!", accept={"application/pdf": [".pdf"], "text/plain": [".txt", ".docx"]}
#         ).send()

#     retriever = cl.user_session.get("client")
#     kb_client = retriever.kb_client
#     address = kb_client.base_url

#     files = [
#         open(f.path, "rb")
#         for f in file
#     ]

#     outputs = []

#     for f in files:
#         async with cl.Step(name="Test") as step:
#         # Step is sent as soon as the context manager is entered
#             step.input = f"Uploading {f.name}"
#             res = kb_client.add_items(files=f)
#             step.output = res
#             outputs.append(res)

#     elements = []
#     for res in outputs:
#         elements.extend([
#                         cl.Text(name="Result", content=str(res.json()), display="inline", language="json"),
#                         cl.Pdf(name="pdf1", display="inline", url=address+res.json()["url"])
#         ])
#     await cl.Message(content=f"Uploaded Files", elements=elements).send()


@cl.on_message
async def on_message(message: cl.Message):

    files = [f for f in message.elements if type(f) == cl.File]

    if files:
        await index_files(files)

    # Pull the session vars
    client = cl.user_session.get("client")
    chain = cl.user_session.get("chain")
    search_namespace = cl.user_session.get("search_namespaces")[0]
    
    
    callbacks = [
            cl.AsyncLangchainCallbackHandler(),
            StreamHandler()
        ]
        
    t = time.time()
    context = client.search(message.content, namespace=search_namespace)
    if len(context) == 0:
        res = {"answer": NO_CONTEXT_MESSAGE, "context": []}
    else:
        context_str = format_docs(context)
        generated_message = await chain.ainvoke({"question": message.content, "context": context_str}, {"callbacks": callbacks})
        res = {"answer": generated_message, "context": context}

    await typed_answer(res["answer"])
    
    answer = ""#res["answer"]
    source_documents = res["context"]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source = source_doc["metadata"]["source"]
            score = source_doc["state"]["query_similarity_score"] * 100
            # Create the text element referenced in the message
            # TODO: needs to link to original file on the server side
            if "page" in source_doc["metadata"]:
                    source_page = source_doc["metadata"]["page"] + 1 # 0 index
                    source_name = f"[{score:.0f}%] {source} (page {source_page})"
            else:
                source_name = f"[{score:.0f}%] {source}"
                
            if ".pdf" not in source.lower():
                element = cl.Text(content=source_doc["page_content"], name=source_name)    
            else:
                element = cl.Pdf(
                    name=source_name,
                    display="page", # inline side page
                    #path=source,
                    url=client.base_url + source,
                    page=source_page
                )
            text_elements.append(element)
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            source_names = "\n".join(source_names)
            answer += f"\n\nSources: \n{source_names}"
        else:
            answer += "\nNo sources found"

    # answer = await chain.ainvoke(message.content, {"callbacks": callbacks})
    
    response = f"{answer}\n[{time.time()-t:.2f} seconds]"
    
    await cl.Message(content=response, elements=text_elements).send()

