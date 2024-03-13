import logging
import os

from chainlit.input_widget import Switch

from auth import auth_user

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Pull in all our environment variables
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-0125")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)
DEFAULT_NAMESPACE = os.environ.get("DEFAULT_NAMESPACE", "carlosfm")

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
                "similarity_threshold": 0.5,
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


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    user = auth_user(username, password)

    if user is not None:
        api_key = user["api_key"]

        knowledge_base_url = user["knowledge_base_url"]
        knowledge_base_api = user["api_key"]

        return cl.User(
            identifier=username,
            metadata={
                "role": "user",
                "provider": "credentials",
                "api_key": api_key,
                "knowledge_base_url": knowledge_base_url,
                "knowledge_base_api": knowledge_base_api,
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

    namespaces = [DEFAULT_NAMESPACE, namespace]
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

    print("Namespaces", namespaces)

    # cl.user_session.set("kb_client", kb_client)

    template = """Answer the question based only on the following context:
    
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    if LLM_BASE_URL:
        llm = ChatOpenAI(base_url=LLM_BASE_URL, model=LLM_MODEL)
    else:
        llm = ChatOpenAI(model=LLM_MODEL)

    print(cl.user_session.get("user").metadata)

    knowledge_base_url = (
        cl.user_session.get("user").metadata.get("knowledge_base_url"),
    )
    knowledge_base_api_key = cl.user_session.get("user").metadata.get(
        "knowledge_base_api"
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
        base_url=knowledge_base_url, api_key=knowledge_base_api_key
    )

    cl.user_session.set("client", client)
    cl.user_session.set("prompt", prompt)
    cl.user_session.set("llm", llm)


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


@cl.action_callback("index_files")
async def on_action(action: cl.Action):

    file = await cl.AskFileMessage(
        content="Please upload a File to begin.", accept={"application/pdf": [".pdf"]}
    ).send()

    await index_files(file)


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
            outputs.append(res)

    elements = []
    for res in outputs:
        elements.extend([
                        cl.Text(name="Result", content=str(res.json()), display="inline", language="json"),
                        cl.Pdf(name="pdf1", display="inline", url=address+res.json()["url"])
        ])
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

    files = [f for f in message.elements if type(f) == cl.File]

    if files:
        await index_files(files)

    # Pull the session vars
    client = cl.user_session.get("client")
    prompt = cl.user_session.get("prompt")
    namespace = cl.user_session.get("namespace")
    llm = cl.user_session.get("llm")
    search_namespaces = cl.user_session.get("search_namespaces")

    print(search_namespaces)

    msg = cl.Message(content="")
    await msg.send()

    search_results = []

    # Search The Rag Instance
    with cl.Step(name="Search") as parent_step:
        parent_step.input = f" search {message.content} in {','.join(search_namespaces)} knowledge bases"

        for namespace in search_namespaces:
            with cl.Step(
                name=f'Search {namespace.replace("_", " ".title())} Knowledge Base'
            ) as individual_search:
                individual_search.input = message.content
                try:
                    context = client.search(message.content, namespace=namespace)
                    for c in context:
                        c["source"] = namespace
                    search_results.extend(context)
                    individual_search.output = context

                    elements = [
                        cl.Text(
                            name="Result",
                            content=str(p["page_content"]),
                            display="inline",
                        )
                        for p in context
                    ]

                    element_names = "Relavent Documents: \n " + "\n".join(
                        [
                            f'{p["metadata"]["source"]} page {p["metadata"]["page"]}'
                            for p in context
                        ]
                    )
                    individual_search.output = element_names

                except Exception as e:
                    individual_search.output = f"{namespace}: {str(e)}"
                    continue

        # sort search_results by the similarity score in x["state"]["query"]["similarity_score"]
        search_results = sorted(
            search_results,
            key=lambda x: x["state"]["query_similarity_score"],
            reverse=True,
        )
        # limit to the top 5
        search_results = search_results[:5]

        elements = [
            cl.Text(name="Result", content=str(p["page_content"]), display="side")
            for p in search_results
        ]

        element_names = "Relavent Documents: \n " + "\n".join(
            [f'{p["metadata"]["source"]} page {p["metadata"]["page"]}' for p in context]
        )

        # parent_step.elements = elements
        parent_step.output = element_names

        if len(context) == 0:
            # This checks if we have found any context in the retrieval
            msg.content = "No context found"
            await msg.update()
            return

        context_str = "\n".join([p["page_content"] for p in context])

        # Limit the context in tesgint:
        context_str = context_str[:1000]
        msg.content = element_names
        msg.elements = elements

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
