import logging
import time
import os

from chainlit.input_widget import Switch

from platform_utils import (
    auth_user,
    fetch_installer
)
from user_status import (
    get_user_status,
    log_user_query
)
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
NO_CONTEXT_MESSAGE = "I don't know how to respond to that. I could not find any relevant information in the documents I have access too right now."

# Temporary Override Functions
OVERRIDE_KNOWLEDGE_BASE_URL = os.environ.get("OVERRIDE_KNOWLEDGE_BASE_URL", "https://knowledgebase.test.k8s.mvp.kalavai.net")

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

# Temporary Download Prompt
# TODO: Add a download page!
async def download_prompt(msg=None):
    if msg is None:
        msg = cl.Message(content="")
        await msg.send()

    content= """You have not yet registered with the Kalavai network.

    Kalavai is a decentralised AI network, which allows any registared user to use the shared AI resources.

    To register, please install the Kalavai application and follow the instructions to create an account.
    
    #TODO: Add a link to the download page
    """

    msg.content = content
    await msg.update()

async def upload_files(message, user_status):

    files = [f for f in message.elements if type(f) == cl.File]

    if files:
        if user_status.registered:
            await index_files(files)
            await cl.Message(content="Files uploaded successfully").send()
        else:
            await download_prompt()
            

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    user = auth_user(username, password)

    if user is not None:
        api_key = user["api_key"]

        return cl.User(
            identifier=user["username"],
            metadata={
                "username": user["username"],
                "role": "user",
                "provider": "credentials",
                "api_key": api_key,
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
    namespace = cl.user_session.get("user").metadata.get("namespace", None)
    gated_namespace = cl.user_session.get("user").metadata.get("gated_namespaces",[])
    public_namespace = cl.user_session.get("user").metadata.get("public_namespaces",[DEFAULT_NAMESPACE])

    if namespace is None:
        logger.warning(
            "No namespace found for user, defaulting to extracting from the identifier. This needs to be passed in from the login."
        )
        namespace = cl.user_session.get("user").identifier.split("@")[0]

    search_namespaces = {
        "open":public_namespace,
        "gated":gated_namespace,
        "private":[namespace] # This is your personal namespace
    }

    cl.user_session.set("username", cl.user_session.get("user").identifier)
    cl.user_session.set("namespace", namespace)
    cl.user_session.set("search_namespaces", search_namespaces)

    #settings = await cl.ChatSettings(
    #    [
    #        Switch(
    #            id=f"ns_{namespace}",
    #            label=namespace.replace("_", " ").title(),
    #            initial=True,
    #        )
    #        for namespace in namespaces
    #    ]
    #).send()

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


async def format_sources(base_url, source_documents, answer= ""):
    """ A Stand In functio that takes a list of sources and formats they for display in the chat"""
    # This may not be the best way to display them.

    if not source_documents:
        return

    text_elements = []

    for source_idx, source_doc in enumerate(source_documents):
        source = source_doc["metadata"]["source"]
        
        score = source_doc["state"]["query_similarity_score"] * 100

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
                display="side", # inline side page
                url=base_url + source,
                page=source_page
            )
        text_elements.append(element)
    source_names = [text_el.name for text_el in text_elements]

    if source_names:
        source_names = "\n ".join(source_names)
        answer += f"\n\nSources: \n{source_names}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


@cl.action_callback("install_button")
async def on_action(action):
    
    # fetch installer and download
    temp_file, filename = fetch_installer(selected_os=action.value)
    with temp_file as filepath:
        elements = [
            cl.File(
                name=filename,
                path=filepath,
                display="inline",
            ),
        ]

        await cl.Message(
            content="Your installer is ready. Run Kalavai client to unlock all content", elements=elements
        ).send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()


@cl.on_message
async def on_message(message: cl.Message):

    msg = await cl.Message(content="").send()

    # Pull the session vars
    client = cl.user_session.get("client")
    chain = cl.user_session.get("chain")
    username = cl.user_session.get("username")
    search_namespaces = cl.user_session.get("search_namespaces")

    # get the user status
    with cl.Step(name=f"Checking User Status") as step:
        user_status = get_user_status(username)
        step.input = username
        step.output = user_status

    # Deal with file uploads and verification
    await upload_files(message, user_status)
    
    callbacks = [
            cl.AsyncLangchainCallbackHandler(),
            StreamHandler()
        ]
        
    t = time.time()
    # Search the knowledge base
    # Todo Search Multiple Indexes

    async def search_sources(search_namespaces, message):
        results = {}
        for domain, namespaces  in search_namespaces.items():
            results[domain] = {}
            for namespace in namespaces:
                with cl.Step(name=f"Searching {domain.title()} Knowledge Base: {namespace}") as step:
                    context = client.search(message, namespace=namespace)
                    step.input = message
                    step.output = context
                    results[domain][namespace] = context

        for domain in results:
            results[domain] = sum(results[domain].values(),[])
            results[domain] = sorted(results[domain], key=lambda x: x["state"]["query_similarity_score"], reverse=True)

        return results
    
    results = await search_sources(search_namespaces, message.content)

    open_context = results["open"]
    gated_context = results["gated"]
    private_context = results["private"]

    context = open_context[:5]
    curtailed_context = open_context[5:]

    # Could we add some extra logic here to see if perhaps the "Best Match" is in the other gates/private sources?
    # IE, we can say "The best documents to answer your question are gated/private, would you like to share your resources to access them?"

    # Any user that is registered can access the gated context
    # Update to user_status.currently_sharing for a more restrictive model.
    extra_actions = []
    if user_status.registered:
        context += gated_context
        gated_context = []

    # Only users currently sharing can access the private context
    if user_status.currently_sharing:
        context += private_context
        private_context = []

    if len(context) == 0:
        res = {"answer": NO_CONTEXT_MESSAGE, "context": []}
    else:
        context_str = format_docs(context)
        generated_message = await chain.ainvoke({"question": message.content, "context": context_str}, {"callbacks": callbacks}, user=username)
        res = {"answer": generated_message, "context": context}

    # Mention that there are more results
    if len(curtailed_context) > 0:
        res["answer"] += f"\n\nThere are {len(curtailed_context)} more results that could be relevant. Please refine your search to see more."

    if not user_status.registered:
        # Remind users to install client
        extra_actions.extend([
            cl.Action(label="Kalavai client (MacOS)", name="install_button", value="macos", description="Click to download installer"),
            cl.Action(label="Kalavai client (Windows)", name="install_button", value="windows", description="Click to download installer"),
            cl.Action(label="Kalavai client (Ubuntu)", name="install_button", value="ubuntu", description="Click to download installer")
        ])
        res["answer"] = "I've noticed you haven't shared your device yet. Install our client app to start sharing and unlock more functionality."

    # If user not sharing, curtail private and gated content
    if not user_status.currently_sharing:
        # Mention that there are gated sources that could be relevant to your query
        # and that they can gain access to these sources by registering
        if len(gated_context) > 0:
            res["answer"] += f"\n\nI found {len(gated_context)} relevant matches in private modules."

        # Mention that there are private sources that could be relevant to your query
        # and that they can gain access to these sources by sharing their resoures on the kalavai network
        if len(private_context) > 0:
            res["answer"] += f"\n\nI found __{len(private_context)} relevant matches against your private documents__."
        
        if len(gated_context) + len(private_context) > 0:
            res["answer"] += "\n\n**To gain access to these sources, share your device with the kalavai client.**"

    # log query from the user for monitoring
    n_responses = len(curtailed_context)+len(gated_context)+len(private_context)
    logs = log_user_query(
        username=username,
        n_responses=n_responses
    )
    logger.debug(f"Logged answer result (n_responses: {n_responses}): {logs}")
    print(username, n_responses, logs)
    await typed_answer(
        f"{res['answer']}\n[{time.time()-t:.2f} seconds]",
        actions=extra_actions
    )

    # Add a quick and dirtly formatting of each of the sources. 
    await format_sources(client.base_url, open_context, answer="The following open sources could be relevant to your query:\n")
    await format_sources(client.base_url, gated_context, answer="The following gated sources could be relevant to your query:\n")
    await format_sources(client.base_url, private_context, answer="The following private sources could be relevant to your query:\n")

    if gated_context or private_context:
        await download_prompt()
