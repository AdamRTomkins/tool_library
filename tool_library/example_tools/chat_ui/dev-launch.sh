#!/bin/sh

#export LLM_API_KEY="sk-..."
#export LLM_MODEL="gpt-3.5-turbo"

export LLM_API_KEY="None"
export LLM_MODEL="phi-2"
export LLM_BASE_URL="http://178.62.13.8:31095"


export RETRIEVER_BASE_URL="http://178.62.13.8:31645"
export RETRIEVER_API_KEY="None"

export CHAT_USE_AUTH="True"
export USERNAME="Adam"
export PASSWORD="1234"

#pip install -r requirements.txt 
sh prepare.sh
chainlit run chat_ui.py -h --port 8011 -w