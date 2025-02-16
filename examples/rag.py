"""
This example demonstrates how to build a retrieval-augmented chatbot using the AutoGRAMS framework. 
The chatbot can answer user questions about documentation stored in a local docs folder. 

IMPORTANT:
- This file should NOT contain a main entry point or code that starts the chatbot directly. 
- You will import 'chatbot' from this file into your own run script and execute it there.
- Make sure you have installed:
    pip install autograms

"""

import os
from typing import List
from autograms.agent_modules.vector_stores import AutoIndex
from common.document_loader import load_documents
from autograms_doc_search import chunk_document

# AutoGRAMS imports
from autograms import autograms_function
from autograms.functional import (
    reply,
    reply_instruction,
    multiple_choice,
    yes_or_no,
    set_system_prompt,
    thought,
    silent_thought,
    get_system_prompt,
    GOTO,
    RETURNTO,
    append_system_prompt,
    extract_last_user_reply
)
from autograms.nodes import location
from autograms.memory import init_user_globals


# --------------------
# Global constants and variables
# --------------------
MAX_CHUNK_CHARS = 2048  # The maximum chunk size in characters
CHUNK_OVERLAP_RAW = 64  # Overlap for chunk_raw
DOC_HEADING_TOKEN = "##"  # We consider '##' any heading marker (common in Markdown)
MAX_SEARCH_ITERATIONS = 1

# 
# doc_description: A short summary of what the docs are about
# docs: A list of all raw text from our documentation
# index: Our FAISS/Numpy index for vector search (depending on if faiss is installed)
#
doc_description = "Documentation for rag."
docs = []
index = None  # Will be set by init_chatbot()

# --------------------
# Initialization function (NOT an @autograms_function)
# This is called externally AFTER the Autogram config is set up, 
#  before the conversation starts.
# --------------------
def init_chatbot(docs_folder="docs", doc_summary="Documentation for the autograms codebase."):
    """
    Initializes the global variables for doc-related RAG.
    This function is optional, but it allows you to configure your 
    document indexing after the AutoGRAMS environment is ready.
    
    :param docs_folder: Path to the folder containing docs to be indexed.
    :param doc_summary: Short text description of your docs.
    """
    global doc_description
    global docs
    global index

    doc_description = doc_summary

    docs = load_documents()
    if not docs:
        return
    chunked_docs = []
    for file_data in docs:
        # file_data is a dict with {'file_path':..., 'content':...}
        new_chunks = chunk_document(file_data['text'], file_data['metadata'])
        for chunk in new_chunks:
            chunked_docs.append(chunk)

    index = AutoIndex.from_texts(chunked_docs)

# --------------------
# The main chatbot function
# --------------------
@autograms_function()
def chatbot():
    """
    This is the root function of our doc-QA chatbot. 
    It enters an infinite loop, replying to the user, 
    then deciding if searching the docs is useful, 
    then possibly calling the retrieve() function to handle retrieval.
    """

    # Set a broad system prompt describing the chatbot
    set_system_prompt(
        "Your role is to give replies in conversational contexts and answer multiple choice questions. Be sure to be follow the INSTRUCTION you are given for your reply. Your main function is to look up and answer questions about a codebase for chatbots called autograms."  
    )

    # Greet the user and enter a loop where we continuously get user input & respond
    reply("Hello! I can answer your questions about autograms. what would you like to know?")
    doc_info=None

    while True:

        user_question = extract_last_user_reply()

        if not(doc_info is None):
            follow_up = yes_or_no(f"Consider the following Information:{doc_info}\nIs this information relevant to the following query '{user_question}'?")
            if follow_up:
                if yes_or_no(f"Consider the following Information:{doc_info}\nDoes this contain enough information to completely answer this query '{user_question}'?"):
                    reply_instruction(f"Using the docs, provide an to the user's question. Include details from:\n{doc_info}")
                    continue
            else:
                doc_info=None

        doc_info = retrieve(user_question,doc_info)  # doc_info is a summary of relevant results
        give_reply(doc_info)

@autograms_function()
def give_reply(doc_info):

    append_system_prompt(f"\n\nThis is additional information that should help answering questions \n{doc_info}")
    # Decide how confident we are that we found an answer

    reply_instruction(
        f"provide an to the user's question."
    )

# --------------------
# The retrieval function
# --------------------
@autograms_function(conv_scope="normal")
def retrieve(user_question,previous_context=None):
    """
    This function attempts to search the docs for relevant info to the user's question,
    up to MAX_SEARCH_ITERATIONS times. It returns a text summary of the found info.

    'conv_scope="normal"' means that thoughts in this function do not persist 
    once we return to the calling function. 
    (We don't want the models conversation memory to get filled with many search steps.)
    """
    append_system_prompt(f"\n\nHere is some additional context that may help {previous_context}")
    
    for i in range(MAX_SEARCH_ITERATIONS):
        # Think about how to refine the query
        if i==0 and not previous_context is None :
            refine_prompt = user_question

        else:
            refine_prompt = thought(
                f"What search query should we use next to help find information for the user? "
                "We can refine our query if needed. If it doesn't seem we need more iteration, say 'DONE' to stop early."
            )

        # Actually run a search on the doc index
        # (We assume 'index' has been initialized by init_chatbot.)
        search_results = index.similarity_search(refine_prompt, k=4)

        # Summarize the search results text
        for doc in search_results:
            append_system_prompt(f"\n\nHere is some additional context that may help {doc['text']}")
        combined_text = "\n\n".join([doc['text'] for doc in search_results])

        # Decide if we should keep going or not
        cont_search = yes_or_no(
            f"We were originally trying to find out he following information: {user_question}.\n\nDo we need to continue searching (maybe we haven't found a complete answer yet)?"
        )
        if not cont_search:
            break

    # Return the combined doc info to the caller
    return combined_text