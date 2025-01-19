from common.chat_history import ChatHistoryManager
from common.config import LLM_MODEL
from common.document_loader import load_documents
from common.vectorstore import create_vectorstore
from common.retriever import create_retriever, create_chain

from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama

from autograms import autograms_function
from autograms.functional import set_system_prompt, extract_last_user_reply, reply

global vector_db
global retriever
global memory
global llm
global chain


    


# Initial message to introduce the chatbot and its functionality
intro_message = (
    "Hi. My name is Kappa and I'm an AI for RAG. "
    "I can help solve some problems for you to work through. "
)


@autograms_function()
def chatbot():

    documents = load_documents()
    if not documents:
        raise Exception("No documents available")

    llm = ChatOllama(model=LLM_MODEL)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
    documents = load_documents()
    vector_db = create_vectorstore(documents)
    retriever = create_retriever(vector_db, llm, True)

    # Create the chain
    chain = create_chain(retriever, llm, memory)

    # Set a system prompt to define the AI's role and behavior
    set_system_prompt(
        """You are an AI assistant tasked with answering questions strictly based on the provided documents.
    Do not use external knowledge or provide answers unrelated to the content retrieved from the documents.
    
    Documents retrieved:
    {context}
    
    Conversation so far:
    {history}
    
    User Question: {question}
    
    Provide a detailed and accurate response based on the documents above."""
    )

    # Start the interaction with the introduction message
    reply(intro_message)

    while True:

        user_question = extract_last_user_reply()
        response =  chain.invoke(input={"question": user_question})

        answer = response

        reply(user_question)