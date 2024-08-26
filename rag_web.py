from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

import time
import os


load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    st.session_state.loader = WebBaseLoader("https://paulgraham.com/greatwork.html")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vector = FAISS.from_documents(
        st.session_state.documents, st.session_state.embeddings
    )


st.title("Chat with Groq RAG AI  :) ")
# Initialize the Groq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")


def is_question_relevant_to_context(context, question):
    """
    This function checks if a question is relevant to the provided context
    using keyword matching.
    """
    context_keywords = set(context.split())
    question_keywords = set(question.split())
    intersection = context_keywords.intersection(question_keywords)
    return len(intersection) > 0


def get_prompt(context, input):
    """
    Creates the prompt for the LLM, including context variables.
    If the context is relevant, include it in the prompt.
    """
    context_template = ""
    if is_question_relevant_to_context(context, input):
        context_template = f"""
        <context>
        {context}
        </context>
        """

    # flexible prompt to allow the model to use its knowledge if context isn't relevant
    return ChatPromptTemplate.from_template(
        """
        {context_template}
        If the context provides relevant information, use it to answer the following question.
        If not, rely on your general knowledge to answer.
        <context>
        {context}
        </context>

        Question: {input}
        """
    )


prompt = st.text_input("Input your prompt here")

if prompt:
    prompt_template = get_prompt(st.session_state.get("context", ""), prompt)

    # Create the document chain and retrieval
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Process the prompt and display the response
    response = retrieval_chain.invoke(
        {
            "input": prompt,
            "context": st.session_state.get("context", ""),
            "context_template": prompt_template.messages[0].format(
                context_template="",
                context=st.session_state.get("context", ""),
                input=prompt,
            ),
        }
    )
    st.write(response["answer"])
