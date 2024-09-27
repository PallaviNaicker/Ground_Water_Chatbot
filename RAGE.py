import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# File paths and API keys
OPENAI_API_KEY = "sk-VOCVMgy97FPU9OfWknSvuVR-GgqdNhc8PQhB4tvgzaT3BlbkFJ2u3xdi9-HDyS4nkxO2ytXiQpcCFpXofc8XTpsJ8JcA"  # Replace with your actual API key
NVIDIA_PDF_PATH = r"C:\Users\Pallavi\OneDrive\Desktop\FDP\Groundwater.pdf"
VECTOR_DB_DIRECTORY = r"C:\Users\Pallavi\OneDrive\Desktop\FDP\vector"
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIRECTORY, "faiss_index.index")
GPT_MODEL_NAME = 'gpt-4'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

@st.cache_resource(show_spinner=False)
def load_and_split_document(pdf_path):
    """Loads and splits the document into pages."""
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

@st.cache_resource(show_spinner=False)
def split_text_into_chunks(_pages, chunk_size, chunk_overlap):
    """Splits text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(_pages)

@st.cache_resource(show_spinner=False)
def create_embeddings(_api_key):
    """Creates embeddings from text."""
    return OpenAIEmbeddings(openai_api_key=_api_key)

@st.cache_resource(show_spinner=False)
def setup_vector_database(_documents, _embeddings):
    """Sets up a FAISS vector database for storing embeddings."""
    faiss_index = FAISS.from_documents(documents=_documents, embedding=_embeddings)
    return faiss_index

@st.cache_resource(show_spinner=False)
def initialize_chat_model(_api_key, _model_name):
    """Initializes the chat model with specified AI model."""
    return ChatOpenAI(openai_api_key=_api_key, model_name=_model_name, temperature=0.0)

@st.cache_resource(show_spinner=False)
def create_retrieval_qa_chain(_chat_model, _vector_database):
    """Creates a retrieval QA chain combining model and database."""
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    return ConversationalRetrievalChain.from_llm(_chat_model, retriever=_vector_database.as_retriever(), memory=memory)

@st.cache_data
def ask_question_and_get_answer(_qa_chain, _question):
    """Asks a question and retrieves the answer."""
    return _qa_chain({"question": _question})['answer']

# Streamlit App
def main():
    st.title("RAG-based Question Answering System")
    st.write("Ask questions and get responses based on the document.")

    # Process the preloaded PDF file
    pages = load_and_split_document(NVIDIA_PDF_PATH)
    documents = split_text_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = create_embeddings(OPENAI_API_KEY)
    vector_database = setup_vector_database(documents, embeddings)
    chat_model = initialize_chat_model(OPENAI_API_KEY, GPT_MODEL_NAME)
    qa_chain = create_retrieval_qa_chain(chat_model, vector_database)

    st.success("PDF Loaded and Processed!")

    # Question Input
    question = st.text_input("Ask a question:")
    
    if question:
        # Get answer
        answer = ask_question_and_get_answer(qa_chain, question)
        st.write(f"**Question:** {question}")
        st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
