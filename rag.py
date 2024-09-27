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
OPENAI_API_KEY = "sk-NdLCwBkLroNbbHEpVzZApYdZ96qxNncD9g5YLb-x7QT3BlbkFJv3mz08aOXVXbBHsw_8z8Cc8sNfMdiM0Cg_qcokndQA"  # Replace with your actual API key
NVIDIA_PDF_PATH = "Groundwater.pdf"
VECTOR_DB_DIRECTORY = "OneDrive/Genai workshop/vector"
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIRECTORY, "faiss_index.index")
GPT_MODEL_NAME = 'gpt-4'
CHUNK_SIZE = 700
CHUNK_OVERLAP = 50

@st.cache_resource(show_spinner=False)
def load_and_split_document(pdf_path):
    """Loads and splits the document into pages."""
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

@st.cache_resource(show_spinner=False)
def split_text_into_chunks(pages, chunk_size, chunk_overlap):
    """Splits text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

@st.cache_resource(show_spinner=False)
def create_embeddings(api_key):
    """Creates embeddings from text."""
    return OpenAIEmbeddings(openai_api_key=api_key)

@st.cache_resource(show_spinner=False)
def setup_vector_database(documents, embeddings):
    """Sets up a FAISS vector database for storing embeddings."""
    faiss_index = FAISS.from_documents(documents=documents, embedding=embeddings)
    return faiss_index

@st.cache_resource(show_spinner=False)
def initialize_chat_model(api_key, model_name):
    """Initializes the chat model with specified AI model."""
    return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.0)

@st.cache_resource(show_spinner=False)
def create_retrieval_qa_chain(chat_model, vector_database):
    """Creates a retrieval QA chain combining model and database."""
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    return ConversationalRetrievalChain.from_llm(chat_model, retriever=vector_database.as_retriever(), memory=memory)

def ask_question_and_get_answer(qa_chain, question):
    """Asks a question and retrieves the answer."""
    return qa_chain({"question": question})['answer']

# Streamlit App
def main():
    st.title("RAG-based Question Answering System")
    st.write("Upload a PDF, ask questions, and get responses based on the document.")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        # Process PDF file
        with open(NVIDIA_PDF_PATH, "wb") as f:
            f.write(uploaded_file.read())
        
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
