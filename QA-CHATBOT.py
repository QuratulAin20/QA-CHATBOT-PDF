import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader      

from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ-KEY")
os.environ['HF_API_KEY'] = os.getenv('HF-TOKEN')

groq_api_key = os.getenv("GROQ-KEY")
hf_api_key = os.getenv("HF-TOKEN")

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
Answer the based upon the provided context only.
please provide the most accurate response based upon the question 
<context>
{context}
<context>
Question:{input}
"""

)
def create_vector_embedding():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = HuggingFaceEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("data")
            st.session_state.docs = st.session_state.loader.load()
            
            if not st.session_state.docs:
                st.error("No documents loaded. Please check the PDF file.")
                return
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
        except Exception as e:
            st.error(f"An error occurred: {e}")

user_prompt= st.text_input("Enter your query")
if st.button ("document embeddings"):
    create_vector_embedding()
    st.write("vector database is ready")


if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({'input':user_prompt}) 


    st.write(response['answer'])

    with st.expander('Document Similarity search'):
        for i,doc in enumerate (response['context']):
            st.write(doc.page_content)
            st.write("----------------")






