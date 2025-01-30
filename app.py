import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@st.cache_data
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_data
def get_vector_store(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

@st.cache_resource
def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def process_pdfs(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    st.session_state.processed = True
    return "PDFs processed successfully!"

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversation_chain()
    response=chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def main():
    st.title("Chat with multiple PDFs")
    
    tab1, tab2 = st.tabs(["Upload PDFs", "Chat"])
    
    with tab1:
        pdf_docs = st.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                status = process_pdfs(pdf_docs)
                st.success(status)
    
    with tab2:
        if not st.session_state.processed:
            st.warning("Please upload and process PDFs first")
        else:
            user_question = st.text_input("Ask a question from the PDF files")
            if st.button("Submit"):
                with st.spinner("Generating response..."):
                    response = user_input(user_question)
                    st.write(response)

if __name__=="__main__":
    main()

