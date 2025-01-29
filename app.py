import gradio as gr
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

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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
    with gr.Blocks() as demo:
        gr.Markdown("# Chat with multiple PDFs")
        
        with gr.Tab("Upload PDFs"):
            pdf_input = gr.File(file_count="multiple", label="Upload your PDF files")
            process_btn = gr.Button("Process")
            output = gr.Textbox(label="Status")
            process_btn.click(fn=process_pdfs, inputs=pdf_input, outputs=output)
            
        with gr.Tab("Chat"):
            question = gr.Textbox(label="Ask a question from the PDF files")
            submit_btn = gr.Button("Submit")
            answer = gr.Textbox(label="Reply")
            submit_btn.click(fn=user_input, inputs=question, outputs=answer)
    
    demo.launch(share=True)

if __name__=="__main__":
    main()
