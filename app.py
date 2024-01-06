import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    *streamlit
    *open AI
    *langchain            
    
 
    ''')
    add_vertical_space(5)
    st.write('made by varun m s')
    st.write('varunmsaji01@gmail.com')
 
load_dotenv()
 
def main():
    st.header("Chat with PDF 💬")
    
    pdf = st.file_uploader("upload the pdf",type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings()
        Vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
        
        query = st.text_input("ask the question")

        if query:
        
             docs = Vectorstore.similarity_search(query=query)
             llm = OpenAI(temperature=0)
             chain = load_qa_chain(llm=llm,chain_type='stuff')
             response = chain.run(input_documents=docs,question=query)
             st.write(response)



             

        
 
          


if __name__=='__main__':
    main()