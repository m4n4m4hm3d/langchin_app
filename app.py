import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()
google_Api = os.getenv('GOOGLE_API_KEY')
Pinecone_api = os.getenv('PINECONE_API_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=1)
embaddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docsearch = PineconeVectorStore.from_existing_index(
    index_name='deeplearning',
    embedding=embaddings,
)

system_prompt = (
    '''you are a assistent for question answering.
    You have to answer the question based on the given context.
    and if you don't have the answer then you can ask the user to provide the answer politely.
    use 5 sentences to answer the question. write the python code properly.{context}'''
)

prompt = ChatPromptTemplate.from_messages(
    [
       ('system', system_prompt),
       ('human', '{input}')
    ]
)
retriver = docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 5})

qus_ans_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriver, qus_ans_chain)



import streamlit as st

st.image("8gflcp.jpg", width=100)
st.title('DEEP LEARNING WITH PYTHON')

user_input = st.text_input("Ask anything about Deep learning")

if st.button("Submit"):
    if user_input:
        with st.spinner("Processing your question..."):  # Show a spinner while processing
            try:
                # Invoke the RAG chain
                response = rag_chain.invoke({"input": user_input})
                st.write("**Answer:**")
                st.write(response['answer'])  # Display the answer
            except Exception as e:
                st.error(f"An error occurred: {e}")  # Display error message
    else:
        st.warning("Please enter a question.")
