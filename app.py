from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader 
import tempfile, os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("PDF Q&A")
st.write("Upload a PDF")

pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])

if pdf_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

        print(pdf_path)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter()
        docs = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
        vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4o-mini",temperature=0),retriever=retriever)

        st.subheader("Ask a question")
        query = st.text_input("Your question:")

        if query:
            with st.spinner("Searching"):
                answer = qa_chain.run(query)
                st.write("**Answer**", answer)