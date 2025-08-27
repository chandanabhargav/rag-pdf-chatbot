# 📘 RAG PDF Q&A App  

A simple Retrieval-Augmented Generation (RAG) app to query PDFs conversationally.  

👉 **[Live Demo](https://rag-pdf-chatbot-6n3eeyic4qwsdxt32jcfq9.streamlit.app/)**  

### Features
- Upload PDFs and ask natural language questions
- Uses LangChain + FAISS for semantic search
- GPT-4 answers grounded in retrieved document passages
- Streamlit UI for user-friendly interaction  

### Tech Stack
Python · LangChain · FAISS · HuggingFace · OpenAI API · Streamlit  

### Example
```bash
Ask: "Summarize the contract obligations in section 3"
Answer: "The vendor must provide support for 12 months..."