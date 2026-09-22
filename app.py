import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

st.title("📚 PDF Question Answering System")
st.write("Upload a PDF and ask questions about it.")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    # Save uploaded PDF
    pdf_path = "uploaded.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.success(f"PDF loaded: {len(documents)} pages")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    st.write(f"Created {len(chunks)} text chunks.")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in FAISS
    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Create retriever
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    # Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    # Question
    question = st.text_input("Ask a question about the PDF:")

    if question:

        # Retrieve relevant chunks
        relevant_docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
        )

        # Prompt
        prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context,
say "I could not find the answer in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        # Show sources
        with st.expander("📖 Sources"):
            for doc in relevant_docs:
                st.write(
                    f"Page: {doc.metadata.get('page', 'Unknown') + 1}"
                )
                st.write(doc.page_content[:500])