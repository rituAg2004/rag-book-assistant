import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant", layout="centered")

st.title("📚 RAG Book Assistant")
st.write("Upload your book PDF, build the vector store, and ask questions.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Document Upload & Vector Database Creation ---
uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    st.info(f"Loaded: **{uploaded_file.name}**")

    if st.button("Create / Rebuild Vector Database"):
        with st.spinner("Processing PDF, generating chunks, and creating embeddings..."):
            # Write uploaded buffer to a temporary PDF file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)
                chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

                embedding_model = MistralAIEmbeddings(model="mistral-embed")

                # Build Chroma vectorstore
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory="chroma_db"
                )

                st.success(f"Vector database created successfully with {len(chunks)} chunks!")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- Chat & Querying Section ---
if os.path.exists("chroma_db"):
    st.divider()
    st.subheader("Ask Questions From the Document")

    # Load persistent vectorstore and retriever
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatCohere(
        model="command-r-08-2024",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant.\n"
            "Use ONLY the provided context to answer the question.\n"
            "If the answer is not present in the context, say: \"I could not find the answer in the document.\""
        ),
        (
            "human",
            "Context:{context}\nQuestion:{question}"
        )
    ])

    # Display past conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user query
    if query := st.chat_input("Ask a question about the book..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                response = llm.invoke(final_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})