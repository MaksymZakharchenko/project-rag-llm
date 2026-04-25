import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

st.set_page_config(page_title="AI Document Assistant")

st.title("📄 AI Document Intelligence System")

# ----------------------------
# 1. Upload PDF
# ----------------------------
uploaded_file = st.file_uploader("📤 Wgraj PDF", type=["pdf"])

# ----------------------------
# 2. Session state (żeby nie budować za każdym kliknięciem)
# ----------------------------
if "qa_ready" not in st.session_state:
    st.session_state.qa_ready = False

if uploaded_file and not st.session_state.qa_ready:

    with st.spinner("🔄 Przetwarzam PDF..."):

        # zapis tymczasowy pliku
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # 3. Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # 4. Vector DB
        db = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        retriever = db.as_retriever(search_kwargs={"k": 3})

        # 5. LLM
        llm = Ollama(model="llama3")

        # 6. Prompt
        prompt = ChatPromptTemplate.from_template("""
        Answer ONLY using the context below.

        Context:
        {context}

        Question:
        {question}
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def qa_chain(query):
            docs = retriever.invoke(query)
            context = format_docs(docs)

            chain = prompt | llm | StrOutputParser()

            result = chain.invoke({
                "context": context,
                "question": query
            })

            return result, docs

        st.session_state.qa_chain = qa_chain
        st.session_state.qa_ready = True

        os.remove(pdf_path)

        st.success("✅ PDF gotowy! Możesz zadawać pytania.")

# ----------------------------
# 3. Chat / pytania
# ----------------------------
if st.session_state.qa_ready:

    query = st.text_input("❓ Zadaj pytanie do dokumentu")

    if st.button("Szukaj"):
        if query:
            result, docs = st.session_state.qa_chain(query)

            st.write("### 🧠 Odpowiedź")
            st.write(result)

            st.write("### 📚 Źródła")
            for d in docs:
                st.write(d.page_content[:300])