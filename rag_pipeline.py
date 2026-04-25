from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_qa_chain():
    # Embeddingi
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Baza
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = Ollama(model="llama3")

    # Prompt (bardzo ważne dla RAG!)
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {question}
    """)

    # RAG pipeline (LCEL)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)

        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "context": context,
            "question": query
        })

        return result, docs

    return run