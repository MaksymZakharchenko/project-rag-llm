# AI Document Intelligence System (Local RAG with LLM)

## Overview

This project is a **local Retrieval-Augmented Generation (RAG) system** that allows users to upload PDF documents and ask questions about their content in natural language.

The system combines **semantic search (vector embeddings)** with a **local Large Language Model (LLM)** to generate answers strictly based on the provided documents.

Everything runs locally — no external APIs are required.

---

## Key Features

- Upload PDF documents via web interface
- Semantic search using vector embeddings
- Local LLM inference (no cloud API required)
- Retrieval-Augmented Generation (RAG pipeline)
- Answers grounded in document context
- Fast interactive UI using Streamlit

---

## Tech Stack

- Ollama – local LLM runtime (Llama 3)
- LangChain – orchestration framework for RAG
- ChromaDB – vector database for embeddings
- Streamlit – web interface
- HuggingFace Embeddings – text vectorization
- PyPDF – PDF parsing and extraction

---

## Architecture

```text
PDF Upload
   ↓
Text Extraction (PyPDF)
   ↓
Chunking (RecursiveCharacterTextSplitter)
   ↓
Embedding Generation (HuggingFace)
   ↓
Vector Storage (ChromaDB)
   ↓
Semantic Retrieval
   ↓
LLM (Llama 3 via Ollama)
   ↓
Final Answer + Source Context
