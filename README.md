# DocuQuery RAG Engine

A lightweight, framework-free Retrieval-Augmented Generation (RAG) engine built from first principles. This project demonstrates how to build a production-ready AI pipeline by cleanly separating a custom Python retrieval backend from a responsive frontend.

## Architecture

This system intentionally avoids heavy orchestration frameworks (like LangChain or LlamaIndex) to maintain absolute control over token management, chunking boundaries, and vector math.

*   **Backend (`api.py` & `src/`):** A FastAPI server that handles document ingestion, custom sliding-window chunking, and streaming API endpoints.
*   **Vector Engine:** Local `sentence-transformers` embeddings combined with NumPy for explicit cosine similarity calculations. 
*   **LLM Inference:** Groq API for ultra-low latency text generation.
*   **Frontend (`app.py`):** A Streamlit chat interface that consumes the FastAPI streaming endpoints.

## Features

*   **Sentence-Aware Chunking:** Preserves semantic boundaries with sliding-window overlap to prevent context loss.
*   **Score Thresholding:** Implements strict similarity thresholds to reject irrelevant context before it reaches the LLM, eliminating hallucinations for out-of-scope queries.
*   **Asynchronous Streaming:** Token-by-token generation for a highly responsive user experience.
*   **Decoupled Architecture:** Frontend and backend run independently, allowing easy migration to a React/Next.js web client in the future.

##  Prerequisites

*   Python 3.10+
*   A valid [Groq API Key](https://console.groq.com/)

## ⚙️ Local Setup

1. **Clone the repository**
```bash
   git clone [https://github.com/YourUsername/docuquery-rag.git](https://github.com/YourUsername/docuquery-rag.git)
   cd docuquery-rag
```

Set up a virtual environment

python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

Install dependencies

pip install fastapi uvicorn streamlit groq sentence-transformers numpy requests

Configure Environment Variables
Create a .env file in the root directory and securely store your API key:

Plaintext
GROQ_API_KEY=your_actual_api_key_here

Running the Application
This is a full-stack system requiring both the API and the UI to run simultaneously.

1. Start the API Engine (Terminal 1)
uvicorn api:app --reload
The API will boot and expose endpoints at http://127.0.0.1:8000

2. Start the Frontend UI (Terminal 2)
streamlit run app.py
   
