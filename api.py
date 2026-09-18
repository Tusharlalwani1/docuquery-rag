import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv


from src.chunker import load_documents_from_folder, create_sentence_chunks
from src.vector_store import VectorStore
from src.generator import RAGGenerator


load_dotenv()

app = FastAPI(title="DocuQuery Core API")


store = VectorStore()
generator = RAGGenerator()

@app.on_event("startup")
def startup_event():
    """Runs once when the server starts to ingest documents."""
    docs_folder = "./data"
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        with open(os.path.join(docs_folder, "sample.txt"), "w") as f:
            f.write("FastAPI is a modern web framework. Retrieval-Augmented Generation prevents AI hallucinations.")
            
    print("Ingesting data for the API...")
    text = load_documents_from_folder(docs_folder)
    chunks = create_sentence_chunks(text, chunk_size=3, overlap=1)
    
    store.build_index(chunks)
    print(f"API Ready! {len(chunks)} chunks loaded into vector memory.")


class QueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    """Takes a JSON query, retrieves context, and streams the LLM response."""
    

    results = store.retrieve(request.query, top_k=3, threshold=0.30)
    retrieved_texts = [chunk for chunk, score in results]
    
    return StreamingResponse(
        generator.stream_answer(request.query, retrieved_texts), 
        media_type="text/event-stream"
    )