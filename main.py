from dotenv import load_dotenv
load_dotenv()

import os
from src.chunker import load_documents_from_folder, create_sentence_chunks
from src.vector_store import VectorStore
from src.generator import RAGGenerator

def run():
    docs_folder = "./data"
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        with open(os.path.join(docs_folder, "sample.txt"), "w") as f:
            f.write("Full-stack AI systems integrate language models with custom document stores.")

    print("Ingesting documents and calculating embeddings...")
    text = load_documents_from_folder(docs_folder)
    chunks = create_sentence_chunks(text, chunk_size=3, overlap=1)
    
    store = VectorStore()
    store.build_index(chunks)
    generator = RAGGenerator()

    print(f"System ready! Ingested {len(chunks)} contextual chunks.\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break

        results = store.retrieve(query, top_k=3, threshold=0.30)
        retrieved_texts = [chunk for chunk, score in results]

        print("\n[Matches Found]")
        for chunk, score in results:
            print(f" - (Score: {score:.2f}) {chunk[:80]}...")

        print("\nAnswer: ", end="", flush=True)
        for token in generator.stream_answer(query, retrieved_texts):
            print(token, end="", flush=True)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run()