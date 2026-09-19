import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks: list[str]):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(chunks, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.35) -> list[tuple[str, float]]:
        query_vec = self.embedder.encode(query, normalize_embeddings=True)
        
        # When vectors are normalized, dot product equals cosine similarity
        scores = np.dot(self.embeddings, query_vec)
        
        # Sort indices by score in descending order
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append((self.chunks[idx], score))
                
        return results