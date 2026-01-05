import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGComplaintAnalyzer:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        
    def create_embeddings(self, complaints):
        embeddings = self.embedder.encode(complaints)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        return embeddings
    
    def search_similar(self, query, k=3):
        query_embed = self.embedder.encode([query])
        distances, indices = self.index.search(query_embed, k)
        return indices[0]
