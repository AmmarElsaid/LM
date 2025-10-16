from sentence_transformers import SentenceTransformer
import openai
import numpy as np

class EmbeddingManager:
    def __init__(self, use_openai=False, embedding_dim=384):
        
        self.use_openai = use_openai
        self.embedding_dim = embedding_dim

        if not use_openai:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text):
        text = str(text).strip()
        if not text:
            # return zero vector if input is empty
            return [0.0] * self.embedding_dim

        if self.use_openai:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response['data'][0]['embedding']
        else:
            return self.model.encode([text])[0].tolist()

    def embed_texts(self, texts):
        texts = [str(t).strip() if t else "" for t in texts]
        embeddings = []
        for t in texts:
            emb = self.embed_text(t)
            embeddings.append(emb)
        return embeddings