import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import requests
import json

class EmbeddingManager:
    def __init__(self, use_openai=False):
        self.use_openai = use_openai
        
        if use_openai:
            # OpenAI embeddings
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model = "text-embedding-3-small"
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        else:
            # HuggingFace embeddings (default)
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        if not text or not text.strip():
            return [0.0] * 384  # Return zero vector for empty text
            
        if self.use_openai:
            # OpenAI embedding
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        else:
            # HuggingFace embedding
            return self.model.encode(text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts at once"""
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            return []
            
        if self.use_openai:
            # OpenAI batch embedding
            response = self.client.embeddings.create(
                input=valid_texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        else:
            # HuggingFace batch embedding
            return self.model.encode(valid_texts).tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.use_openai:
            return 1536  # OpenAI text-embedding-3-small dimension
        else:
            return 384   # all-MiniLM-L6-v2 dimension