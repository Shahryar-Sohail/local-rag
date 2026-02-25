import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import embedding


class EmbeddingManager:
    def __init__(self,model_name: str="mxbai-embed-large"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading model: {self.model_name}")
            self.model = OllamaEmbeddings(model=self.model_name)
            # print(f"Model Loaded Successfully. Embedding Dimension:{self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error Loading Model {self.model_name}:{e}")
            raise

    def generate_embedding(self,texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model Not Loaded")
        print(f"Generating embedding for {len(texts)} texts...")
        embeddings = self.model.embed_documents(texts)
        embeddings_np = np.array(embeddings)
        print(f"Generated Embedding with shape {embeddings_np.shape}")
        return embeddings_np


embedding_manager = EmbeddingManager()
print(embedding_manager)

