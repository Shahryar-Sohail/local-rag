from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Switched to Ollama
from src.data_loader import load_all_documents
import numpy as np

class EmbeddingPipeline:
    def __init__(self, model_name: str = "mxbai-embed-large", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_model = OllamaEmbeddings(model=model_name)
        print(f"[INFO] Connected to Ollama embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> List[float]:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating Ollama embeddings for {len(texts)} chunks...")

        # embed_documents is the LangChain standard method
        embeddings = self.embeddings_model.embed_documents(texts)

        print(f"[INFO] Generated {len(embeddings)} embedding vectors.")
        return embeddings


if __name__ == "__main__":

    from langchain_core.documents import Document

    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)

    if embeddings:
        print(f"[DEBUG] First vector size: {len(embeddings[0])}")  # mxbai-embed-large is 1024 dims