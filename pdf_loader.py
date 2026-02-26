import numpy as np
from langchain_core import documents
from langchain_ollama import OllamaEmbeddings
import os
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any,Tuple
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import embedding


def process_all_pdf(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob('**/*.pdf'))

    print(f"Found{len(pdf_files)} PDF files")

    for pdf in pdf_files:
        print(f"\nProcessing: {pdf.name}")
        try:
            loader = PyPDFLoader(str(pdf))
            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = pdf.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"✔ Loaded: {len(documents)} pages")

        except Exception as e:
            print(f"Error {e}")

    print(f"\n Total Documents Loaded: {len(all_documents)}")
    return all_documents

all_pdf_documents = process_all_pdf("./pdf")


def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",","]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"\nSplit Documents: {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\n Example Chunk")
        # print(f"Content: {split_docs[0].page_content[:200]}...")
        # print(f"Meta Data {split_docs[0].metadata}")

    return split_docs



chunks = split_documents(all_pdf_documents)

class EmbeddingManager:
    def __init__(self,model_name: str="mxbai-embed-large"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading model: {self.model_name}")
            self.model = OllamaEmbeddings(model=self.model_name)
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
# print(embedding_manager)






#Vector Store code here
class VectorStore:
    def __init__(self,collection_name: str="pdf_documents", persist_directory: str="../data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            #Create Persistent Chroma Client
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #Get or Create Collection
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={"description": "PDF documents embedding for RAG"}
            )
            print(f"Initialized Store for Collection {self.collection_name}")
            print(f"Existing Documents in Collection {self.collection.count()}")

        except Exception as e:
            print(f"Error Initializing Store {e}")
            raise

    def add_documents(self,documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings do not match")

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_lenght'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        #add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"Added {len(documents)} documents to vector store")
            print(f"Total no of documents in collection {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store {e}")
            raise

vectorstore = VectorStore()
# print(vectorstore)



#Text to Embeddings
texts = [doc.page_content for doc in chunks]
# print(texts)


#Generate Embeddings
embeddings = embedding_manager.generate_embedding(texts)

#Store in Vector Data Base
vectorstore.add_documents(chunks, embeddings)