import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_ollama import ChatOllama

load_dotenv()


class RAGSearch:

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "mxbai-embed-large",
                 llm_model: str = "llama3.2:1b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # Local Ollama setup
        self.llm = ChatOllama(model=llm_model, temperature=0)
        print(f"[INFO] Local Ollama LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."


        prompt = f"""Use the following context to answer the user query concisely.
Query: {query}
Context: {context}
Answer:"""

        response = self.llm.invoke([prompt])
        return response.content