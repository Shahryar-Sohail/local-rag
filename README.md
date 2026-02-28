# 🤖 Local RAG Assistant: AI PC Recommender 🚀

This is an **Advanced Retrieval-Augmented Generation (RAG)** system built locally to chat with private documents. Inspired by the Krish Naik series, this version is optimized to run 100% locally on your machine using **Ollama**, saving costs and ensuring data privacy.



---

### 🛠️ Tech Stack
* **Framework:** LangChain 🦜🔗
* **LLM:** Llama 3.2:1b (Local via Ollama) 🦙
* **Embeddings:** mxbai-embed-large (1024-dim) 🔢
* **Vector Store:** FAISS (Facebook AI Similarity Search) ⚡
* **Database:** Pickle (Metadata storage) 💾
* **Environment:** Python 3.10+ 🐍

---

### 📥 Installation & Setup

1. **Clone the Repository** 📂
```bash
git clone https://github.com/Shahryar-Sohail/local-rag/
cd local-rag
```
2. **Create & Activate Virtual Environment** 🍦
```bash
python -m venv .venv
```
# On Windows:
```bash
.venv\Scripts\activate
```
3. **Install Dependencies** 📦
```bash
pip install -r requirements.txt
```

4. **Setup Local Models (Ollama)** 📥
```bash
ollama pull llama3.2:1b
ollama pull mxbai-embed-large
```

5.🚀 **Running the Project**
To test the backend pipeline and see the AI in action:
```bash
python app.py
```

# ⚙️ How It WorksIngestion: 
### 1-PDFs and text files are loaded from the data/ directory.
### 2-Chunking: Documents are split into manageable pieces using RecursiveCharacterTextSplitter.
### 3-Embedding: Each chunk is converted into a 1024-dimensional vector using the mxbai-embed-large model.
### 4-Indexing: Vectors are stored in a FAISS index for high-speed similarity search.
### 5-Retrieval: When a query is made, the system finds the top-$k$ most relevant chunks.
### 6-Generation: Llama 3.2 uses the retrieved context to generate a concise, factual summary.