# 📚 rag-book-assistant

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload PDF books or documents, dynamically chunk and embed their contents into a vector database, and perform context-grounded conversational Q&A.

Powered by **LangChain**, **Mistral AI Embeddings**, **ChromaDB**, **Cohere Command-R**, and **Streamlit**.

---

## 🌟 Key Features

- **Dynamic PDF Ingestion:** Upload any book or document in `.pdf` format directly via the interactive web UI.
- **Intelligent Chunking:** Employs LangChain's `RecursiveCharacterTextSplitter` with configurable sizes and overlaps to maintain semantic continuity.
- **High-Performance Vector Search:** Leverages `MistralAIEmbeddings` (`mistral-embed`) combined with **Maximal Marginal Relevance (MMR)** in **ChromaDB** to ensure diverse, high-relevance context retrieval.
- **Strict Context-Grounded Responses:** Powered by `ChatCohere` (`command-r-08-2024`) with strict anti-hallucination prompting (strictly relies on retrieved excerpts).
- **Interactive Chat Interface:** Session-managed conversational UI with full message history powered by Streamlit.

---

## 🏗️ Architecture & Pipeline

```text
[ Upload PDF ]
      │
      ▼
[ PyPDFLoader ] ──► Extract text content & metadata
      │
      ▼
[ RecursiveCharacterSplitter ] ──► Chunk size: 1000 | Overlap: 200
      │
      ▼
[ Mistral AI Embeddings ] ──► Generate dense vector representations
      │
      ▼
[ ChromaDB Storage ] ──► Persisted vector store in `chroma_db/`
      │
      ▼
[ User Question ] ──► MMR Vector Search (k=4, fetch_k=10, λ=0.5)
      │
      ▼
[ Cohere Command-R LLM ] ──► Context-bounded answer generation
      │
      ▼
[ Streamlit UI Output ]
```

---

## 📁 Project Structure

```text
rag-book-assistant/
├── app.py                 # Streamlit UI & unified RAG pipeline
├── main.py                # Terminal-based interactive query interface
├── create_database.py     # Standalone batch ingestion script
├── .env                   # Environment variables and private API keys
├── .env.example           # Example environment template
├── .gitignore             # Git ignore patterns for keys, DB, and cache
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+ (tested on Python 3.11 / 3.12 / 3.13)
- Active API keys for:
  - [Mistral AI Console](https://console.mistral.ai/)
  - [Cohere Dashboard](https://dashboard.cohere.com/)

---

### 2. Environment Setup

Clone the repository and navigate into the project directory:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/rag-book-assistant.git
cd rag-book-assistant
```

Create and activate a virtual environment:

**PowerShell (Windows):**
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Bash (Linux/macOS):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

Install required libraries using `uv` (recommended for speed) or `pip`:

```bash
# Using uv
uv pip install streamlit langchain langchain-community langchain-chroma langchain-mistralai langchain-cohere pypdf python-dotenv

# Or using standard pip
pip install streamlit langchain langchain-community langchain-chroma langchain-mistralai langchain-cohere pypdf python-dotenv
```

---

### 4. Configure Environment Variables

Create a `.env` file in the root directory by copying the example template:

```bash
cp .env.example .env
```

Open `.env` and fill in your API credentials:

```ini
MISTRAL_API_KEY="your_mistral_api_key_here"
COHERE_API_KEY="your_cohere_api_key_here"
```

---

## 💻 Running the Application

### Option A: Web Interface (Streamlit) — Recommended

Launch the browser-based assistant:

```bash
streamlit run app.py
```

1. Open `http://localhost:8501` in your web browser.
2. Use the file uploader to provide any PDF book or paper.
3. Click **Create / Rebuild Vector Database** to trigger chunking and vector indexing.
4. Submit queries through the chat input at the bottom.

---

### Option B: Command Line Interface (CLI)

If you prefer running standalone ingestion and terminal querying:

1. Place your target PDF inside your workspace (e.g., `document loaders/deeplearning.pdf`).
2. Generate the vector store:
   ```bash
   python create_database.py
   ```
3. Run the interactive terminal loop:
   ```bash
   python main.py
   ```

---

## ⚙️ Retrieval & Chunking Configuration

You can adjust parameters inside `app.py` or `create_database.py`:

- **Text Splitter (`RecursiveCharacterTextSplitter`):**
  - `chunk_size`: Maximum characters per segment (default: `1000`).
  - `chunk_overlap`: Overlapping characters between adjacent segments to preserve context boundaries (default: `200`).
- **Vector Search (`vectorstore.as_retriever`):**
  - `search_type="mmr"`: Maximal Marginal Relevance balances factual relevance with informational diversity.
  - `k`: Number of final chunks passed into the prompt context (default: `4`).
  - `fetch_k`: Candidate pool fetched before diversity filtering (default: `10`).
  - `lambda_mult`: Controls balance between relevance (`1.0`) and diversity (`0.0`) (default: `0.5`).

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
