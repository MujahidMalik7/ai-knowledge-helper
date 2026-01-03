# AI Knowledge Helper - RAG System with Mistral 7B

A Retrieval-Augmented Generation (RAG) system built with FastAPI and Mistral 7B (via Ollama) for intelligent document Q&A.

## 🎯 Features

- **Document Upload**: Process PDFs, TXT, CSV, and image files
- **Smart Q&A**: Answer questions using RAG pipeline with Mistral 7B
- **Auto-Summarization**: Generate document summaries (ML Component)
- **Evaluation Metrics**: Retrieval quality scoring
- **Reranking**: Cross-encoder for improved accuracy

## 📋 Assignment Components

### ✅ Part 1: Data Processing
- Document ingestion with OCR fallback
- Text cleaning and chunking (800 chars, 150 overlap)
- Embeddings with `BAAI/bge-base-en-v1.5`
- FAISS vector database

### ✅ Part 2: Retrieval-Augmented QA
- Query embedding → FAISS search → Cross-encoder reranking
- Mistral 7B answer generation via Ollama
- **Bonus**: Retrieval evaluation metrics

### ✅ Part 3: ML Component (Auto-Summarization)
- Automatic document summarization on upload
- Uses Mistral 7B for high-quality summaries

### ✅ Part 4: API Deployment
- RESTful API with FastAPI
- Endpoints: `/upload`, `/ask`, `/summary`, `/stats`

## 🚀 Setup Instructions

### Prerequisites
1. **Python 3.9+**
2. **Ollama** (for Mistral 7B)
3. **System dependencies** (Linux/Mac):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr poppler-utils
   
   # macOS
   brew install tesseract poppler
   ```

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-knowledge-helper
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and start Ollama**
   ```bash
   # Install Ollama from https://ollama.ai
   
   # Pull Mistral 7B
   ollama pull mistral:7b
   
   # Start Ollama server (in separate terminal)
   ollama serve
   ```

## 🎮 Usage

### Start the API server
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI

### Example Usage

#### 1. Upload a document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "your_document.pdf",
  "chunks_created": 45,
  "summary": "This document discusses...",
  "message": "Document processed and ready for questions"
}
```

#### 2. Ask a question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "relevance_score": 0.82,
  "retrieved_chunks": ["chunk1...", "chunk2..."],
  "metadata": {
    "evaluation": {
      "avg_score": 0.82,
      "quality": "Excellent",
      "num_chunks": 5
    }
  }
}
```

#### 3. Get document summary
```bash
curl -X GET "http://localhost:8000/summary"
```

#### 4. Check system stats
```bash
curl -X GET "http://localhost:8000/stats"
```

## 📊 Evaluation Metrics

The system includes built-in evaluation:

- **Relevance Score**: 0.0 - 1.0 (FAISS similarity)
- **Quality Rating**: Excellent (>0.7) | Good (>0.55) | Moderate (>0.4) | Low
- **Retrieval Stats**: Number of chunks, top score

## 🏗️ Project Structure

```
ai-knowledge-helper/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── test_documents/     # Sample documents for testing
└── report.md          # Assignment report (2-3 pages)
```

## 🔧 Configuration

Edit these variables in `main.py`:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b"

# Chunk settings
chunk_size = 800
chunk_overlap = 150

# Retrieval settings
top_k = 5  # Number of chunks to retrieve
```

## 🧪 Testing

Test with the provided sample documents or your own:

```bash
# Test with a PDF
curl -X POST "http://localhost:8000/upload" -F "file=@sample.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key points"}'
```

## 🐛 Troubleshooting

### Ollama connection error
```
Cannot connect to Ollama. Make sure it's running: ollama serve
```
**Solution**: Start Ollama in a separate terminal: `ollama serve`

### Tesseract not found
```
TesseractNotFoundError
```
**Solution**: Install tesseract-ocr (see Prerequisites)

### Out of memory
**Solution**: Reduce `chunk_size` or use fewer `top_k` chunks

## 📝 Tools & Technologies

- **FastAPI**: REST API framework
- **Mistral 7B**: LLM via Ollama
- **SentenceTransformers**: Embeddings (`BAAI/bge-base-en-v1.5`)
- **FAISS**: Vector similarity search
- **Cross-Encoder**: Reranking (`ms-marco-MiniLM-L-6-v2`)
- **Unstructured**: Document parsing
- **LangChain**: Text splitting

## 🎥 Demo Video

[Link to 60-120 second Loom demo]

## 📄 License

MIT License

## 👤 Author

[Your Name]

## 📧 Contact

For questions or issues, please open a GitHub issue.
