"""
FastAPI RAG System with Mistral 7B (Ollama) + Auto-Summarization
Complete implementation for AI Knowledge Helper assignment
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import tempfile
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import json
import re
from pathlib import Path

# ========== DOCUMENT LOADING ==========
try:
    from unstructured.partition.auto import partition
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("⚠️ Unstructured not available. Install with: pip install unstructured[all-docs]")

# ========== FASTAPI APP ==========
app = FastAPI(title="AI Knowledge Helper", version="1.0.0")

# ========== GLOBAL STATE ==========
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)

# Storage for document chunks and FAISS index
document_store = {
    "chunks": [],
    "index": None,
    "embeddings": None,
    "metadata": {},
    "summary": ""
}

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b"

# ========== REQUEST/RESPONSE MODELS ==========
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class AnswerResponse(BaseModel):
    answer: str
    relevance_score: float
    retrieved_chunks: List[str]
    metadata: dict

class SummaryResponse(BaseModel):
    summary: str
    total_chunks: int
    document_name: str

# ========== HELPER FUNCTIONS ==========

def clean_text(text: str) -> str:
    """Clean extracted text"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.replace('\x0c', '')
    text = re.sub(r'\s+\n', '\n', text)
    return text.strip()

def load_document(file_path: str) -> str:
    """Universal document loader with OCR fallback"""
    if not UNSTRUCTURED_AVAILABLE:
        # Simple fallback for TXT files
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        raise ImportError("Unstructured library required for PDF/image processing")
    
    try:
        elements = partition(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=["eng"],
            skip_infer_table_types=[]
        )
        
        text_parts = []
        for element in elements:
            if element.category == "Table":
                text_parts.append(f"Table:\n{element.text}\n")
            else:
                text_parts.append(element.text)
        
        full_text = "\n\n".join([str(part) for part in text_parts if part])
        
        if len(full_text.strip()) > 50:
            return full_text
    except Exception as e:
        print(f"Unstructured failed: {e}. Trying OCR...")
    
    # OCR fallback
    if file_path.lower().endswith(".pdf"):
        images = convert_from_path(file_path, dpi=300)
    else:
        images = [Image.open(file_path)]
    
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n\n"
    
    return text

def call_ollama(prompt: str, max_tokens: int = 512) -> str:
    """Call Mistral 7B via Ollama API"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            raise Exception(f"Ollama error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure it's running: ollama serve"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

def generate_summary(chunks: List[str], doc_name: str) -> str:
    """Generate document summary using Mistral 7B (Part 3: ML Component)"""
    # Take first few chunks for summary (avoid token limits)
    sample_text = "\n\n".join(chunks[:5])
    
    prompt = f"""Summarize the following document in 3-5 clear sentences. Focus on the main topics and key information.

Document: {doc_name}

Text:
{sample_text[:2000]}

Summary:"""
    
    summary = call_ollama(prompt, max_tokens=200)
    return summary

def evaluate_retrieval(query: str, chunks: List[str], scores: List[float]) -> dict:
    """Evaluate retrieval quality (Part 2: Bonus)"""
    if not chunks:
        return {"avg_score": 0.0, "quality": "No results"}
    
    avg_score = sum(scores[:len(chunks)]) / len(chunks)
    avg_score = max(0.0, min(1.0, avg_score))
    
    # Quality assessment
    if avg_score > 0.7:
        quality = "Excellent"
    elif avg_score > 0.55:
        quality = "Good"
    elif avg_score > 0.4:
        quality = "Moderate"
    else:
        quality = "Low"
    
    return {
        "avg_score": round(avg_score, 3),
        "quality": quality,
        "num_chunks": len(chunks),
        "top_score": round(scores[0], 3) if scores else 0.0
    }

# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "AI Knowledge Helper API",
        "endpoints": {
            "upload": "/upload (POST)",
            "ask": "/ask (POST)",
            "summary": "/summary (GET)",
            "stats": "/stats (GET)"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process document (PDF, TXT, CSV, etc.)
    Part 1: Data Processing & Embeddings
    """
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load and clean document
        print(f"Loading document: {file.filename}")
        raw_text = load_document(tmp_path)
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from document")
        
        cleaned_text = clean_text(raw_text)
        
        # Split into chunks
        chunks = text_splitter.split_text(cleaned_text)
        chunk_list = [{"id": i, "text": c.strip()} for i, c in enumerate(chunks)]
        
        print(f"Created {len(chunk_list)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = embedding_model.encode(
            [c["text"] for c in chunk_list],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Generate document summary (Part 3: ML Component)
        print("Generating summary...")
        summary = generate_summary([c["text"] for c in chunk_list], file.filename)
        
        # Store globally
        document_store["chunks"] = chunk_list
        document_store["index"] = index
        document_store["embeddings"] = embeddings
        document_store["metadata"] = {
            "filename": file.filename,
            "total_chunks": len(chunk_list),
            "embedding_dim": embeddings.shape[1]
        }
        document_store["summary"] = summary
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunk_list),
            "summary": summary,
            "message": "Document processed and ready for questions"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer question using RAG pipeline
    Part 2: Retrieval-Augmented QA with evaluation
    """
    if document_store["index"] is None:
        raise HTTPException(status_code=400, detail="No document uploaded. Upload a document first.")
    
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Step 1: Retrieve candidates
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    D, I = document_store["index"].search(query_emb, request.top_k * 2)
    
    candidate_chunks = []
    scores = []
    seen = set()
    
    for score, idx in sorted(zip(D[0], I[0]), reverse=True):
        if idx not in seen and 0 <= idx < len(document_store["chunks"]):
            candidate_chunks.append(document_store["chunks"][idx]["text"])
            scores.append(float(score))
            seen.add(idx)
        if len(candidate_chunks) >= request.top_k * 2:
            break
    
    # Step 2: Rerank with cross-encoder
    if len(candidate_chunks) > 5:
        pairs = [[query, chunk] for chunk in candidate_chunks]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
        final_chunks = [c for c, _ in ranked[:request.top_k]]
    else:
        final_chunks = candidate_chunks[:request.top_k]
    
    # Step 3: Evaluate retrieval quality (Bonus)
    eval_metrics = evaluate_retrieval(query, final_chunks, scores)
    
    # Step 4: Build context
    context = "\n\n".join(final_chunks)
    
    # Step 5: Generate answer with Mistral 7B via Ollama
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
- Be concise and accurate
- Quote relevant parts from the context when possible
- If the answer is not in the context, say "The information is not available in the document"
- Try to understand the user's intent and answer accordingly

Context:
{context}

Question: {query}

Answer:"""
    
    answer = call_ollama(prompt, max_tokens=300)
    
    return AnswerResponse(
        answer=answer,
        relevance_score=eval_metrics["avg_score"],
        retrieved_chunks=final_chunks,
        metadata={
            "evaluation": eval_metrics,
            "model": MODEL_NAME,
            "chunks_used": len(final_chunks)
        }
    )

@app.get("/summary", response_model=SummaryResponse)
async def get_summary():
    """
    Get document summary (Part 3: ML Component - Auto-Summarization)
    """
    if not document_store["summary"]:
        raise HTTPException(status_code=400, detail="No document uploaded")
    
    return SummaryResponse(
        summary=document_store["summary"],
        total_chunks=document_store["metadata"].get("total_chunks", 0),
        document_name=document_store["metadata"].get("filename", "Unknown")
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if document_store["index"] is None:
        return {"status": "No document loaded"}
    
    return {
        "status": "Document loaded",
        "metadata": document_store["metadata"],
        "model": MODEL_NAME,
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }

@app.delete("/reset")
async def reset_system():
    """Reset document store"""
    document_store["chunks"] = []
    document_store["index"] = None
    document_store["embeddings"] = None
    document_store["metadata"] = {}
    document_store["summary"] = ""
    
    return {"status": "System reset successfully"}

# ========== MAIN ==========
if __name__ == "__main__":
    print("🚀 Starting AI Knowledge Helper API...")
    print("📝 Make sure Ollama is running: ollama serve")
    print(f"🤖 Model: {MODEL_NAME}")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
