# 🚀 Quick Start Guide - AI Knowledge Helper

Get your RAG system running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] Ollama installed (from https://ollama.ai)
- [ ] Mistral 7B downloaded (`ollama pull mistral:7b`)

## Step-by-Step Setup

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Download Poppler: https://blog.alivate.com.au/poppler-windows/

### 2. Create Project

```bash
# Create project folder
mkdir ai-knowledge-helper
cd ai-knowledge-helper

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Python Packages

Save the `requirements.txt` and run:
```bash
pip install -r requirements.txt
```

This will take 2-5 minutes to download all dependencies.

### 4. Start Ollama

**In a separate terminal:**
```bash
# Make sure Mistral is downloaded
ollama pull mistral:7b

# Start Ollama server
ollama serve
```

Leave this terminal running!

### 5. Start FastAPI Server

**In your main terminal:**
```bash
python main.py
```

You should see:
```
🚀 Starting AI Knowledge Helper API...
📝 Make sure Ollama is running: ollama serve
🤖 Model: mistral:7b
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6. Test the API

**Open your browser:**
- Visit: http://localhost:8000/docs
- You'll see interactive API documentation (Swagger UI)

**Or use the test script:**
```bash
python test_api.py sample.txt
```

## 🎯 First API Calls

### Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"
```

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

### Get Summary

```bash
curl -X GET "http://localhost:8000/summary"
```

## 📝 Using the Interactive Docs

1. Go to http://localhost:8000/docs
2. Click on any endpoint (e.g., `/upload`)
3. Click "Try it out"
4. Upload a file or enter parameters
5. Click "Execute"
6. See the response!

## 🐛 Troubleshooting

### "Cannot connect to Ollama"
**Fix:** Start Ollama in separate terminal: `ollama serve`

### "TesseractNotFoundError"
**Fix:** Install tesseract-ocr (see step 1)

### "ModuleNotFoundError"
**Fix:** Make sure virtual environment is activated and requirements installed

### "Out of memory"
**Fix:** Reduce chunk size in `main.py`:
```python
chunk_size=500  # Instead of 800
```

### Slow responses (>10 seconds)
**Fix:** This is normal for first query (model loading). Subsequent queries will be faster.

## 📊 Expected Performance

- **Upload (50-page PDF)**: 5-10 seconds
- **First question**: 10-15 seconds (model loading)
- **Subsequent questions**: 2-5 seconds
- **Memory usage**: ~2GB RAM

## 🎥 Recording Your Demo

For the assignment, record a 60-120 second Loom video showing:

1. **Upload** a document (show the response with summary)
2. **Ask 2-3 questions** (show the answers and relevance scores)
3. **Get summary** (show the auto-generated summary)
4. **Show stats** (optional, shows system info)

**Tips:**
- Keep it concise and clear
- Highlight the key features (RAG, evaluation, summarization)
- Show the relevance scores working
- Mention the tech stack (Mistral 7B, FAISS, etc.)

## 🚀 Next Steps

Once everything works:

1. ✅ Test with your own documents
2. ✅ Try different question types
3. ✅ Check the evaluation scores
4. ✅ Review the code and understand the pipeline
5. ✅ Complete the report (use REPORT.md template)
6. ✅ Record your demo video
7. ✅ Push to GitHub

## 📚 Useful Commands

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# List running models
ollama list

# Stop the FastAPI server
Ctrl+C

# Deactivate virtual environment
deactivate
```

## 🎓 Understanding the Flow

```
User uploads PDF
    ↓
Text extracted + chunked → embeddings → FAISS index
    ↓
Summary generated (Part 3: ML Component)
    ↓
User asks question
    ↓
Query embedded → FAISS search → Rerank → Top chunks
    ↓
Chunks + question → Mistral 7B → Answer
    ↓
Return answer + relevance score (evaluation)
```

## 💡 Pro Tips

1. **Use small documents first** (1-10 pages) to test quickly
2. **Ask specific questions** for better retrieval scores
3. **Check the relevance scores** - if < 0.4, document may not have the answer
4. **Use the /docs page** - it's easier than curl commands
5. **Keep Ollama terminal visible** - you'll see model activity

## ✅ Checklist Before Submission

- [ ] All endpoints working (/upload, /ask, /summary, /stats)
- [ ] Test script passes all tests
- [ ] Report completed (2-3 pages)
- [ ] Demo video recorded (60-120 seconds)
- [ ] GitHub repo created with all files
- [ ] README.md has setup instructions
- [ ] requirements.txt is complete
- [ ] Code is clean and commented

## 🆘 Need Help?

- Check the full README.md for detailed documentation
- Review REPORT.md for technical details
- Visit http://localhost:8000/docs for API reference
- Check Ollama docs: https://ollama.ai/docs

---

**You're ready!** Start with `python main.py` and visit http://localhost:8000/docs 🎉
