# 📄 RAG Chatbot using Ollama Mistral7B

https://github.com/user-attachments/assets/a888c955-de79-462b-b568-431e4bed79de


A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about their content using local LLM.

## ✨ Features

- Multi-format document support (PDF, DOCX, CSV, TXT)
- OCR support for scanned PDFs
- AI-powered document summarization
- Conversation memory with chat history download
- Completely local - no API costs

## 🛠️ Technology Stack

- **LLM**: Ollama Mistral 7B
- **Embeddings**: SentenceTransformers (BAAI/bge-base-en-v1.5)
- **Vector DB**: FAISS
- **Reranking**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Frontend**: Streamlit

## 📋 Prerequisites (Windows)

Before installation, you need:
- Python 3.8+ (Preferance 3.10) 
- Ollama installed
- Tesseract OCR installed

---

## 🚀 Installation Guide (Windows)

### Step 1: Install Ollama

1. Download Ollama for Windows from: https://ollama.ai
2. Run the installer
3. Open Command Prompt and verify installation:
```bash
ollama --version
```

### Step 2: Install Tesseract OCR

1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (use default installation path: `C:\Program Files\Tesseract-OCR`)
3. Add Tesseract to PATH:
   - Search "Environment Variables" in Windows
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\Program Files\Tesseract-OCR`
   - Click "OK" on all windows

4. Verify installation in Command Prompt:
```bash
tesseract --version
```

### Step 3: Clone/Download Project

Download or clone this repository to your local machine.

### Step 4: Create Virtual Environment

Open Command Prompt in project folder:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Setup Ollama & Download Model

### Step 1: Start Ollama Server

Open Command Prompt and run:
```bash
ollama serve
```

**Keep this terminal open** - Ollama must be running for the chatbot to work.

### Step 2: Download Mistral Model

Open a **NEW** Command Prompt and run:
```bash
ollama pull mistral:7b
```

This will download the Mistral 7B model (~4GB). Wait for download to complete.

### Step 3: Verify Model Installation

```bash
ollama list
```

You should see `mistral:7b` in the list.

### Step 4: Test Ollama

```bash
ollama run mistral:7b
```

Type a test message. If it responds, Ollama is working correctly. Type `/bye` to exit.

### Step 5: Check Ollama Port

Ollama runs on port `11434` by default. Verify it's running:
```bash
curl http://localhost:11434
```

You should see: `Ollama is running`

---

## ▶️ Running the Application

### Step 1: Ensure Ollama is Running

In one Command Prompt:
```bash
ollama serve
```

**Keep this window open!**

### Step 2: Start Streamlit App

In a **NEW** Command Prompt (in project folder with venv activated):
```bash
streamlit run app.py
```

### Step 3: Access Application

- Browser should automatically open to `http://localhost:8501`
- If not, manually open the URL shown in terminal

---

## 🐛 Common Issues & Solutions

### Issue 1: "Ollama connection error"

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Check if model exists
ollama list

# Test the model
ollama run mistral:7b
```

### Issue 2: "Tesseract not found"

**Solution:**
- Verify Tesseract is installed: `tesseract --version`
- Check PATH is set correctly
- Restart Command Prompt after adding to PATH

### Issue 3: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 4: Ollama port already in use

**Solution:**
```bash
# Check what's running on port 11434
netstat -ano | findstr :11434

# Kill the process if needed (replace PID with actual number)
taskkill /PID <PID> /F

# Restart Ollama
ollama serve
```

### Issue 5: Streamlit port 8501 already in use

**Solution:**
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── app.py                 # Streamlit frontend
├── rag_backend.py         # Backend RAG logic
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## 🔄 Quick Start Checklist

- [ ] Python 3.8+ (Preferance 3.10) 
- [ ] Ollama installed and `ollama --version` works
- [ ] Tesseract installed and `tesseract --version` works
- [ ] Virtual environment created and activated
- [ ] Dependencies installed via `pip install -r requirements.txt`
- [ ] Mistral model downloaded: `ollama pull mistral:7b`
- [ ] Ollama server running: `ollama serve` (in separate terminal)
- [ ] Streamlit app started: `streamlit run app.py`
- [ ] Browser opened to `http://localhost:8501`

---

## 📝 Notes

- **Keep Ollama running**: The `ollama serve` command must stay running while using the chatbot
- **First run**: Initial model loading may take 10-20 seconds
- **Large documents**: Processing time increases with document size
- **Memory**: Ensure at least 8GB RAM for smooth operation
