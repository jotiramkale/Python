# PDF Analyzer

A FastAPI-powered RAG workspace for asking grounded questions about PDF documents.

## Stack

- FastAPI and Uvicorn
- HTML, CSS, and vanilla JavaScript
- PyPDF and RecursiveCharacterTextSplitter
- Sentence Transformers with `all-MiniLM-L6-v2`
- Persistent ChromaDB
- LangChain and Groq

## Run locally

1. Activate the project environment:

```powershell
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Add your server-side key to `.env`:

```text
GROQ_API_KEY=your_key_here
```

4. Start the API and frontend:

```powershell
python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000.

The Groq key is used only by the backend and is never sent to the browser.
