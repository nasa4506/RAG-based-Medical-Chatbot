# Quick Start Guide 🚀

## Prerequisites Checklist:
- ✅ Python 3.12+ installed
- ✅ Virtual environment activated
- ✅ Ollama installed and running
- ✅ Pinecone API key

## Step-by-Step Instructions:

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Install/Update Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root (if you don't have one):

```env
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 4. Start Ollama (If not running)

Open a **new terminal window** and run:

```bash
ollama serve
```

Keep this terminal open while running the application.

### 5. Verify Ollama Model is Available

In another terminal, check if the model exists:

```bash
ollama list
```

If `qwen3:4b-instruct` is not listed, download it:

```bash
ollama pull qwen3:4b-instruct
```

### 6. Run the Application

Make sure you're in the project root directory with venv activated, then:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### 7. Access the Application

Open your web browser and go to:
```
http://localhost:8080
```

## Troubleshooting

**Port 8080 already in use?**
Change the port in `app.py` (line 110) or run:
```bash
uvicorn app:app --host 0.0.0.0 --port 8081 --reload
```

**Ollama connection error?**
- Make sure `ollama serve` is running
- Verify the model exists: `ollama list`
- Test the model: `ollama run qwen3:4b-instruct`

**Pinecone error?**
- Check your `.env` file has the correct API key
- Verify the index `medical-chatbot` exists in your Pinecone dashboard

**Module not found errors?**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

