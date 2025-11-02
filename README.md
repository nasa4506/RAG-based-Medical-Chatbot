# Medical Chatbot 🤖🏥

An AI-powered medical chatbot application that uses Retrieval Augmented Generation (RAG) to provide accurate medical information based on a knowledge base of medical documents. The application leverages local LLM models via Ollama and Pinecone vector database for efficient document retrieval.

## ✨ Features

- **AI-Powered Conversations**: Chat with an intelligent medical assistant powered by Ollama's local LLM models
- **RAG-based Responses**: Retrieves relevant context from medical documents to provide accurate answers
- **Beautiful Modern UI**: Responsive React-based frontend with smooth animations and intuitive design
- **Real-time Chat**: Fast and responsive chat interface with typing indicators
- **Local LLM Support**: Uses Ollama for privacy and offline capabilities
- **Vector Search**: Leverages Pinecone for efficient semantic search across medical documents

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for building LLM applications
- **Ollama**: Local LLM runtime for running models like `qwen3:4b-instruct`
- **Pinecone**: Vector database for storing and retrieving document embeddings
- **HuggingFace Embeddings**: Sentence transformers for document embeddings via `langchain-huggingface` (`all-MiniLM-L6-v2`)
- **Python 3.12+**: Programming language

### Frontend
- **React 18**: JavaScript library for building user interfaces
- **HTML5 & CSS3**: Modern web technologies
- **Vanilla JavaScript**: No build tools required

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.12+** - [Download Python](https://www.python.org/downloads/)
2. **Ollama** - [Install Ollama](https://ollama.ai/download)
3. **Pinecone Account** - [Sign up for Pinecone](https://www.pinecone.io/)
4. **Node.js** (optional, for development) - Not required as we use CDN React

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Medical_chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama Model

Download and install the required model in Ollama:

```bash
ollama pull qwen3:4b-instruct
```

Make sure Ollama is running:
```bash
ollama serve
```

### 5. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
```

You can get your Pinecone API key from the [Pinecone Console](https://app.pinecone.io/).

### 6. Prepare Your Medical Documents

Place your PDF medical documents in the `data/` directory. The application will process these documents and create embeddings.

### 7. Set Up Vector Store (First Time Only)

If you haven't already set up the Pinecone index, run the `store_index.py` script:

```bash
python src/store_index.py
```

This will:
- Load PDF documents from the `data/` directory
- Split them into chunks
- Generate embeddings
- Upload to Pinecone vector database

## 🎯 Usage

### Start the Application

```bash
python app.py
```

The application will start on `http://localhost:8080`

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8080
```

### Using the Chatbot

1. Type your medical question in the input field
2. Click "Send" or press Enter
3. The chatbot will retrieve relevant information from the knowledge base
4. View the AI-generated response in the chat interface

### API Endpoints

#### Chat Endpoint
```http
POST /api/chat
Content-Type: application/json

{
  "message": "What is acne?"
}
```

**Response:**
```json
{
  "answer": "Acne is a common skin disease characterized by..."
}
```

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Medical Chatbot API is running"
}
```

## 📁 Project Structure

```
Medical_chatbot/
├── app.py                  # FastAPI application and main entry point
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .env                   # Environment variables (create this)
├── data/                  # Medical PDF documents
│   └── Medical_book.pdf
├── static/                # Frontend files
│   ├── index.html         # Main HTML file
│   ├── app.js             # React application
│   └── styles.css         # Styling
├── src/                   # Source code
│   ├── __init__.py
│   ├── helper.py          # Helper functions (PDF loading, embeddings)
│   ├── prompt.py          # System prompts
│   └── store_index.py     # Script to populate Pinecone index
└── research/              # Research and experiments
    └── trials.ipynb       # Jupyter notebook with experiments
```

## 🔧 Configuration

### Changing the LLM Model

To use a different Ollama model, edit `app.py`:

```python
chatModel = ChatOllama(model="your-model-name")
```

Then make sure to pull the model:
```bash
ollama pull your-model-name
```

### Adjusting Retrieval Parameters

In `app.py`, you can modify the retriever settings:

```python
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  # Number of documents to retrieve
)
```

### Customizing the Prompt

Edit `src/prompt.py` to modify the system prompt that guides the AI's responses.

## 🐛 Troubleshooting

### Ollama Connection Issues

If you encounter connection errors with Ollama:

1. Ensure Ollama is running: `ollama serve`
2. Verify the model is installed: `ollama list`
3. Test the model: `ollama run qwen3:4b-instruct`

### Pinecone Errors

1. Verify your API key in the `.env` file
2. Check that the index name matches: `medical-chatbot`
3. Ensure the index exists in your Pinecone dashboard

### Empty Responses

- Check that documents have been indexed in Pinecone
- Verify the retriever is finding relevant documents
- Ensure Ollama is responding correctly

## 📝 Notes

- **Medical Disclaimer**: This chatbot is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
- **Data Privacy**: All processing happens locally (with Ollama) or through your Pinecone account
- **Performance**: Response times depend on your system's capabilities and Ollama model size

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Nasar Haider Jafri**
- Email: nasarjafri14@gmail.com

## 🙏 Acknowledgments

- LangChain community for the excellent framework
- Ollama team for local LLM support
- Pinecone for vector database infrastructure
- HuggingFace for embeddings models

---

**Made with ❤️ for better healthcare information access**

