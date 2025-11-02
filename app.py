from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Medical Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pinecone setup
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index configuration
index_name = "medical-chatbot"

# Connect to existing Pinecone index
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Ollama model (using qwen3:4b-instruct as in trials.ipynb)
chatModel = ChatOllama(model="qwen3:4b-instruct")

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create RAG chain using LCEL (LangChain Expression Language) - LangChain 1.0 pattern
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain_input(user_input: str):
    """Format input for RAG chain"""
    docs = retriever.invoke(user_input)
    return {"context": format_docs(docs), "input": user_input}

rag_chain = (
    RunnablePassthrough() | create_rag_chain_input
) | prompt | chatModel | StrOutputParser()

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

# Serve static files (React app - CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main HTML file"""
    return FileResponse("static/index.html", media_type="text/html")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat requests and return AI responses using RAG chain
    """
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        print(f"Received query: {request.message}")
        
        # Invoke RAG chain (returns string directly in LCEL)
        answer = rag_chain.invoke(request.message)
        
        if not answer:
            answer = "I apologize, but I couldn't generate a response."
        print(f"Response: {answer}")
        
        return ChatResponse(answer=answer)
    
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Medical Chatbot API is running"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
