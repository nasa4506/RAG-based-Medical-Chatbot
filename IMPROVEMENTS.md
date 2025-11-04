# Answer Quality Improvements for Medical Chatbot

## Changes Implemented

### 1. **Updated System Prompt** (`src/prompt.py`)
- **Before:** Limited to "three sentences maximum and keep the answer concise"
- **After:** Encourages comprehensive, detailed answers with:
  - Structured sections (Definition, Causes, Symptoms, Treatment)
  - Inclusion of medical terminology
  - Multiple relevant points from context
  - Professional medical writing style

### 2. **Increased Document Retrieval** (`app.py`)
- **Before:** `k=3` documents retrieved
- **After:** `k=10` documents retrieved
- **Rationale:** With Qwen3-4B's 256K context window, we can process significantly more context for comprehensive answers

### 3. **Enhanced Model Configuration** (`app.py`)
- **Added Parameters:**
  - `temperature=0.7`: More diverse and detailed responses
  - `num_predict=2048`: Longer response capability (increased from default ~512)
  - `top_p=0.9`: Nucleus sampling for quality
  - `top_k=40`: Focused vocabulary selection

### 4. **Improved Context Formatting** (`app.py`)
- **Before:** Simple `\n\n` joining
- **After:** Structured format with document numbers and source information
- **Benefit:** Better organization helps model understand and utilize context

## Additional Improvement Options

### Option A: Implement Re-ranking (Advanced)
Retrieve more documents initially, then re-rank for quality:

```python
# Retrieve more candidates
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 20}  # Get more candidates
)

# Then re-rank top 10 based on relevance (implement cross-encoder or LLM-based reranking)
```

### Option B: Hybrid Search
Combine semantic search with keyword search for better retrieval:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# Use MMR (Maximal Marginal Relevance) for diversity
retriever = docsearch.as_retriever(
    search_type="mmr",  # Instead of "similarity"
    search_kwargs={"k": 10, "fetch_k": 20}
)
```

### Option C: Query Expansion
Expand user queries to retrieve more relevant documents:

```python
def expand_query(question: str):
    """Add related medical terms to improve retrieval"""
    expansions = [
        f"{question}",
        f"medical definition {question}",
        f"symptoms causes treatment {question}"
    ]
    return expansions

# Use expanded queries for retrieval
```

### Option D: Multi-step Reasoning
Use the model's reasoning capabilities:

```python
# Update prompt to encourage step-by-step thinking
system_prompt = """
You are a medical assistant. Answer in this structure:
1. Brief overview
2. Detailed explanation with medical terminology
3. Relevant symptoms/causes/treatments
4. Important considerations
"""
```

### Option E: Context Compression
Use LLM to compress/summarize retrieved documents before final answer:

```python
def compress_context(docs, question):
    """Summarize retrieved docs to keep most relevant parts"""
    # Use LLM to extract key information from docs related to question
    pass
```

## Testing Recommendations

1. **Test with various question types:**
   - Simple definitions: "What is X?"
   - Complex queries: "Explain the causes and treatments of X"
   - Comparison questions: "Difference between X and Y"

2. **Monitor:**
   - Response length (target: 200-500 words for complex questions)
   - Medical accuracy
   - Context utilization
   - Response time

3. **Fine-tune parameters:**
   - Adjust `k` value (try 8, 10, 12, 15)
   - Adjust `temperature` (0.5-0.9 range)
   - Adjust `num_predict` based on desired length

## Expected Improvements

- **Answer Length:** 5-10x increase (from ~50 words to 200-400 words)
- **Detail Level:** More comprehensive with medical terminology
- **Context Usage:** Better utilization of retrieved documents
- **Structure:** More organized, readable answers

## Rollback Instructions

If answers become too verbose, you can:
1. Reduce `k` value back to 5-7
2. Add length constraints to prompt
3. Reduce `num_predict` to 1024
4. Keep detailed prompt but add "Be concise but comprehensive"

