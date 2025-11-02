from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

# Check if index exists
index_exists = pc.has_index(index_name)

if not index_exists:
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index {index_name} already exists.")

index = pc.Index(index_name)

# Check if index already has vectors
index_stats = index.describe_index_stats()
existing_vectors = index_stats.get('total_vector_count', 0)

if existing_vectors > 0:
    print(f"\n⚠️  WARNING: Index already contains {existing_vectors} vectors!")
    response = input("Do you want to delete existing vectors and re-upload? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("Deleting all existing vectors...")
        # Delete all vectors by deleting the index and recreating it
        pc.delete_index(index_name)
        print("Waiting for index deletion...")
        import time
        time.sleep(5)  # Wait for deletion to complete
        
        # Recreate the index
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        index = pc.Index(index_name)
        print("Index recreated. Proceeding with upload...")
    else:
        print("Keeping existing vectors. Adding new documents may create duplicates.")
        print("To avoid duplicates, consider using a different index name or clearing this one first.")
        proceed = input("Continue anyway? (yes/no): ").strip().lower()
        if proceed != 'yes':
            print("Aborted.")
            exit(0)

print(f"\nUploading {len(text_chunks)} document chunks to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

print(f"✅ Successfully uploaded {len(text_chunks)} document chunks to {index_name}")
print(f"📊 You can now run app.py to use the chatbot!")