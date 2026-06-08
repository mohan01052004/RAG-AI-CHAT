"""Script to recreate Pinecone index with correct dimensions"""
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")

# Delete existing index
print(f"Deleting existing index '{index_name}'...")
try:
    pc.delete_index(index_name)
    print("Index deleted successfully!")
except Exception as e:
    print(f"Note: {e}")

# Create new index with 384 dimensions for all-MiniLM-L6-v2
print(f"Creating new index '{index_name}' with 384 dimensions...")
pc.create_index(
    name=index_name,
    dimension=384,  # Changed from 1024 to 384
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region=os.getenv("PINECONE_ENV", "us-east-1")
    )
)
print("Index created successfully!")
print("You can now upload your PDFs again.")
