import git
import os
from pathlib import Path
import pymongo
from pymongo import ReplaceOne
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import json

# MongoDB Atlas connection
REPO_DIR = "byzfl_repo"
client = pymongo.MongoClient("mongodb+srv://helenlkhoury:PpMIpywcSCsxqb2n@rag.kpcg2.mongodb.net/?retryWrites=true&w=majority&appName=RAG")
db = client.byzfl_db  
collection = db.code_chunks 

# Load Sentence Transformer Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384-dimensional vectors

# 1️⃣ Clone Repository
def clone_repository(repo_url: str, target_dir: str) -> None:
    """Clone the GitHub repository to a local directory."""
    if not os.path.exists(target_dir):
        print(f"Cloning repository from {repo_url} to {target_dir}...")
        git.Repo.clone_from(repo_url, target_dir)
    else:
        print(f"Repository already cloned at {target_dir}")

# 2️⃣ Extract Code Files
def extract_code_files(repo_dir: str) -> list[dict]:
    """Extract content from code files in the repository."""
    code_files = []
    valid_extensions = {".py", ".cpp", ".h", ".hpp"}
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if Path(file).suffix in valid_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    code_files.append({
                        "path": file_path,
                        "content": content,
                        "filename": file
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return code_files

# 3️⃣ Chunking Code Content
def chunk_content(content: str, max_length: int = 512) -> list[str]:
    """Split content into chunks of max_length characters."""
    return [content[i:i + max_length] for i in range(0, len(content), max_length)]

# 4️⃣ Generate Embeddings
def generate_embeddings(code_files: list[dict]) -> list[dict]:
    """Generate embeddings for each code file, splitting into chunks if needed."""
    documents = []
    for file in code_files:
        chunks = chunk_content(file["content"])
        file_size = len(file["content"])
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            documents.append({
                "path": file["path"],
                "filename": file["filename"],
                "size": file_size,
                "chunk_id": i,
                "content": chunk,
                "embedding": embedding
            })
    return documents

# 5️⃣ Store Documents and Measure Data Size
def store_documents(documents: list[dict]) -> None:
    """Store documents with embeddings and metadata in MongoDB."""
    start_time = time.time()
    requests = [
        ReplaceOne(
            {"path": doc["path"], "chunk_id": doc["chunk_id"]},  # Unique identifier
            doc,
            upsert=True
        )
        for doc in documents
    ]
    if requests:
        result = collection.bulk_write(requests)
        print(f"Stored {result.upserted_count} new documents, modified {result.modified_count} existing documents.")
    
    latency = time.time() - start_time
    data_size = len(json.dumps(documents)) / 1024 / 1024  # MB
    print(f"Write Latency: {latency:.2f} seconds")
    print(f"Data Size: {data_size:.2f} MB")

# 6️⃣ Create Vector Index (Manual Step or Programmatic Attempt)
def create_vector_index():
    """Attempt to create the vector index programmatically (requires Atlas Search API access)."""
    try:
        # Note: MongoDB Atlas Search indexes are typically created via the UI or Atlas API, not PyMongo directly.
        # This is a placeholder; you’ll need to create it manually or use the Atlas API.
        index_definition = {
            "fields": [
                {
                    "numDimensions": 384,
                    "path": "embedding",
                    "similarity": "cosine",
                    "type": "vector"
                }
            ]
        }
        print("Vector index creation must be done manually in Atlas UI or via Atlas API.")
        print("Index definition to use:")
        print(json.dumps(index_definition, indent=2))
    except Exception as e:
        print(f"Error attempting to create index: {e}")

# Main Execution
if __name__ == "__main__":
    # Step 1: Drop existing collection (optional, for fresh start)
    collection.drop()
    print("Dropped existing 'code_chunks' collection (if it existed).")

    # Step 2: Create vector index (manual step required)
    create_vector_index()
    print("Please create the vector index named 'vector_index' in Atlas UI with the above definition.")
    print("Waiting 30 seconds for manual index creation (adjust as needed)...")
    time.sleep(30)  # Give time to manually create the index if needed

    # Step 3: Clone, process, and populate
    clone_repository("https://github.com/LPD-EPFL/byzfl.git", REPO_DIR)
    code_files = extract_code_files(REPO_DIR)
    documents = generate_embeddings(code_files)
    store_documents(documents)

    # Verify collection population
    doc_count = collection.count_documents({})
    print(f"Collection 'code_chunks' now contains {doc_count} documents.")