import git
import os
from pathlib import Path
import pymongo
from pymongo import ReplaceOne
from sentence_transformers import SentenceTransformer
import numpy as np
import time

REPO_DIR = "byzfl_repo"

client = pymongo.MongoClient("mongodb+srv://helenlkhoury:PpMIpywcSCsxqb2n@rag.kpcg2.mongodb.net/?retryWrites=true&w=majority&appName=RAG")
db = client.byzfl_db  
collection = db.code_chunks 

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # embedding model (384-dimensional vectors)


def clone_repository(repo_url: str, target_dir: str) -> None:
    """Clone the GitHub repository to a local directory."""
    if not os.path.exists(target_dir):
        print(f"Cloning repository from {repo_url} to {target_dir}...")
        git.Repo.clone_from(repo_url, target_dir)
    else:
        print(f"Repository already cloned at {target_dir}")

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

def chunk_content(content: str, max_length: int = 512) -> list[str]:
    """Split content into chunks of max_length characters."""
    return [content[i:i + max_length] for i in range(0, len(content), max_length)]

def generate_embeddings(code_files: list[dict]) -> list[dict]:
    """Generate embeddings for each code file, splitting into chunks if needed."""
    documents = []
    for file in code_files:
        chunks = chunk_content(file["content"])
        file_size = len(file["content"])  # Measure file size in characters
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()  # Convert NumPy array to list
            documents.append({
                "path": file["path"],
                "filename": file["filename"],
                "size": file_size,  # Include size
                "chunk_id": i,
                "content": chunk,
                "embedding": embedding  # Ensure this matches the MongoDB vector index
            })
    return documents

def store_documents(documents: list[dict]) -> None:
    """Store documents with embeddings and metadata in MongoDB."""
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

def vector_search(query: str, k: int = 5) -> list[dict]:
    """Perform a vector search on the collection using the correct Atlas Vector Search syntax."""
    query_embedding = model.encode(query).tolist()
    print("Query embedding length:", len(query_embedding))  # Debug

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "filename": 1,
                "size": 1,  # Include size in results
                "content": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$sort": {"size": -1}  # Sort by document size (descending)
        }
    ]

    try:
        results = list(collection.aggregate(pipeline))
        print("Raw search results:", results)  # Debug output
        return results
    except Exception as e:
        print(f"Error executing vector search: {e}")
        return []

def rag_query(query: str) -> str:
    """Perform a RAG query and return a response using retrieved code chunks."""
    results = vector_search(query)
    if not results:
        return "No relevant code found."
    context = "\n\n".join([f"{r['filename']}:\n{r['content']}" for r in results])
    return f"Based on the code in the byzfl repository:\n{context}"

def benchmark_queries(queries: list[str]) -> dict:
    """Benchmark the RAG system with multiple queries."""
    results = {}
    for query in queries:
        start_time = time.time()
        response = rag_query(query)
        latency = time.time() - start_time
        results[query] = {
            "latency": latency,
            "response": response
        }
        print(f"Query: {query}")
        print(f"Latency: {latency:.2f} seconds")
        print(f"Response: {response[:200]}...\n")  # Truncate for brevity
    return results

# Main execution
if __name__ == "__main__":
    # Clone and process repository
    clone_repository("https://github.com/LPD-EPFL/byzfl.git", REPO_DIR)
    code_files = extract_code_files(REPO_DIR)
    documents = generate_embeddings(code_files)
    store_documents(documents)

    # Wait for index to be active (manual check required in Atlas UI)
    print("Please ensure the 'vector_index' is active in Atlas Search before proceeding with queries.")

    # Benchmark with test queries
    test_queries = [
        "How does the byzfl code handle Byzantine faults?",
        "What is the role of the consensus algorithm in byzfl?",
        "Show me the message passing logic in byzfl."
    ]
    benchmark_results = benchmark_queries(test_queries)
    avg_latency = np.mean([r["latency"] for r in benchmark_results.values()])
    print(f"Average Latency: {avg_latency:.2f} seconds")
