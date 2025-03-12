import git
import os
from pathlib import Path
import pymongo
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# MongoDB connection
REPO_DIR = "byzfl_repo"
client = pymongo.MongoClient("mongodb+srv://helenlkhoury:PpMIpywcSCsxqb2n@rag.kpcg2.mongodb.net/?retryWrites=true&w=majority&appName=RAG")
db = client.byzfl_db  
collection = db.code_chunks  

# Load Sentence Transformer Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # 384-dimensional vectors

# 1️⃣ Clone Repository (Same as before)
def clone_repository(repo_url: str, target_dir: str) -> None:
    if not os.path.exists(target_dir):
        print(f"Cloning repository from {repo_url} to {target_dir}...")
        git.Repo.clone_from(repo_url, target_dir)
    else:
        print(f"Repository already cloned at {target_dir}")

# 2️⃣ Extract Code Files
def extract_code_files(repo_dir: str) -> list[dict]:
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
    return [content[i:i + max_length] for i in range(0, len(content), max_length)]

# 4️⃣ Generating Embeddings
def generate_embeddings(code_files: list[dict]) -> list[dict]:
    documents = []
    for file in code_files:
        chunks = chunk_content(file["content"])
        file_size = len(file["content"])  # Measure file size in characters
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()  # Convert NumPy array to list
            documents.append({
                "path": file["path"],
                "filename": file["filename"],
                "size": file_size,
                "chunk_id": i,
                "content": chunk,
                "embedding": embedding  # Store without an index
            })
    return documents

# 5️⃣ Storing Documents Without Indexing
def store_documents(documents: list[dict]) -> None:
    collection.insert_many(documents)  # No indexing, raw insert
    print(f"Stored {len(documents)} documents without an index.")

# 6️⃣ Brute-Force Vector Search (Without Index)
def brute_force_search(query: str, k: int = 5) -> list[dict]:
    """Manually retrieve all documents and compute similarity in Python."""
    query_embedding = model.encode(query)

    # Retrieve all documents (VERY SLOW if dataset is large)
    all_docs = list(collection.find({}, {"filename": 1, "size": 1, "content": 1, "embedding": 1}))

    # Compute cosine similarity manually
    results = []
    for doc in all_docs:
        doc_embedding = np.array(doc["embedding"])
        similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        results.append({**doc, "score": similarity})

    # Sort results by similarity (descending order)
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]

    return results

# 7️⃣ Measure MongoDB Performance Without Index
def check_performance():
    """Check MongoDB performance using db.currentOp() and explain()."""
    
    admin_db = client["admin"]  # Switch to admin database

    # Check current operations (requires admin access)
    try:
        print("\n🔍 Checking current operations in MongoDB (db.currentOp()):")
        current_operations = admin_db.command("currentOp")
        print(current_operations)
    except pymongo.errors.OperationFailure as e:
        print(f"⚠️ Permission Error: {e}")

    # Use explain() to analyze query execution
    print("\n🛠️ Checking query performance with explain():")
    sample_query = collection.find({"filename": "attacks.py"}).explain()
    print(sample_query)

# 8️⃣ Running a RAG Query
def rag_query(query: str) -> str:
    results = brute_force_search(query)
    if not results:
        return "No relevant code found."
    context = "\n\n".join([f"{r['filename']}:\n{r['content']}" for r in results])
    return f"Based on the code in the byzfl repository:\n{context}"

# 9️⃣ Benchmark Queries Without Index
def benchmark_queries(queries: list[str]) -> dict:
    results = {}
    for query in queries:
        start_time = time.time()
        response = rag_query(query)
        latency = time.time() - start_time
        results[query] = {"latency": latency, "response": response}

        print(f"Query: {query}")
        print(f"Latency: {latency:.2f} seconds")
        print(f"Response: {response[:200]}...\n")  # Truncate output

    return results

# 🔟 Main Execution
if __name__ == "__main__":
    # Clone and process repository
    clone_repository("https://github.com/LPD-EPFL/byzfl.git", REPO_DIR)
    code_files = extract_code_files(REPO_DIR)
    documents = generate_embeddings(code_files)
    store_documents(documents)

    # 🔍 Check MongoDB Performance
    check_performance()

    # Run benchmark tests
    test_queries = [
        "How does the byzfl code handle Byzantine faults?",
        "What is the role of the consensus algorithm in byzfl?",
        "Show me the message passing logic in byzfl."
    ]
    benchmark_results = benchmark_queries(test_queries)
    avg_latency = np.mean([r["latency"] for r in benchmark_results.values()])
    print(f"⚡ Average Latency Without Indexing: {avg_latency:.2f} seconds")
