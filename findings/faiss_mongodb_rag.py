import git
import os
from pathlib import Path
import pymongo
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time

# MongoDB connection
REPO_DIR = "byzfl_repo"
client = pymongo.MongoClient("mongodb+srv://helenlkhoury:PpMIpywcSCsxqb2n@rag.kpcg2.mongodb.net/?retryWrites=true&w=majority&appName=RAG")
db = client.byzfl_db  
collection = db.code_chunks  

# Load Sentence Transformer Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384-dimensional vectors
dimension = 384  # Embedding dimension for FAISS index

# 1️⃣ Clone Repository
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

# 4️⃣ Generating Embeddings and Building FAISS Index
def generate_embeddings_and_index(code_files: list[dict]) -> tuple[list[dict], faiss.IndexFlatL2]:
    documents = []
    embeddings = []
    for file in code_files:
        chunks = chunk_content(file["content"])
        file_size = len(file["content"])
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk)
            embeddings.append(embedding)
            documents.append({
                "path": file["path"],
                "filename": file["filename"],
                "size": file_size,
                "chunk_id": i,
                "content": chunk,
                "embedding": embedding.tolist()  # Store as list in MongoDB
            })

    # Build FAISS index
    embeddings_np = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings_np)  # Add embeddings to FAISS index

    return documents, index

# 5️⃣ Storing Documents in MongoDB
def store_documents(documents: list[dict]) -> None:
    collection.delete_many({})  # Clear existing data (optional, for fresh start)
    collection.insert_many(documents)
    print(f"Stored {len(documents)} documents in MongoDB.")

# 6️⃣ FAISS Vector Search
def faiss_search(query: str, index: faiss.IndexFlatL2, k: int = 5) -> list[dict]:
    query_embedding = model.encode(query).reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_embedding, k)  # FAISS search

    # Fetch corresponding documents from MongoDB
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        doc = collection.find_one({"chunk_id": int(idx % 1000), "size": {"$exists": True}})  # Adjust logic based on unique identifier
        if doc:
            results.append({**doc, "score": 1 - distance / 2})  # Normalize score (optional)

    return results

# 7️⃣ Measure MongoDB and FAISS Performance
def check_performance(index: faiss.IndexFlatL2):
    admin_db = client["admin"]

    # Check MongoDB current operations
    try:
        print("\n🔍 Checking current operations in MongoDB (db.currentOp()):")
        current_operations = admin_db.command("currentOp")
        print(current_operations)
    except pymongo.errors.OperationFailure as e:
        print(f"⚠️ Permission Error: {e}")

    # FAISS index stats
    print("\n🛠️ FAISS Index Stats:")
    print(f"Total vectors in FAISS index: {index.ntotal}")

# 8️⃣ Running a RAG Query with FAISS
def rag_query(query: str, index: faiss.IndexFlatL2) -> str:
    results = faiss_search(query, index)
    if not results:
        return "No relevant code found."
    context = "\n\n".join([f"{r['filename']}:\n{r['content']}" for r in results])
    return f"Based on the code in the byzfl repository:\n{context}"

# 9️⃣ Benchmark Queries with FAISS
def benchmark_queries(queries: list[str], index: faiss.IndexFlatL2) -> dict:
    results = {}
    for query in queries:
        start_time = time.time()
        response = rag_query(query, index)
        latency = time.time() - start_time
        results[query] = {"latency": latency, "response": response}

        print(f"Query: {query}")
        print(f"Latency: {latency:.2f} seconds")
        print(f"Response: {response[:200]}...\n")

    return results

# 🔟 Main Execution
if __name__ == "__main__":
    # Clone and process repository
    clone_repository("https://github.com/LPD-EPFL/byzfl.git", REPO_DIR)
    code_files = extract_code_files(REPO_DIR)
    documents, faiss_index = generate_embeddings_and_index(code_files)
    store_documents(documents)

    # Check performance
    check_performance(faiss_index)

    # Run benchmark tests
    test_queries = [
        "How does the byzfl code handle Byzantine faults?",
        "What is the role of the consensus algorithm in byzfl?",
        "Show me the message passing logic in byzfl."
    ]
    benchmark_results = benchmark_queries(test_queries, faiss_index)
    avg_latency = np.mean([r["latency"] for r in benchmark_results.values()])
    print(f"⚡ Average Latency with FAISS: {avg_latency:.2f} seconds")