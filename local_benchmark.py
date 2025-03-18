import git
import os
from pathlib import Path
import pymongo
from pymongo import ReplaceOne
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import json
import psutil  # For disk I/O
import concurrent.futures  # For concurrency testing
import statistics

# MongoDB local connection
REPO_DIR = "byzfl_repo"
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Default local MongoDB connection
db = client.byzfl_db  # Local database name
collection = db.code_chunks 

# Load Sentence Transformer Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384-dimensional vectors

# Utility to measure disk I/O
def get_disk_io():
    io = psutil.disk_io_counters()
    return io.read_bytes, io.write_bytes

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

# 4️⃣ Generate Embeddings with Initial Indexing Time
def generate_embeddings(code_files: list[dict]) -> list[dict]:
    start_time = time.time()
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
    indexing_time = time.time() - start_time
    print(f"Initial Indexing Time (Embedding Generation): {indexing_time:.2f} seconds")
    return documents

# 5️⃣ Store Documents with Write Latency, Disk I/O, and Data Size
def store_documents(documents: list[dict]) -> None:
    start_time = time.time()
    initial_read_bytes, initial_write_bytes = get_disk_io()
    
    requests = [ReplaceOne({"path": doc["path"], "chunk_id": doc["chunk_id"]}, doc, upsert=True) for doc in documents]
    if requests:
        result = collection.bulk_write(requests)
        print(f"Stored {result.upserted_count} new documents, modified {result.modified_count} existing documents.")
    
    write_latency = time.time() - start_time
    final_read_bytes, final_write_bytes = get_disk_io()
    
    disk_read = (final_read_bytes - initial_read_bytes) / 1024 / 1024  # MB
    disk_write = (final_write_bytes - initial_write_bytes) / 1024 / 1024  # MB
    data_size = len(json.dumps(documents)) / 1024 / 1024  # MB
    
    # Get index size from local MongoDB
    stats = db.command("collStats", "code_chunks")
    index_size = stats.get("totalIndexSize", 0) / 1024 / 1024  # Convert bytes to MB
    
    print(f"Write Latency: {write_latency:.2f} seconds")
    print(f"Disk I/O - Read: {disk_read:.2f} MB, Write: {disk_write:.2f} MB")
    print(f"Data Size: {data_size:.2f} MB")
    print(f"Index Size: {index_size} MB")

# 6️⃣ Vector Search with Search Latency and Query Overhead
def vector_search(query: str, k: int = 5) -> list[dict]:
    overhead_start = time.time()
    query_embedding = model.encode(query).tolist()
    query_overhead = time.time() - overhead_start
    
    search_start = time.time()
    # For local MongoDB, we'll use a simpler similarity search since $vectorSearch is Atlas-specific
    results = collection.find(
        {"embedding": {"$exists": True}}
    ).sort([("size", -1)]).limit(k)
    
    # Calculate cosine similarity manually
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

    scored_results = []
    for doc in results:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        scored_results.append({
            "filename": doc["filename"],
            "size": doc["size"],
            "content": doc["content"],
            "score": similarity
        })
    
    # Sort by similarity score and take top k
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    final_results = scored_results[:k]
    
    search_latency = time.time() - search_start
    
    print(f"Search Latency: {search_latency:.2f} seconds")
    print(f"Query Overhead: {query_overhead:.2f} seconds")
    return final_results

# 7️⃣ RAG Query
def rag_query(query: str, k: int = 5) -> str:
    results = vector_search(query, k)
    if not results:
        return "No relevant code found."
    context = "\n\n".join([f"{r['filename']}:\n{r['content']}" for r in results])
    return f"Based on the code in the byzfl repository:\n{context}"

# 8️⃣ Benchmark Queries with Different k Values
def benchmark_queries(queries: list[str], k_values: list[int] = [2, 5, 10]) -> dict:
    results = {}
    for k in k_values:
        print(f"\nBenchmarking with k={k}")
        k_results = {}
        for query in queries:
            start_time = time.time()
            response = rag_query(query, k)
            latency = time.time() - start_time
            k_results[query] = {"latency": latency, "response": response}
            print(f"Query: {query}, Latency: {latency:.2f} seconds")
        results[k] = k_results
    return results

# 9️⃣ Concurrency Test
def concurrency_test(queries: list[str], num_threads: int = 10):
    print(f"\nConcurrency Test with {num_threads} threads")
    def run_query(query):
        start_time = time.time()
        rag_query(query)
        return time.time() - start_time
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        latencies = list(executor.map(run_query, queries * num_threads))  # Repeat queries for load
    
    avg_latency = statistics.mean(latencies)
    std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    print(f"Average Latency under {num_threads} concurrent queries: {avg_latency:.2f} seconds")
    print(f"Std Dev: {std_dev:.2f} seconds")

# Main Execution
if __name__ == "__main__":
    # Step 1: Drop existing collection (optional)
    collection.drop()
    print("Dropped existing 'code_chunks' collection (if it existed).")

    # Step 2: Create index for local MongoDB (simpler index since vector search is manual)
    collection.create_index([("path", 1), ("chunk_id", 1)])
    print("\nCreated basic index on 'path' and 'chunk_id' for local MongoDB.")

    # Step 3: Clone, process, and populate
    clone_repository("https://github.com/LPD-EPFL/byzfl.git", REPO_DIR)
    code_files = extract_code_files(REPO_DIR)
    documents = generate_embeddings(code_files)
    store_documents(documents)

    # Verify collection population
    doc_count = collection.count_documents({})
    print(f"Collection 'code_chunks' now contains {doc_count} documents.")

    # Step 4: Test queries
    test_queries = [
        "How does the byzfl code handle Byzantine faults?",
        "What is the role of the consensus algorithm in byzfl?",
        "Show me the message passing logic in byzfl?"
    ]

    # Benchmark for k=2, 5, 10
    benchmark_results = benchmark_queries(test_queries, k_values=[2, 5, 10])

    # Concurrency test
    concurrency_test(test_queries, num_threads=10)

    # Summary
    for k, k_results in benchmark_results.items():
        avg_latency = np.mean([r["latency"] for r in k_results.values()])
        print(f"\nAverage Latency for k={k}: {avg_latency:.2f} seconds")