import os
import redis
import numpy as np
import time
from git import Repo
from llama_cpp import Llama

# Redis Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
INDEX_NAME = "code_index"

# Load embedding model
try:
    EMBEDDING_MODEL = Llama.from_pretrained(
        repo_id="CompendiumLabs/bge-base-en-v1.5-gguf",
        filename="bge-base-en-v1.5-f16.gguf",
        embedding=True
    )
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit(1)

# Connect to Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
except Exception as e:
    print(f"Error connecting to Redis: {e}")
    exit(1)

# Clone the GitHub repository
REPO_URL = "https://github.com/LPD-EPFL/byzfl"
LOCAL_REPO_PATH = "./byzfl_repo"
if not os.path.exists(LOCAL_REPO_PATH):
    print("Cloning repository...")
    try:
        Repo.clone_from(REPO_URL, LOCAL_REPO_PATH)
    except Exception as e:
        print(f"Error cloning repository: {e}")
        exit(1)

def get_code_files(repo_path):
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".md", ".txt", ".sh", ".yaml", ".json")):
                file_paths.append(os.path.join(root, file))
    return file_paths

def load_code_content(file_paths):
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    return texts

def get_embedding(text):
    try:
        embedding = EMBEDDING_MODEL.embed(text)
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error computing embedding: {e}")
        return None

def create_redis_index():
    try:
        redis_client.flushdb()
        print("✅ Redis index created!")
    except Exception as e:
        print(f"⚠️ Error creating Redis index: {e}")

def add_to_redis(file_paths, texts):
    try:
        for i, text in enumerate(texts):
            embedding = get_embedding(text)
            if embedding is not None:
                redis_client.hset(f"doc:{i}", mapping={
                    "embedding": embedding.tobytes(),  # Stored as bytes
                    "text": text,
                    "file_path": file_paths[i]
                })
        print("✅ Documents added to Redis!")
    except Exception as e:
        print(f"Error adding to Redis: {e}")

def query_redis(query_text, k=5):
    start_time = time.time()
    
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return []
    embedding_time = time.time() - start_time
    
    retrieval_start = time.time()
    results = redis_client.keys("doc:*")
    retrieval_time = time.time() - retrieval_start
    
    redis_fetch_start = time.time()
    retrieved_files = []
    for key in results:
        # Get text fields with hgetall (decoded as strings)
        doc = redis_client.hgetall(key)
        file_path = doc["file_path"]
        # Get embedding separately as bytes
        emb_bytes = redis_client.hget(key, "embedding")  # hget returns bytes for binary data
        doc_embedding = np.frombuffer(emb_bytes, dtype=np.float32)
        similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        retrieved_files.append((file_path, similarity))
    
    retrieved_files.sort(key=lambda x: x[1], reverse=True)
    redis_fetch_time = time.time() - redis_fetch_start
    
    total_time = time.time() - start_time
    print(f"⏱ Query Latency Breakdown: Embedding={embedding_time:.4f}s, Retrieval={retrieval_time:.4f}s, Redis Fetch={redis_fetch_time:.4f}s, Total={total_time:.4f}s")
    
    return retrieved_files[:k]

def measure_throughput(queries, k=5, duration=10):
    start_time = time.time()
    query_count = 0

    while time.time() - start_time < duration:
        for query in queries:
            query_redis(query, k)
            query_count += 1
            if time.time() - start_time >= duration:
                break
    
    elapsed_time = time.time() - start_time
    qps = query_count / elapsed_time
    print(f"⚡ Throughput: {qps:.2f} queries per second")
    return qps

def get_redis_memory_usage():
    try:
        info = redis_client.info("memory")
        used_memory = info["used_memory_human"]
        print(f"🗄️ Redis Memory Usage: {used_memory}")
        return used_memory
    except Exception as e:
        print(f"Error getting memory usage: {e}")
        return None

def test_scalability(corpus_sizes, query_text, k=5):
    results = []
    for size in corpus_sizes:
        print(f"\n🚀 Testing with {size} documents")
        create_redis_index()
        limited_paths = file_paths[:size]
        limited_docs = documents[:size]
        add_to_redis(limited_paths, limited_docs)
        start_time = time.time()
        query_redis(query_text, k)
        retrieval_time = time.time() - start_time
        memory_usage = get_redis_memory_usage()
        results.append((size, retrieval_time, memory_usage))
    return results

def get_database_size():
    try:
        size = redis_client.dbsize()
        print(f"📦 Redis database size: {size} keys")
        return size
    except Exception as e:
        print(f"Error getting database size: {e}")
        return 0

# Execution
file_paths = get_code_files(LOCAL_REPO_PATH)
documents = load_code_content(file_paths)
create_redis_index()
add_to_redis(file_paths, documents)
benchmark_queries = ["How does Byzantine resilience work?", "What is the FL training process?"]

print("\n🔍 Running Benchmark Tests...")
throughput = measure_throughput(benchmark_queries)
scalability_results = test_scalability([10, 50, 100], "Byzantine resilience in federated learning")

query_text = "Byzantine resilience in federated learning"
results = query_redis(query_text, k=3)

print("\nExample query results:")
for file_path, score in results:
    print(f"File: {file_path}, Score: {score:.4f}")

get_database_size()
get_redis_memory_usage()