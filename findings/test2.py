import os
import time
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from git import Repo

# Utility: Measure execution time
def time_function(func, *args, **kwargs):
    """Utility to measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    """Compute text embedding."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Initialize FAISS index
d = 384  # MiniLM-L6-v2 produces 384-dimensional embeddings
index = faiss.IndexFlatL2(d)
file_mapping = {}

# Clone the GitHub repository (if not already cloned)
REPO_URL = "https://github.com/LPD-EPFL/byzfl"
LOCAL_REPO_PATH = "./byzfl_repo"

if not os.path.exists(LOCAL_REPO_PATH):
    print("Cloning repository...")
    Repo.clone_from(REPO_URL, LOCAL_REPO_PATH)

# Read all text-based files from the repo
def get_code_files(repo_path):
    """Extract all source code files from the repository."""
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".md", ".txt", ".sh", ".yaml", ".json")):  # Add more extensions if needed
                file_paths.append(os.path.join(root, file))
    return file_paths

def load_code_content(file_paths):
    """Load content from each file and return as text list."""
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(content)
                file_mapping[len(texts) - 1] = file_path  # Store file index mapping
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    return texts

code_files = get_code_files(LOCAL_REPO_PATH)
documents = load_code_content(code_files)

# Add embeddings to FAISS index
def add_to_database(texts):
    """Add code documents to FAISS database."""
    embeddings = np.vstack([get_embedding(text) for text in texts])
    index.add(embeddings)

# Query FAISS
def query_database(query, k=5):
    """Retrieve top-k relevant code files, handling empty results safely."""
    query_embedding = get_embedding(query)

    # Ensure FAISS is not empty before querying
    if index.ntotal == 0:
        print("⚠️ FAISS index is empty. Returning no results.")
        return []

    distances, indices = index.search(query_embedding, k)

    # Ensure index values are within bounds
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(file_mapping):  # Check bounds
            results.append((file_mapping[idx], distances[0][i]))
    
    return results

# Benchmark retrieval
def benchmark_retrieval(texts, query_texts, k=5):
    add_to_database(texts)
    retrieval_times = []
    for query in query_texts:
        _, time_taken = time_function(query_database, query, k)
        retrieval_times.append(time_taken)
    return retrieval_times

# Run Benchmarking
queries = ["How does Byzantine resilience work?", "What is the FL training process?"]
print("Benchmark on full repo:", benchmark_retrieval(documents, queries))

# Query example
print("\nExample query results:")
results = query_database("Byzantine resilience in federated learning", k=3)
for file_path, score in results:
    print(f"File: {file_path}, Score: {score}")
