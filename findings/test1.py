import time
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

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
d = 384  # Embedding size
index = faiss.IndexFlatL2(d)

def add_to_database(texts):
    """Add documents to FAISS database."""
    embeddings = np.vstack([get_embedding(text) for text in texts])
    index.add(embeddings)

def query_database(query, k=5):
    """Retrieve top-k results from FAISS index."""
    query_embedding = get_embedding(query)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Benchmarking retrieval performance
def benchmark_retrieval(texts, query_texts, k=5):
    add_to_database(texts)
    retrieval_times = []
    for query in query_texts:
        _, time_taken = time_function(query_database, query, k)
        retrieval_times.append(time_taken)
    return retrieval_times

# Mutation tests
def mutate_database(mutation_type, texts):
    """Modify database by adding/removing content."""
    if mutation_type == 'add':
        new_texts = ["New information about AI scaling", "Recent advancements in transformers"]
        add_to_database(new_texts)
    elif mutation_type == 'remove':
        index.reset()  # Remove all and re-add limited subset
        add_to_database(texts[:len(texts)//2])
    elif mutation_type == 'corrupt':
        corrupted_data = ["\u0000\u0000\u0000", "!!!###"]
        add_to_database(corrupted_data)
    else:
        raise ValueError("Invalid mutation type")

# Example usage
documents = ["Federated learning is a distributed approach", "Transformers are changing AI landscape", "Byzantine fault tolerance in AI"]
queries = ["What is federated learning?", "How do transformers work?"]

# Initial Benchmark
print("Benchmark before mutation:", benchmark_retrieval(documents, queries))

# Apply mutations and re-test
mutate_database('add', documents)
print("Benchmark after addition:", benchmark_retrieval(documents, queries))

mutate_database('remove', documents)
print("Benchmark after removal:", benchmark_retrieval(documents, queries))

mutate_database('corrupt', documents)
print("Benchmark after corruption:", benchmark_retrieval(documents, queries))
