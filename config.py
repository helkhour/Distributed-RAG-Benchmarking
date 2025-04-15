MODEL_CONFIGS = {
    "sentence-transformers/all-MiniLM-L6-v2": {"embedding_size": 384, "parameters": 22_700_000},
    "meta-llama/Meta-Llama-3.1-8B": {"embedding_size": 4096, "parameters": 8_000_000_000},
}
DB_URI = "mongodb://localhost:27017/?directConnection=true"  # Update if using Atlas
DB_NAME = "rag_db"
COLLECTION_NAME = "hotpotqa_docs"
K = 3
NUM_CANDIDATES = 30
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"