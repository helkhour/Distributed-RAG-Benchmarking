MODEL_CONFIGS = {
    "meta-llama/Meta-Llama-3.1-8B": {
        "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "embedding_size": 4096,
        "parameters": 8_000_000_000
    },
}
DB_URI = "mongodb://localhost:27017/?directConnection=true"
DB_NAME = "rag_db"
COLLECTION_NAME = "hotpotqa_docs"
K = 3
NUM_CANDIDATES = 30
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"