# config.py
MODEL_CONFIGS = {
    "sentence-transformers/all-mpnet-base-v2": {"embedding_size": 768},
    "sentence-transformers/all-MiniLM-L6-v2": {"embedding_size": 384}
}
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DB_URI = "mongodb://localhost:32768/?directConnection=true"
DB_NAME = "rag_db"
COLLECTION_NAME = "hotpotqa_docs"
K = 3
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"