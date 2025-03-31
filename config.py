# config.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384  # Confirmed from all-MiniLM-L6-v2
# config.py
DB_URI = "mongodb://localhost:27777/?directConnection=true"  # Updated port
DB_NAME = "rag_db"
COLLECTION_NAME = "hotpotqa_docs"
K = 5  # Top k results to retrieve
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"