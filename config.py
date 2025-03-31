# config.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384  # Size for all-MiniLM-L6-v2
DB_URI = "mongodb://localhost:32768/?directConnection=true"  # Adjust port when changing model !!
DB_NAME = "rag_db"
COLLECTION_NAME = "hotpotqa_docs"
K = 3  # Top k results to retrieve
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"