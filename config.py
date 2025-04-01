# Configuration settings for the RAG pipeline

# Model settings
MODEL_CONFIG = {
    "name": "sentence-transformers/all-mpnet-base-v2",
    "embedding_size": 768
}

# Database settings
DB_CONFIG = {
    "uri": "mongodb://localhost:32768/?directConnection=true",
    "name": "rag_db",
    "collection": "hotpotqa_docs",
    "timeout_ms": 5000,
    "index_timeout_s": 300
}

# Retrieval settings
RETRIEVAL_CONFIG = {
    "k": 3,  # Top k results
    "candidate_multiplier": 100  # Multiplier for numCandidates
}

# Dataset settings
DATASET_CONFIG = {
    "name": "rungalileo/ragbench",
    "subset": "hotpotqa"
}