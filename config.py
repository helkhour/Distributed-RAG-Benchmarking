# config.py
MODEL_CONFIGS = {
    # Same Parameters, Different Dimensions (using mxbai - 335M parms - flexible output size)
    "mixedbread-ai/mxbai-embed-large-v1-256": {"embedding_size": 256, "base_model": "mixedbread-ai/mxbai-embed-large-v1", "parameters": 335_000_000},
    "mixedbread-ai/mxbai-embed-large-v1-512": {"embedding_size": 512, "base_model": "mixedbread-ai/mxbai-embed-large-v1", "parameters": 335_000_000},
    "mixedbread-ai/mxbai-embed-large-v1-1024": {"embedding_size": 1024, "base_model": "mixedbread-ai/mxbai-embed-large-v1", "parameters": 335_000_000},
    # Different Parameters, Same Dimension (384)
    "sentence-transformers/all-MiniLM-L6-v2": {"embedding_size": 384, "parameters": 22_700_000},
    "intfloat/e5-small-v2": {"embedding_size": 384, "parameters": 33_000_000},
    "thenlper/gte-base-384": {"embedding_size": 384, "base_model": "thenlper/gte-base", "parameters": 110_000_000},
    # Same Parameters, Same Dimension (768)
    "sentence-transformers/all-mpnet-base-v2": {"embedding_size": 768, "parameters": 110_000_000},
    "BAAI/bge-base-en-v1.5": {"embedding_size": 768, "parameters": 110_000_000},
    "thenlper/gte-base": {"embedding_size": 768, "parameters": 110_000_000},
}

DB_URI = "mongodb://localhost:32768/?directConnection=true"
DB_NAME = "rag_db"
COLLECTION_NAME = "hotpotqa_docs"
K = 3
NUM_CANDIDATES = 30
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"