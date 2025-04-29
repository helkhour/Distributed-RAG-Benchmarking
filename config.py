MODEL_CONFIGS = {
    # Same Parameters, Different Dimensions (using mxbai - 335M parms - flexible output size)
    "mixedbread-ai/mxbai-embed-large-v1-256": {"embedding_size": 256, "base_model": "mixedbread-ai/mxbai-embed-large-v1"},
    "mixedbread-ai/mxbai-embed-large-v1-512": {"embedding_size": 512, "base_model": "mixedbread-ai/mxbai-embed-large-v1"},
    "mixedbread-ai/mxbai-embed-large-v1-1024": {"embedding_size": 1024, "base_model": "mixedbread-ai/mxbai-embed-large-v1"},

    # Different Parameters, Same Dimension (384)
    "sentence-transformers/all-MiniLM-L6-v2": {"embedding_size": 384},  # ~22.7M
    "intfloat/e5-small-v2": {"embedding_size": 384},                  # ~33M
    "thenlper/gte-base-384": {"embedding_size": 384, "base_model": "thenlper/gte-base"},  # ~110M, projected

    # Same Parameters, Same Dimension (768)
    "sentence-transformers/all-mpnet-base-v2": {"embedding_size": 768},  # ~110M
    "BAAI/bge-base-en-v1.5": {"embedding_size": 768},                  # ~110M
    "thenlper/gte-base": {"embedding_size": 768},                      # ~110M
}

DB_URI = "mongodb://localhost:32768/?directConnection=true"
DB_NAME = "rag_db"
K = 3
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"
