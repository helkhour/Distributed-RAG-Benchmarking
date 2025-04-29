MODEL_CONFIGS = {
    # Same Parameters, Different Dimensions (using mxbai - 335M parms - flexible output size)
    "mixedbread-ai/mxbai-embed-large-v1-256": {"output_dimension": 256, "base_model": "mixedbread-ai/mxbai-embed-large-v1"},
    "mixedbread-ai/mxbai-embed-large-v1-512": {"output_dimension": 512, "base_model": "mixedbread-ai/mxbai-embed-large-v1"},
    "mixedbread-ai/mxbai-embed-large-v1-1024": {"output_dimension": 1024, "base_model": "mixedbread-ai/mxbai-embed-large-v1"},

    # Different Parameters, Same Dimension (384)
    "sentence-transformers/all-MiniLM-L6-v2": {"output_dimension": 384},  # ~22.7M
    "intfloat/e5-small-v2": {"output_dimension": 384},                  # ~33M
    "thenlper/gte-base-384": {"output_dimension": 384, "base_model": "thenlper/gte-base"},  # ~110M, projected

    # Same Parameters, Same Dimension (768)
    "sentence-transformers/all-mpnet-base-v2": {"output_dimension": 768},  # ~110M
    "BAAI/bge-base-en-v1.5": {"output_dimension": 768},                  # ~110M
    "thenlper/gte-base": {"output_dimension": 768},                      # ~110M
}

DB_URI = "mongodb://localhost:32768/?directConnection=true"
DB_NAME = "rag_db"
K = 3
DATASET_NAME = "rungalileo/ragbench"
SUBSET_NAME = "hotpotqa"
