# main.py
from data_loader import load_and_store_data
from evaluation import evaluate_retrieval_performance
from embedding_utils import EmbeddingGenerator
from config import MODEL_CONFIGS, DEFAULT_MODEL

def main():
    print("Starting RAG evaluation with full dataset and model comparison...")

    # Study 2: Compare embedding models
    models = ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"]
    for model_name in models:
        print(f"\n=== Evaluating Model: {model_name} ===")
        embedding_size = MODEL_CONFIGS[model_name]["embedding_size"]
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator(model_name)

        # Load and store full dataset with this model
        collection, dataset, embedding_storage_time = load_and_store_data(
            limit=None, embedding_generator=embedding_generator, embedding_size=embedding_size
        )

        # Study 1: Sequential evaluation (baseline)
        print("\nRunning sequential evaluation...")
        sequential_metrics = evaluate_retrieval_performance(
            dataset, collection, embedding_generator, max_workers=1
        )

        # Study 2: Parallel evaluation to test query capacity
        worker_counts = [2, 4, 8, 16]  # Test with increasing parallelism
        for workers in worker_counts:
            print(f"\nRunning parallel evaluation with {workers} workers...")
            parallel_metrics = evaluate_retrieval_performance(
                dataset, collection, embedding_generator, max_workers=workers
            )

        # Print embedding time
        print(f"Time to embed and store data using {model_name} (embedding size: {embedding_size}): {embedding_storage_time:.2f} seconds")

if __name__ == "__main__":
    main()

