# main.py
from data_loader import load_and_store_data
from evaluation import evaluate_retrieval_performance
from embedding_utils import EmbeddingGenerator
from config import MODEL_CONFIGS

def run_study(model_name, embedding_size):
    """Run a single study for a model."""
    print(f"\n=== Evaluating Model: {model_name} ===")
    embedding_generator = EmbeddingGenerator(model_name, embedding_size)
    collection, dataset, embedding_storage_time = load_and_store_data(
        limit=40, embedding_generator=embedding_generator, embedding_size=embedding_size
    )

    print("\nRunning sequential evaluation...")
    sequential_metrics = evaluate_retrieval_performance(
        dataset, collection, embedding_generator, max_workers=1
    )
    return embedding_storage_time, sequential_metrics

def summarize_results(model_name, embedding_time, sequential_metrics):
    config = MODEL_CONFIGS[model_name]

    print(f"\n=== Summary for Model: {model_name} ===")
    print(f"Embedding Time (s): {embedding_time:.2f}")
    print(f"Latency (s/query): {sequential_metrics['avg_latency']:.4f}")
    print(f"Throughput (q/s): {sequential_metrics['throughput']:.2f}")
    print(f"Precision (%): {sequential_metrics['avg_precision'] * 100:.2f}")

def main():
    print("Starting RAG evaluation...")
    models = [
        # Same Parameters, Different Dimensions
        "mixedbread-ai/mxbai-embed-large-v1-256",
        "mixedbread-ai/mxbai-embed-large-v1-512",
        "mixedbread-ai/mxbai-embed-large-v1-1024",
        # Different Parameters, Same Dimension (384)
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/e5-small-v2",
        "thenlper/gte-base-384",
        # Different Parameters, Same Dimension (768)
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-base"
    ]

    for model_name in models:
        embedding_size = MODEL_CONFIGS[model_name]["embedding_size"]
        embedding_time, sequential_metrics = run_study(model_name, embedding_size)
        summarize_results(model_name, embedding_time, sequential_metrics)

if __name__ == "__main__":
    main()