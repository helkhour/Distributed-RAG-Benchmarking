from data_loader import load_and_store_data
from evaluation import evaluate_retrieval_performance
from embedding_utils import EmbeddingGenerator
from config import MODEL_CONFIGS
from system_evaluation import SystemEvaluator

def run_study(model_name, embedding_size):
    evaluator = SystemEvaluator()
    print(f"\n=== Evaluating Model: {model_name} ===")
    
    evaluator.log_resources("Before Data Load")
    quantize = "Llama-3.1" in model_name
    embedding_generator = EmbeddingGenerator(model_name, embedding_size, quantize=quantize)
    
    evaluator.start_monitoring()
    collection, dataset, data_timings, data_stats = load_and_store_data(
        limit=None, embedding_generator=embedding_generator, embedding_size=embedding_size
    )
    data_duration, data_cpu_delta = evaluator.end_monitoring("Data Load")
    
    print("\nRunning evaluation (30 candidates)...")
    evaluator.start_monitoring()
    metrics_30 = evaluate_retrieval_performance(
        dataset, collection, embedding_generator, max_workers=4, num_candidates=30
    )
    eval_duration_30, eval_cpu_delta_30 = evaluator.end_monitoring("Evaluation (30 candidates)")
    
    print("\nRunning evaluation (100 candidates)...")
    evaluator.start_monitoring()
    metrics_100 = evaluate_retrieval_performance(
        dataset, collection, embedding_generator, max_workers=4, num_candidates=100
    )
    eval_duration_100, eval_cpu_delta_100 = evaluator.end_monitoring("Evaluation (100 candidates)")
    
    return {
        "data_timings": data_timings,
        "data_stats": data_stats,
        "metrics_30": metrics_30,
        "metrics_100": metrics_100,
        "eval_duration_30": eval_duration_30,
        "eval_duration_100": eval_duration_100
    }

def summarize_results(model_name, timings):
    config = MODEL_CONFIGS[model_name]
    embedding_size = config["embedding_size"]
    parameters = config.get("parameters", "Unknown")

    data_timings = timings["data_timings"]
    data_stats = timings["data_stats"]
    metrics_30 = timings["metrics_30"]
    metrics_100 = timings["metrics_100"]
    eval_duration_30 = timings["eval_duration_30"]
    eval_duration_100 = timings["eval_duration_100"]

    # Aggregate all timings
    all_timings = {
        "Database Connection": data_timings["db_connection"],
        "Dataset Loading": data_timings["dataset_load"],
        "Document Preparation": data_timings["doc_preparation"],
        "Database Clearing": data_timings["db_clearing"],
        "Database Clearing (before batch)": data_timings["db_clearing_batch"],
        "Embedding Generation (Document)": data_timings["embedding_generation"],
        "Sequential Document Storage": data_timings["sequential_storage"],
        "Batch Document Storage": data_timings["batch_storage"],
        "Document Indexing": data_timings["document_indexing"],
        "Query Preprocessing (30 candidates)": metrics_30["query_timings"]["query_preprocessing"],
        "Query Encoding (30 candidates)": metrics_30["query_timings"]["query_encoding"],
        "Vector Similarity Search (30 candidates)": metrics_30["query_timings"]["vector_search"],
        "Evaluation Overhead (30 candidates)": metrics_30["query_timings"]["evaluation_overhead"],
        "Query Preprocessing (100 candidates)": metrics_30["query_timings"]["query_preprocessing"],
        "Query Encoding (100 candidates)": metrics_100["query_timings"]["query_encoding"],
        "Vector Similarity Search (100 candidates)": metrics_100["query_timings"]["vector_search"],
        "Evaluation Overhead (100 candidates)": metrics_100["query_timings"]["evaluation_overhead"]
    }

    # Calculate total time
    total_time = sum(all_timings.values())
    
    # Calculate proportions
    proportions = {key: (value / total_time * 100) if total_time > 0 else 0.0 for key, value in all_timings.items()}

    # Calculate storage time difference
    sequential_time = all_timings["Sequential Document Storage"]
    batch_time = all_timings["Batch Document Storage"]
    storage_time_diff = sequential_time - batch_time
    storage_speedup = (sequential_time / batch_time) if batch_time > 0 else float('inf')

    print(f"\n=== Summary for Model: {model_name} ===")
    print(f"Model Specs:")
    if parameters != "Unknown":
        print(f"  Parameters: {parameters:,} (~{parameters // 1_000_000}M)")
    else:
        print(f"  Parameters: {parameters}")
    print(f"  Embedding Size: {embedding_size}")
    
    print(f"\nDataset and Database Statistics:")
    print(f"{'Metric':<40} {'Value':<20}")
    print("-" * 60)
    print(f"{'HotpotQA Entries':<40} {data_stats['hotpotqa_entries']:<20}")
    print(f"{'PubMedQA Entries':<40} {data_stats['pubmedqa_entries']:<20}")
    print(f"{'Total Dataset Entries':<40} {data_stats['total_entries']:<20}")
    print(f"{'HotpotQA Documents':<40} {data_stats['hotpotqa_docs']:<20}")
    print(f"{'PubMedQA Documents':<40} {data_stats['pubmedqa_docs']:<20}")
    print(f"{'Total Documents':<40} {data_stats['total_docs']:<20}")
    print(f"{'HotpotQA Size (MB)':<40} {data_stats['hotpotqa_size_mb']:<20.2f}")
    print(f"{'PubMedQA Size (MB)':<40} {data_stats['pubmedqa_size_mb']:<20.2f}")
    print(f"{'Total Dataset Size (MB)':<40} {data_stats['total_size_mb']:<20.2f}")
    print(f"{'Documents Stored (Sequential)':<40} {data_stats['docs_stored_sequential']:<20}")
    print(f"{'Documents Stored (Batch)':<40} {data_stats['docs_stored_batch']:<20}")
    print(f"{'Database Entries':<40} {data_stats['db_entries']:<20}")
    print(f"{'Database Size (MB)':<40} {metrics_30['db_size_mb']:<20.2f}")
    print(f"{'Total Queries Evaluated':<40} {metrics_30['total_queries']:<20}")
    
    print(f"\nTiming Breakdown (Total Time: {total_time:.2f}s):")
    print(f"{'Process':<40} {'Time (s)':<12} {'Proportion (%)':<15}")
    print("-" * 67)
    for key, duration in all_timings.items():
        print(f"{key:<40} {duration:<12.2f} {proportions[key]:<15.2f}")
    
    print(f"\nStorage Comparison:")
    print(f"  Sequential Storage Time: {sequential_time:.2f}s")
    print(f"  Batch Storage Time: {batch_time:.2f}s")
    print(f"  Time Difference (Sequential - Batch): {storage_time_diff:.2f}s")
    print(f"  Speedup (Sequential / Batch): {storage_speedup:.2f}x")
    
    print(f"\nEvaluation Metrics (30 candidates):")
    print(f"  Latency (s/query): {metrics_30['avg_latency']:.4f}")
    print(f"  Throughput (q/s): {metrics_30['throughput']:.2f}")
    print(f"  Precision (%): {metrics_30['avg_precision'] * 100:.2f}")
    print(f"Evaluation Metrics (100 candidates):")
    print(f"  Latency (s/query): {metrics_100['avg_latency']:.4f}")
    print(f"  Throughput (q/s): {metrics_100['throughput']:.2f}")
    print(f"  Precision (%): {metrics_100['avg_precision'] * 100:.2f}")

def main():
    print("Starting RAG evaluation...")
    models = [
        "meta-llama/Meta-Llama-3.1-8B",
        "mixedbread-ai/mxbai-embed-large-v1-256",
        "mixedbread-ai/mxbai-embed-large-v1-512",
        "mixedbread-ai/mxbai-embed-large-v1-1024",
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/e5-small-v2",
        "thenlper/gte-base-384",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-base"
    ]

    for model_name in models:
        embedding_size = MODEL_CONFIGS[model_name]["embedding_size"]
        timings = run_study(model_name, embedding_size)
        summarize_results(model_name, timings)

if __name__ == "__main__":
    main()