from data_loader import load_and_store_data
from evaluation import evaluate_retrieval_performance
from embedding_utils import EmbeddingGenerator
from config import MODEL_CONFIGS
from system_evaluation import SystemEvaluator
import logging

def run_study(model_name, embedding_size):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    evaluator = SystemEvaluator()
    logger.info(f"\n=== Evaluating Model: {model_name} ===")
    
    results = {}
    for mode in ["sequential", "batch"]:
        logger.info(f"Running {mode} embedding mode")
        quantize = "Llama-3.1" in model_name
        embedding_generator = EmbeddingGenerator(model_name, embedding_size, quantize=quantize)
        
        evaluator.start_monitoring()
        collection, dataset, data_timings, data_stats = load_and_store_data(
            limit=None, embedding_generator=embedding_generator, embedding_size=embedding_size, use_batch=(mode == "batch")
        )
        data_duration, data_cpu_delta = evaluator.end_monitoring(f"Data Load ({mode})")
        
        logger.info(f"\nRunning evaluation (30 candidates, {mode})...")
        evaluator.start_monitoring()
        metrics_30 = evaluate_retrieval_performance(
            dataset, collection, embedding_generator, max_workers=4, num_candidates=30
        )
        #### parallelizing with 4 workers. can try with 1 worker too for full frawework !!! 
        eval_duration_30, eval_cpu_delta_30 = evaluator.end_monitoring(f"Evaluation (30 candidates, {mode})")
        
        logger.info(f"\nRunning evaluation (100 candidates, {mode})...")
        evaluator.start_monitoring()
        metrics_100 = evaluate_retrieval_performance(
            dataset, collection, embedding_generator, max_workers=4, num_candidates=100
        )
        eval_duration_100, eval_cpu_delta_100 = evaluator.end_monitoring(f"Evaluation (100 candidates, {mode})")
        
        if metrics_30["avg_precision"] == 0 or metrics_100["avg_precision"] == 0:
            logger.warning(f"Zero precision detected for {model_name} ({mode}). Check embedding dimensions and vector index.")
        
        results[mode] = {
            "data_timings": data_timings,
            "data_stats": data_stats,
            "metrics_30": metrics_30,
            "metrics_100": metrics_100,
            "eval_duration_30": eval_duration_30,
            "eval_duration_100": eval_duration_100
        }
    
    return results

def summarize_results(model_name, results):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    config = MODEL_CONFIGS[model_name]
    embedding_size = config["embedding_size"]
    parameters = config.get("parameters", "Unknown")

    for mode in ["sequential", "batch"]:
        timings = results[mode]
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
            "Document Collection": data_timings["doc_collection"],
            "Database Clearing": data_timings["db_clearing"],
            f"{mode.capitalize()} Embedding": data_timings[f"{mode}_embedding"],
            "Embedding Generation (Document)": data_timings["embedding_generation"],
            "Batch Document Storage": data_timings["batch_storage"],
            "Document Indexing": data_timings["document_indexing"],
            "Query Preprocessing (30 candidates)": metrics_30["query_timings"]["query_preprocessing"],
            "Query Encoding (30 candidates)": metrics_30["query_timings"]["query_encoding"],
            "Vector Similarity Search (30 candidates)": metrics_30["query_timings"]["vector_search"],
            "Evaluation Overhead (30 candidates)": metrics_30["query_timings"]["evaluation_overhead"],
            "Query Preprocessing (100 candidates)": metrics_100["query_timings"]["query_preprocessing"],
            "Query Encoding (100 candidates)": metrics_100["query_timings"]["query_encoding"],
            "Vector Similarity Search (100 candidates)": metrics_100["query_timings"]["vector_search"],
            "Evaluation Overhead (100 candidates)": metrics_100["query_timings"]["evaluation_overhead"]
        }

        # Calculate total time
        total_time = sum(all_timings.values())
        
        # Calculate proportions
        proportions = {key: (value / total_time * 100) if total_time > 0 else 0.0 for key, value in all_timings.items()}

        logger.info(f"\n=== Summary for Model: {model_name} ({mode.capitalize()} Embedding) ===")
        logger.info(f"Model Specs:")
        if parameters != "Unknown":
            logger.info(f"  Parameters: {parameters:,} (~{parameters // 1_000_000}M)")
        else:
            logger.info(f"  Parameters: {parameters}")
        logger.info(f"  Embedding Size: {embedding_size}")
        
        logger.info(f"\nDataset and Database Statistics:")
        logger.info(f"{'Metric':<40} {'Value':<20}")
        logger.info("-" * 60)
        logger.info(f"{'HotpotQA Entries':<40} {data_stats['hotpotqa_entries']:<20}")
        logger.info(f"{'PubMedQA Entries':<40} {data_stats['pubmedqa_entries']:<20}")
        logger.info(f"{'Total Dataset Entries':<40} {data_stats['total_entries']:<20}")
        logger.info(f"{'HotpotQA Documents':<40} {data_stats['hotpotqa_docs']:<20}")
        logger.info(f"{'PubMedQA Documents':<40} {data_stats['pubmedqa_docs']:<20}")
        logger.info(f"{'Total Documents':<40} {data_stats['total_docs']:<20}")
        logger.info(f"{'HotpotQA Size (MB)':<40} {data_stats['hotpotqa_size_mb']:<20.2f}")
        logger.info(f"{'PubMedQA Size (MB)':<40} {data_stats['pubmedqa_size_mb']:<20.2f}")
        logger.info(f"{'Total Dataset Size (MB)':<40} {data_stats['total_size_mb']:<20.2f}")
        logger.info(f"{'Documents Stored (Batch)':<40} {data_stats['docs_stored_batch']:<20}")
        logger.info(f"{'Database Entries':<40} {data_stats['db_entries']:<20}")
        logger.info(f"{'Database Size (MB)':<40} {metrics_30['db_size_mb']:<20.2f}")
        logger.info(f"{'Total Queries Evaluated':<40} {metrics_30['total_queries']:<20}")
        logger.info(f"{'Embedding Mode':<40} {data_stats['embedding_mode']:<20}")
        
        logger.info(f"\nTiming Breakdown (Total Time: {total_time:.2f}s):")
        logger.info(f"{'Process':<40} {'Time (s)':<12} {'Proportion (%)':<15}")
        logger.info("-" * 67)
        for key, duration in all_timings.items():
            logger.info(f"{key:<40} {duration:<12.2f} {proportions[key]:<15.2f}")
        
        logger.info(f"\nEvaluation Metrics (30 candidates):")
        logger.info(f"  Latency (s/query): {metrics_30['avg_latency']:.4f}")
        logger.info(f"  Throughput (q/s): {metrics_30['throughput']:.2f}")
        logger.info(f"  Precision (%): {metrics_30['avg_precision'] * 100:.2f}")
        logger.info(f"\nEvaluation Metrics (100 candidates):")
        logger.info(f"  Latency (s/query): {metrics_100['avg_latency']:.4f}")
        logger.info(f"  Throughput (q/s): {metrics_100['throughput']:.2f}")
        logger.info(f"  Precision (%): {metrics_100['avg_precision'] * 100:.2f}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting RAG evaluation...")
    models = [
        "meta-llama/Meta-Llama-3.1-8B", ok
        "mixedbread-ai/mxbai-embed-large-v1-256", ok
        "mixedbread-ai/mxbai-embed-large-v1-512",
        "mixedbread-ai/mxbai-embed-large-v1-1024", ok
        "sentence-transformers/all-MiniLM-L6-v2", ok 
        "intfloat/e5-small-v2", ok 
        "thenlper/gte-base-384", no 
        "sentence-transformers/all-mpnet-base-v2", ok
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-base"   ok 
    ]

    for model_name in models:
        embedding_size = MODEL_CONFIGS[model_name]["embedding_size"]
        results = run_study(model_name, embedding_size)
        summarize_results(model_name, results)

if __name__ == "__main__":
    main()