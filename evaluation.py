import time
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from config import DB_URI, DB_NAME
from retrieval import retrieve_top_k
import logging

def process_query(entry, collection, embedding_generator, num_candidates):
    """Process a single query and return metrics with timing."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    query = entry["question"]
    start_time = time.time()
    query_embedding_list, embed_timings = embedding_generator.generate_embedding(query)
    query_embedding = query_embedding_list[0]
    results, search_timings = retrieve_top_k(query_embedding, collection, num_candidates)
    latency = time.time() - start_time

    relevant_docs = entry["documents"]
    retrieved_texts = [r["text"] for r in results]
    relevant_count = sum(any(retrieved.strip() == ref.strip() for ref in relevant_docs) for retrieved in retrieved_texts)

    logger.debug(f"Query: {query}, Latency: {latency:.4f}s, Relevant: {relevant_count}/{len(retrieved_texts)}")
    
    timings = {
        "query_preprocessing": embed_timings["query_preprocessing"],
        "query_encoding": embed_timings["query_encoding"],
        "vector_search": search_timings["vector_search"]
    }
    
    return {
        "latency": latency,
        "results": results,
        "retrieved_texts": retrieved_texts,
        "relevant_count": relevant_count,
        "query": query,
        "has_results": bool(results),
        "timings": timings
    }

def evaluate_retrieval_performance(dataset, collection, embedding_generator, max_workers=4, num_candidates=100):
    """Evaluate retrieval performance with parallel queries and timing."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    total_queries = len(dataset)
    queries_with_results = 0
    total_relevant = 0
    total_retrieved = 0
    total_latency = 0
    query_timings = {
        "query_preprocessing": 0.0,
        "query_encoding": 0.0,
        "vector_search": 0.0,
        "evaluation_overhead": 0.0
    }

    logger.info(f"Starting evaluation with {total_queries} queries, {max_workers} workers, {num_candidates} candidates")
    start_total_time = time.time()
    
    # Time parallel processing
    start_overhead = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda entry: process_query(entry, collection, embedding_generator, num_candidates), dataset
        ))
    query_timings["evaluation_overhead"] = time.time() - start_overhead

    for res in results:
        total_latency += res["latency"]
        if res["has_results"]:
            queries_with_results += 1
        total_relevant += res["relevant_count"]
        total_retrieved += len(res["retrieved_texts"])
        for key in query_timings:
            if key in res["timings"]:
                query_timings[key] += res["timings"][key]

    total_time = time.time() - start_total_time

    retrieval_success = queries_with_results / total_queries if total_queries > 0 else 0
    avg_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    throughput = total_queries / total_time if total_time > 0 else 0

    client = MongoClient(DB_URI)
    db = client[DB_NAME]
    stats = db.command("dbStats")
    db_size_mb = stats["dataSize"] / (1024 * 1024)
    doc_count = collection.count_documents({})

    logger.info("\n=== Retrieval Performance Summary ===")
    logger.info(f"Database Size: {db_size_mb:.2f} MB ({doc_count} documents)")
    logger.info(f"Retrieval Success: {retrieval_success:.2%} ({queries_with_results}/{total_queries} queries)")
    logger.info(f"Average Precision: {avg_precision:.2%} ({total_relevant}/{total_retrieved} docs)")
    logger.info(f"Average Latency: {avg_latency:.4f} seconds/query")
    logger.info(f"Throughput: {throughput:.2f} queries/second (with {max_workers} workers, {num_candidates} candidates)")

    return {
        "retrieval_success": retrieval_success,
        "avg_precision": avg_precision,
        "avg_latency": avg_latency,
        "throughput": throughput,
        "db_size_mb": db_size_mb,
        "doc_count": doc_count,
        "total_time": total_time,
        "query_timings": query_timings,
        "total_queries": total_queries
    }