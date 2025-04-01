# evaluation.py
import time
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from config import DB_URI, DB_NAME
from retrieval import retrieve_top_k

def process_query(entry, collection, embedding_generator):
    """Process a single query and return metrics."""
    query = entry["question"]
    start_time = time.time()
    query_embedding = embedding_generator.generate_embedding(query)
    results = retrieve_top_k(query_embedding, collection)
    latency = time.time() - start_time

    relevant_docs = entry["documents"]
    retrieved_texts = [r["text"] for r in results]
    relevant_count = sum(text in relevant_docs for text in retrieved_texts)

    return {
        "latency": latency,
        "results": results,
        "retrieved_texts": retrieved_texts,
        "relevant_count": relevant_count,
        "query": query,
        "has_results": bool(results)
    }

def evaluate_retrieval_performance(dataset, collection, embedding_generator, max_workers=1):
    """Evaluate retrieval performance with optional parallel queries."""
    total_queries = len(dataset)
    queries_with_results = 0
    total_relevant = 0
    total_retrieved = 0
    total_latency = 0

    start_total_time = time.time()
    if max_workers == 1:
        # Sequential processing
        results = [process_query(entry, collection, embedding_generator) for entry in dataset]
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda entry: process_query(entry, collection, embedding_generator), dataset
            ))

    # Aggregate results
    for res in results:
        total_latency += res["latency"]
        if res["has_results"]:
            queries_with_results += 1
        total_relevant += res["relevant_count"]
        total_retrieved += len(res["retrieved_texts"])

        # Print query details
        #print(f"\nQuery: {res['query']}")
        #print(f"Top {len(res['results'])} results (Latency: {res['latency']:.4f} seconds):")
        #for i, result in enumerate(res["results"], 1):
        #    print(f"{i}. {result['text'][:100]}...")
        #print("Retrieved relevant? (manual check):")
        #for i, text in enumerate(res["retrieved_texts"], 1):
        #    print(f"{i}. {text in dataset[results.index(res)]['documents']}")

    total_time = time.time() - start_total_time

    # Calculate metrics
    retrieval_success = queries_with_results / total_queries if total_queries > 0 else 0
    avg_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    throughput = total_queries / total_time if total_time > 0 else 0

    # Database stats
    client = MongoClient(DB_URI)
    db = client[DB_NAME]
    stats = db.command("dbStats")
    db_size_mb = stats["dataSize"] / (1024 * 1024)
    doc_count = collection.count_documents({})

    # Print summary
    print("\n=== Retrieval Performance Summary ===")
    print(f"Database Size: {db_size_mb:.2f} MB ({doc_count} documents)")
    print(f"Retrieval Success: {retrieval_success:.2%} ({queries_with_results}/{total_queries} queries returned results)")
    print(f"Average Precision: {avg_precision:.2%} ({total_relevant}/{total_retrieved} relevant documents retrieved)")
    print(f"Average Latency: {avg_latency:.4f} seconds/query")
    print(f"Throughput: {throughput:.2f} queries/second (with {max_workers} workers)")

    return {
        "retrieval_success": retrieval_success,
        "avg_precision": avg_precision,
        "avg_latency": avg_latency,
        "throughput": throughput,
        "db_size_mb": db_size_mb,
        "doc_count": doc_count,
        "total_time": total_time
    }