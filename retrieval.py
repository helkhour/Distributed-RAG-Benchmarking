from config import K
from system_evaluation import SystemEvaluator
import time

LOG_LIMIT = 4
log_counter = 0

def retrieve_top_k(query_embedding, collection, num_candidates=100):
    """Retrieve top K documents based on vector similarity with timing."""
    global log_counter
    evaluator = SystemEvaluator()

    if log_counter < LOG_LIMIT:
        evaluator.log_resources(f"Before Retrieval {log_counter + 1}")
    
    # Time vector similarity search
    start_time = time.time()
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "index": "vector_index",
                "limit": K,
                "numCandidates": num_candidates
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "question_ids": 1,
                "source": 1
            }
        }
    ])
    results_list = list(results)
    search_duration = time.time() - start_time
    
    if log_counter < LOG_LIMIT:
        evaluator.log_resources(f"After Retrieval {log_counter + 1}")
        log_counter += 1

    
    
    return results_list, {"vector_search": search_duration}