# retrieval.py
from config import K, NUM_CANDIDATES
from system_evaluation import SystemEvaluator

LOG_LIMIT = 4
log_counter = 0

def retrieve_top_k(query_embedding, collection):
    """Retrieve top K documents based on vector similarity."""
    global log_counter
    evaluator = SystemEvaluator()

    if log_counter < LOG_LIMIT:
        evaluator.log_resources(f"Before Retrieval {log_counter + 1}")
    
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "index": "vector_index",
                "limit": K,
                "numCandidates": NUM_CANDIDATES
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "question_id": 1,
                "source": 1
            }
        }
    ])
    results_list = list(results)
    
    if log_counter < LOG_LIMIT:
        evaluator.log_resources(f"After Retrieval {log_counter + 1}")
        log_counter += 1
    
    return results_list