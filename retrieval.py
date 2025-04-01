from config import RETRIEVAL_CONFIG

def retrieve_top_k(query_embedding, collection, k=RETRIEVAL_CONFIG["k"]):
    try:
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "index": "vector_index",
                    "limit": k,
                    "numCandidates": k * RETRIEVAL_CONFIG["candidate_multiplier"]
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
        return list(results)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []