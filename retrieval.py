# retrieval.py
from config import K

def retrieve_top_k(query_embedding, collection):
    """Retrieve top K documents based on vector similarity."""
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "index": "vector_index",
                "limit": K,  # Top K results (e.g., 5)
                "numCandidates": K * 100  # Consider 10x the limit as candidates (e.g., 50)
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