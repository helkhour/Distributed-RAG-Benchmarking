# retrieval.py
from pymongo import MongoClient
from config import DB_URI, DB_NAME, COLLECTION_NAME, K

def retrieve_top_k(query_embedding, collection):
    """Retrieve top k documents based on vector similarity."""
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "limit": K
            }
        },
        {"$project": {"text": 1, "_id": 0}}  # Return only the document text
    ])
    return list(results)