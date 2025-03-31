# vector_util.py
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from config import MODEL_NAME, EMBEDDING_SIZE, DB_URI, DB_NAME, COLLECTION_NAME

model = SentenceTransformer(MODEL_NAME)

def generate_embedding(text):
    return model.encode(text).tolist()

def setup_vector_index():
    try:
        client = MongoClient(DB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        result = client.admin.command('ping')
        print("Ping result:", result)  # Should print {'ok': 1.0}
        print("Connected to MongoDB successfully!")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        collection.create_search_index({
            "name": "vector_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "type": "knnVector",
                            "dimensions": EMBEDDING_SIZE,
                            "similarity": "cosine"
                        }
                    }
                }
            }
        })
        
        indexes = list(collection.list_search_indexes())
        if any(index["name"] == "vector_index" for index in indexes):
            print("Vector index created successfully!")
        else:
            print("Warning: Vector index not found.")
        
        return collection
    except ServerSelectionTimeoutError as e:
        print(f"Connection error: {e}")
        print("Ensure MongoDB is running in Docker and accessible at", DB_URI)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise