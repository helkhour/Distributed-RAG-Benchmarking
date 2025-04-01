# vector_util.py
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from config import MODEL_NAME, EMBEDDING_SIZE, DB_URI, DB_NAME, COLLECTION_NAME
import time

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
        
        # Drop the existing vector index if it exists
        indexes = list(collection.list_search_indexes())
        for index in indexes:
            if index["name"] == "vector_index":
                collection.drop_search_index(index["name"])
                print("Existing 'vector_index' dropped.")

        # Create the new vector index
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

        # Wait for the index to become ready
        print("Waiting for vector index to be ready...")
        start_time = time.time()
        timeout = 300  # Timeout after 5 minutes (in seconds)
    
        while True:
            indexes = list(collection.list_search_indexes())            
            # Find the vector_index
            vector_index = next((index for index in indexes if index["name"] == "vector_index"), None)

            if vector_index:                
                if vector_index.get("status") == "READY":
                    print("Vector index is now ready!")
                    break
                else:
                    print(f"Index is not ready yet. Current state: {vector_index.get('status')}")
            else:
                print("No vector index found.")
            # Sleep for 1 second before checking again
            time.sleep(1)
        return collection
    
    except ServerSelectionTimeoutError as e:
        print(f"Connection error: {e}")
        print("Ensure MongoDB is running in Docker and accessible at", DB_URI)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise