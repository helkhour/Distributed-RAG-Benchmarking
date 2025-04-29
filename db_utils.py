# db_utils.py
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import time
from config import DB_URI, DB_NAME, COLLECTION_NAME

def get_db_connection():
    """Establish and return a MongoDB collection connection."""
    try:
        client = MongoClient(DB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Test connection
        print("Connected to MongoDB successfully!")
        db = client[DB_NAME]
        return db[COLLECTION_NAME]
    except ServerSelectionTimeoutError as e:
        print(f"Connection error: {e}")
        print("Ensure MongoDB is running and accessible at", DB_URI)
        raise

def setup_vector_index(collection, embedding_size):
    """Create or update the vector index for the collection."""
    try:
        # Drop existing vector index if it exists  # Delete this for first run on machine tho
        indexes = list(collection.list_search_indexes())
        for index in indexes:
            if index["name"] == "vector_index":
                collection.drop_search_index(index["name"])
                print("Existing 'vector_index' dropped.")

        # Create new vector index
        collection.create_search_index({
            "name": "vector_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "type": "knnVector",
                            "dimensions": embedding_size,
                            "similarity": "cosine"
                        }
                    }
                }
            }
        })

        print("Waiting for vector index to be ready...")
        start_time = time.time()
        timeout = 300  # 5 minutes
        while time.time() - start_time < timeout:
            indexes = list(collection.list_search_indexes())
            vector_index = next((idx for idx in indexes if idx["name"] == "vector_index"), None)
            if vector_index and vector_index.get("status") == "READY":
                print("Vector index is now ready!")
                return
            time.sleep(1)                                              # inefficient - need to find something else 
        raise TimeoutError("Vector index creation timed out.")
    except Exception as e:
        print(f"Error setting up vector index: {e}")
        raise