from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import time
from config import DB_URI, DB_NAME, COLLECTION_NAME

def get_db_connection(max_retries=3, retry_delay=5):
    """Establish and return a MongoDB collection connection with retries."""
    for attempt in range(max_retries):
        try:
            client = MongoClient(DB_URI, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            print("Connected to MongoDB successfully!")
            db = client[DB_NAME]
            return db[COLLECTION_NAME]
        except ServerSelectionTimeoutError as e:
            print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Ensure MongoDB is running and accessible at", DB_URI)
                raise

def setup_vector_index(collection, embedding_size, timeout=300, max_interval=5):
    """Create or update the vector index for the collection."""
    try:
        start_time = time.time()
        indexes = list(collection.list_search_indexes())
        for index in indexes:
            if index["name"] == "vector_index":
                collection.drop_search_index(index["name"])
                print("Existing 'vector_index' dropped.")

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
        interval = 0.5
        attempt = 0
        while time.time() - start_time < timeout:
            indexes = list(collection.list_search_indexes())
            vector_index = next((idx for idx in indexes if idx["name"] == "vector_index"), None)
            if vector_index and vector_index.get("status") == "READY":
                print("Vector index is now ready!")
                indexing_duration = time.time() - start_time
                print(f"Document Indexing Duration: {indexing_duration:.2f}s")
                return indexing_duration
            sleep_time = min(interval * (1.5 ** attempt), max_interval)
            time.sleep(sleep_time)
            attempt += 1
        raise TimeoutError("Vector index creation timed out.")
    except Exception as e:
        print(f"Error setting up vector index: {e}")
        raise