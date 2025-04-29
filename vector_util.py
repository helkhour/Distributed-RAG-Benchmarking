from config import MODEL_CONFIG, DB_CONFIG
import time

def setup_vector_index(collection):
    try:
        # Drop existing vector index
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
                            "dimensions": MODEL_CONFIG["output_dimension"],
                            "similarity": "cosine"
                        }
                    }
                }
            }
        })

        # Wait for index to be ready
        print("Waiting for vector index to be ready...")
        start_time = time.time()
        while time.time() - start_time < DB_CONFIG["index_timeout_s"]:
            indexes = list(collection.list_search_indexes())
            vector_index = next((idx for idx in indexes if idx["name"] == "vector_index"), None)
            if vector_index and vector_index.get("status") == "READY":
                print("Vector index is now ready!")
                return
            time.sleep(1)
        raise TimeoutError("Vector index creation timed out.")
    except Exception as e:
        print(f"Error setting up vector index: {e}")
        raise