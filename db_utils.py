from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
import time
import logging
import os
from config import DB_URI, DB_NAME, COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_db_connection(max_retries=5, retry_delay=5, server_timeout_ms=10000):
    """Establish and return a MongoDB collection connection with retries."""
    # Allow DB_URI override via environment variable
    db_uri = os.getenv("MONGO_DB_URI", DB_URI)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to MongoDB at {db_uri} (Attempt {attempt + 1}/{max_retries})")
            client = MongoClient(
                db_uri,
                serverSelectionTimeoutMS=server_timeout_ms,
                connectTimeoutMS=20000,
                socketTimeoutMS=20000
            )
            # Ping the server to verify connection
            client.admin.command('ping')
            logger.info("Connected to MongoDB successfully!")
            db = client[DB_NAME]
            return db[COLLECTION_NAME]
        except ServerSelectionTimeoutError as e:
            logger.error(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to connect to MongoDB at {db_uri} after {max_retries} attempts.")
                logger.error("Ensure MongoDB is running and accessible. Check DB_URI, port, and network settings.")
                raise
        except ConnectionFailure as e:
            logger.error(f"Connection failure: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to connect to MongoDB at {db_uri} after {max_retries} attempts.")
                raise
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            raise

def setup_vector_index(collection, embedding_size, timeout=300, max_interval=5):
    """Create or update the vector index for the collection."""
    try:
        start_time = time.time()
        indexes = list(collection.list_search_indexes())
        for index in indexes:
            if index["name"] == "vector_index":
                collection.drop_search_index(index["name"])
                logger.info("Existing 'vector_index' dropped.")

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

        logger.info("Waiting for vector index to be ready...")
        interval = 0.5
        attempt = 0
        while time.time() - start_time < timeout:
            indexes = list(collection.list_search_indexes())
            vector_index = next((idx for idx in indexes if idx["name"] == "vector_index"), None)
            if vector_index and vector_index.get("status") == "READY":
                logger.info("Vector index is now ready!")
                indexing_duration = time.time() - start_time
                logger.info(f"Document Indexing Duration: {indexing_duration:.2f}s")
                return indexing_duration
            sleep_time = min(interval * (1.5 ** attempt), max_interval)
            time.sleep(sleep_time)
            attempt += 1
        raise TimeoutError("Vector index creation timed out.")
    except Exception as e:
        logger.error(f"Error setting up vector index: {e}")
        raise