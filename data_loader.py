# data_loader.py
from datasets import load_dataset
import time
from config import DATASET_NAME, SUBSET_NAME
from db_utils import get_db_connection, setup_vector_index

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None):
    """Load dataset and store it in MongoDB with embeddings."""
    if embedding_generator is None or embedding_size is None:        
        raise ValueError("Error : Embedding generator and size are required.")
    # Load dataset
    split = "test" if limit is None else f"test[:{limit}]"
    dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=split)

    # Connect to MongoDB
    collection = get_db_connection()
    collection.delete_many({})  # Clear existing data

    # Time embedding and storage
    start_time = time.time()
    for entry in dataset:
        for doc in entry["documents"]:
            embedding = embedding_generator.generate_embedding(doc)
            collection.insert_one({
                "text": doc,
                "embedding": embedding,
                "question_id": entry["id"],
                "source": "test"
            })
    embedding_storage_time = time.time() - start_time

    # Setup vector index
    setup_vector_index(collection, embedding_size)

    total_docs = collection.count_documents({})
    print(f"Stored {total_docs} documents in MongoDB.")
    return collection, dataset, embedding_storage_time