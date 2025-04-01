# data_loader.py
from datasets import load_dataset
from vector_util import generate_embedding, setup_vector_index
from config import DATASET_NAME, SUBSET_NAME, DB_URI, DB_NAME, COLLECTION_NAME


def load_and_store_test_data(limit):
    """Load HotpotQA test data and store documents in MongoDB."""
    # Load full test split
    if limit is None:
        dataset_test = load_dataset(DATASET_NAME, name=SUBSET_NAME, split="test")
    else:
        dataset_test = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=f"test[:{limit}]")
    
    # Connect to MongoDB using config
    from pymongo import MongoClient
    client = MongoClient(DB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Clear existing data
    collection.delete_many({})
    
    # Store documents with embeddings
    for entry in dataset_test:
        documents = entry["documents"]
        for doc in documents:
            embedding = generate_embedding(doc)
            collection.insert_one({
                "text": doc,
                "embedding": embedding,
                "question_id": entry["id"],
                "source": "test"
            })
    
    # Create the vector index after data is inserted
    collection = setup_vector_index()
    
    total_docs = collection.count_documents({})
    print(f"Stored {total_docs} documents in MongoDB.")
    
    return collection, dataset_test