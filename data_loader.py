# data_loader.py
from datasets import load_dataset
from vector_util import generate_embedding, setup_vector_index
from config import DATASET_NAME, SUBSET_NAME

def load_and_store_test_data(limit=10):
    """Load HotpotQA test data and store documents in MongoDB."""
    # Load test split
    dataset_test = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=f"test[:{limit}]")
    
    # Connect to MongoDB
    from pymongo import MongoClient
    client = MongoClient("mongodb://localhost:27017")
    db = client["rag_db"]
    collection = db["hotpotqa_docs"]
    
    # Clear existing data (optional)
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
    
    # Now create the vector index after data is inserted
    from vector_util import setup_vector_index
    collection = setup_vector_index()  # This now works because the collection exists
    
    print(f"Stored {collection.count_documents({})} documents in MongoDB.")
    
    return collection, dataset_test