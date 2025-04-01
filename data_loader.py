from datasets import load_dataset
from vector_util import EmbeddingGenerator, setup_vector_index
from db_utils import MongoDBClient
from config import DATASET_CONFIG

def load_dataset(name=DATASET_CONFIG["name"], subset=DATASET_CONFIG["subset"], split="test", limit=None):
    split_str = f"{split}[:{limit}]" if limit else split
    return load_dataset(name, name=subset, split=split_str)

def store_documents(dataset, collection, embedding_generator):
    collection.delete_many({})
    for entry in dataset:
        documents = entry["documents"]
        for doc in documents:
            embedding = embedding_generator.generate_embedding(doc)
            collection.insert_one({
                "text": doc,
                "embedding": embedding,
                "question_id": entry["id"],
                "source": "test"
            })
    setup_vector_index(collection)
    total_docs = collection.count_documents({})
    print(f"Stored {total_docs} documents in MongoDB.")

def load_and_store_test_data(limit=None):
    db_client = MongoDBClient()
    collection = db_client.get_collection()
    dataset = load_dataset(limit=limit)
    embedding_generator = EmbeddingGenerator()
    store_documents(dataset, collection, embedding_generator)
    return collection, dataset