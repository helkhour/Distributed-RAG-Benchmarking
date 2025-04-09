# data_loader.py
from datasets import load_dataset
from config import DATASET_NAME, SUBSET_NAME
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None):
    """Load dataset and store it in MongoDB with embeddings."""
    evaluator = SystemEvaluator()
    if embedding_generator is None or embedding_size is None:
        raise ValueError("Error: Embedding generator and size are required.")
    
    # Log initial state
    evaluator.log_resources("Before Dataset Load")
    
    # Load dataset
    split = "test" if limit is None else f"test[:{limit}]"
    evaluator.start_monitoring()
    dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=split)
    dataset_load_duration, _ = evaluator.end_monitoring("Dataset Load")
    
    # Connect to MongoDB and clear collection
    collection = get_db_connection()
    evaluator.log_resources("Before Clear Collection")
    collection.delete_many({})
    evaluator.log_resources("After Clear Collection")
    
    # Generate embeddings and prepare documents
    evaluator.start_monitoring()
    docs = []
    for entry in dataset:
        for doc in entry["documents"]:
            embedding = embedding_generator.generate_embedding(doc)
            docs.append({
                "text": doc,
                "embedding": embedding,
                "question_id": entry["id"],
                "source": "test"
            })
    embedding_duration, embedding_cpu_delta = evaluator.end_monitoring("Embedding Generation")
        # docs = []
    # for entry in dataset:
    #     for doc in entry["documents"]:
    #         embedding = embedding_generator.generate_embedding(doc)
    #         docs.append({
    #             "text": doc,
    #             "embedding": embedding,
    #             "question_id": entry["id"],
    #             "source": "test"
    #         })
    # collection.insert_many(docs)  # Batch insert for more efficiency ???? 

    # Batch insert documents
    evaluator.start_monitoring()
    collection.insert_many(docs)
    storage_duration, storage_cpu_delta = evaluator.end_monitoring("Storage")
    
    # Total embedding + storage time (for compatibility with prior runs)
    embedding_storage_time = embedding_duration + storage_duration
    
    # Setup vector index
    evaluator.start_monitoring()
    setup_vector_index(collection, embedding_size)
    index_duration, index_cpu_delta = evaluator.end_monitoring("Index Setup")
    
    # Log final state
    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})
    print(f"Stored {total_docs} documents in MongoDB.")
    
    return collection, dataset, embedding_storage_time