# data_loader.py
from datasets import load_dataset, concatenate_datasets
from config import DATASET_NAME, SUBSET_NAME
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None):
    """Load hotpotqa and pubmedqa test splits, store in MongoDB with embeddings."""
    evaluator = SystemEvaluator()
    if embedding_generator is None or embedding_size is None:
        raise ValueError("Error: Embedding generator and size are required.")
    
    evaluator.log_resources("Before Dataset Load")
    
    split = "test" if limit is None else f"test[:{limit}]"
    hotpotqa_dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=split)
    pubmedqa_dataset = load_dataset(DATASET_NAME, name="pubmedqa", split=split)
    
    evaluator.start_monitoring()
    hotpotqa_doc_count = sum(len(entry["documents"]) for entry in hotpotqa_dataset)
    hotpotqa_bytes = sum(len(doc.encode('utf-8')) for entry in hotpotqa_dataset for doc in entry["documents"])
    hotpotqa_mb = hotpotqa_bytes / (1024 * 1024)  # Convert bytes to MB
    
    pubmedqa_doc_count = sum(len(entry["documents"]) for entry in pubmedqa_dataset)
    pubmedqa_bytes = sum(len(doc.encode('utf-8')) for entry in pubmedqa_dataset for doc in entry["documents"])
    pubmedqa_mb = pubmedqa_bytes / (1024 * 1024)  # Convert bytes to MB
    
    combined_dataset = concatenate_datasets([hotpotqa_dataset, pubmedqa_dataset])
    total_doc_count = hotpotqa_doc_count + pubmedqa_doc_count
    total_mb = hotpotqa_mb + pubmedqa_mb
    
    print(f"HotpotQA Test Split Size: {hotpotqa_doc_count} documents, {hotpotqa_mb:.2f} MB")
    print(f"PubMedQA Test Split Size: {pubmedqa_doc_count} documents, {pubmedqa_mb:.2f} MB")
    print(f"Combined Dataset Size: {total_doc_count} documents, {total_mb:.2f} MB")
    
    dataset_load_duration, _ = evaluator.end_monitoring("Dataset Load")
    
    collection = get_db_connection()
    collection.delete_many({})
    
    evaluator.start_monitoring()
    docs = []
    for entry in combined_dataset:
        for doc in entry["documents"]:
            embedding = embedding_generator.generate_embedding(doc)
            docs.append({
                "text": doc,
                "embedding": embedding,
                "question_id": entry["id"],
                "source": "test"
            })
    embedding_duration, embedding_cpu_delta = evaluator.end_monitoring("Embedding Generation")
    
    # Batch insert documents
    evaluator.start_monitoring()
    collection.insert_many(docs)
    storage_duration, storage_cpu_delta = evaluator.end_monitoring("Storage")
    
    embedding_storage_time = embedding_duration + storage_duration
    
    evaluator.start_monitoring()
    setup_vector_index(collection, embedding_size)
    index_duration, index_cpu_delta = evaluator.end_monitoring("Index Setup")
    
    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})
    print(f"Stored {total_docs} documents in MongoDB.")
    
    return collection, combined_dataset, embedding_storage_time