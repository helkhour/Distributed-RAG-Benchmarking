from datasets import load_dataset, concatenate_datasets
from config import DATASET_NAME, SUBSET_NAME
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator
from tqdm import tqdm
import logging
import torch

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    evaluator = SystemEvaluator()
    if embedding_generator is None or embedding_size is None:
        raise ValueError("Error: Embedding generator and size are required.")
    
    evaluator.log_resources("Before Dataset Load")
    
    split = "test" if limit is None else f"test[:{limit}]"
    hotpotqa_dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=split)
    pubmedqa_dataset = load_dataset(DATASET_NAME, name="pubmedqa", split=split)
    
    evaluator.start_monitoring()
    hotpotqa_doc_count = sum(len(entry["documents"]) for entry in hotpotqa_dataset)
    pubmedqa_bytes = sum(len(doc.encode('utf-8')) for entry in hotpotqa_dataset for doc in entry["documents"])
    hotpotqa_mb = pubmedqa_bytes / (1024 * 1024)
    
    pubmedqa_doc_count = sum(len(entry["documents"]) for entry in pubmedqa_dataset)
    pubmedqa_bytes = sum(len(doc.encode('utf-8')) for entry in pubmedqa_dataset for doc in entry["documents"])
    pubmedqa_mb = pubmedqa_bytes / (1024 * 1024)
    
    combined_dataset = concatenate_datasets([hotpotqa_dataset, pubmedqa_dataset])
    total_doc_count = hotpotqa_doc_count + pubmedqa_doc_count
    total_mb = hotpotqa_mb + pubmedqa_mb
    
    logger.info(f"HotpotQA Test Split Size: {hotpotqa_doc_count} docs, {hotpotqa_mb:.2f} MB")
    logger.info(f"PubMedQA Test Split Size: {pubmedqa_doc_count} docs, {pubmedqa_mb:.2f} MB")
    logger.info(f"Combined Dataset Size: {total_doc_count} docs, {total_mb:.2f} MB")
    
    all_docs = []
    for entry in combined_dataset:
        for doc in entry["documents"]:
            all_docs.append({
                "text": doc,
                "question_ids": [entry["id"]],
                "source": "test"
            })
    logger.info(f"Total Documents to Embed: {len(all_docs)}")
        
    # Validate dataset structure
    for i, entry in enumerate(combined_dataset):
        if "documents" not in entry or not isinstance(entry["documents"], list):
            logger.warning(f"Entry {i} missing or invalid 'documents' field")
    dataset_load_duration, _ = evaluator.end_monitoring("Dataset Load")
    
    collection = get_db_connection()
    collection.delete_many({})
    
    evaluator.start_monitoring()

    # Batch processing
    batch_size = 8
    docs = []
    texts = []
    batch_size = 8
    for item in tqdm(all_docs, desc="Generating embeddings"):
        texts.append(item["text"])
        if len(texts) >= batch_size:
            embeddings = embedding_generator.generate_embedding(texts)
            for i, embedding in enumerate(embeddings):
                docs.append({
                    "text": texts[i],
                    "embedding": embedding,
                    "question_ids": item["question_ids"],
                    "source": item["source"]
                })
            texts = []
            torch.cuda.empty_cache()
    if texts:
        embeddings = embedding_generator.generate_embedding(texts)
        for i, embedding in enumerate(embeddings):
            docs.append({
                "text": texts[i],
                "embedding": embedding,
                "question_ids": item["question_ids"],
                "source": item["source"]
            })

    embedding_duration, embedding_cpu_delta = evaluator.end_monitoring("Embedding and Storage")
    logger.info(f"Embedding and Storage Duration: {embedding_duration:.2f}s")    
    logger.info("Inserting documents into MongoDB")
    collection.insert_many(docs, ordered=False)
    
    logger.info("Setting up vector index")
    setup_vector_index(collection, embedding_size)
    
    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})
    logger.info(f"Stored {total_docs} documents in MongoDB.")
    
    return collection, combined_dataset, embedding_duration