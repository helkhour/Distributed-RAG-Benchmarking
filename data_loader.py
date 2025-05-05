from datasets import load_dataset, concatenate_datasets
from config import DATASET_NAME, SUBSET_NAME
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator
from tqdm import tqdm
import torch
import logging
import time

def compute_doc_stats(dataset):
    """Compute document count and size in MB for a dataset."""
    doc_count = 0
    total_bytes = 0
    for entry in dataset:
        documents = entry["documents"]
        doc_count += len(documents)
        total_bytes += sum(len(doc.encode("utf-8")) for doc in documents)
    return doc_count, total_bytes / (1024 * 1024)

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None):
    """Load hotpotqa and pubmedqa test splits, store in MongoDB with embeddings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    evaluator = SystemEvaluator()
    if embedding_generator is None or embedding_size is None:
        raise ValueError("Error: Embedding generator and size are required.")
    
    timings = {}
    stats = {}
    
    evaluator.log_resources("Before Dataset Load")
    
    # Time dataset loading
    start_time = time.time()
    split = "test" if limit is None else f"test[:{limit}]"
    logger.info(f"Loading HotpotQA dataset with split: {split}")
    hotpotqa_dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=split)
    logger.info(f"Loading PubMedQA dataset with split: {split}")
    pubmedqa_dataset = load_dataset(DATASET_NAME, name="pubmedqa", split=split)
    timings["dataset_load"] = time.time() - start_time
    logger.info(f"Dataset Loading Duration: {timings['dataset_load']:.2f}s")
    
    evaluator.start_monitoring()
    hotpotqa_doc_count, hotpotqa_mb = compute_doc_stats(hotpotqa_dataset)
    pubmedqa_doc_count, pubmedqa_mb = compute_doc_stats(pubmedqa_dataset)
    
    combined_dataset = concatenate_datasets([hotpotqa_dataset, pubmedqa_dataset])
    total_doc_count = hotpotqa_doc_count + pubmedqa_doc_count
    total_mb = hotpotqa_mb + pubmedqa_mb
    
    # Collect dataset statistics
    stats["hotpotqa_entries"] = len(hotpotqa_dataset)
    stats["pubmedqa_entries"] = len(pubmedqa_dataset)
    stats["total_entries"] = len(combined_dataset)
    stats["hotpotqa_docs"] = hotpotqa_doc_count
    stats["pubmedqa_docs"] = pubmedqa_doc_count
    stats["total_docs"] = total_doc_count
    stats["hotpotqa_size_mb"] = hotpotqa_mb
    stats["pubmedqa_size_mb"] = pubmedqa_mb
    stats["total_size_mb"] = total_mb
    
    logger.info(f"HotpotQA Test Split: {stats['hotpotqa_entries']} entries, {hotpotqa_doc_count} docs, {hotpotqa_mb:.2f} MB")
    logger.info(f"PubMedQA Test Split: {stats['pubmedqa_entries']} entries, {pubmedqa_doc_count} docs, {pubmedqa_mb:.2f} MB")
    logger.info(f"Combined Dataset: {stats['total_entries']} entries, {total_doc_count} docs, {total_mb:.2f} MB")
    
    # Warn if dataset is small
    if stats["total_entries"] < 100:
        logger.warning(f"Dataset size is very small ({stats['total_entries']} entries). Ensure 'limit' is None or dataset split is correct.")
    
    # Time document preparation
    start_time = time.time()
    all_docs = []
    for entry in combined_dataset:
        for doc in entry["documents"]:
            all_docs.append({
                "text": doc,
                "question_ids": [entry["id"]],
                "source": "test"
            })
    timings["doc_preparation"] = time.time() - start_time
    logger.info(f"Document Preparation Duration: {timings['doc_preparation']:.2f}s")
    logger.info(f"Total Documents to Embed: {len(all_docs)}")
    
    timings["dataset_load_total"] = evaluator.end_monitoring("Dataset Load")[0]
    
    # Time database connection
    start_time = time.time()
    collection = get_db_connection()
    timings["db_connection"] = time.time() - start_time
    
    # Time database clearing (before sequential)
    start_time = time.time()
    collection.delete_many({})
    timings["db_clearing"] = time.time() - start_time
    logger.info(f"Database Clearing Duration: {timings['db_clearing']:.2f}s")
    
    # Time embedding generation
    evaluator.start_monitoring()
    start_time = time.time()
    batch_size = 8
    docs = []
    items = []
    for item in tqdm(all_docs, desc="Generating embeddings"):
        items.append(item)
        if len(items) >= batch_size:
            texts = [item["text"] for item in items]
            embeddings = embedding_generator.generate_embedding(texts)
            for i, (embedding, item) in enumerate(zip(embeddings, items)):
                docs.append({
                    "text": item["text"],
                    "embedding": embedding,
                    "question_ids": item["question_ids"],
                    "source": item["source"]
                })
            items = []
            torch.cuda.empty_cache()
    if items:
        texts = [item["text"] for item in items]
        embeddings = embedding_generator.generate_embedding(texts)
        for i, (embedding, item) in enumerate(zip(embeddings, items)):
            docs.append({
                "text": item["text"],
                "embedding": embedding,
                "question_ids": item["question_ids"],
                "source": item["source"]
            })
        torch.cuda.empty_cache()
    timings["embedding_generation"] = time.time() - start_time
    logger.info(f"Embedding Generation Duration: {timings['embedding_generation']:.2f}s")
    
    # Time sequential document storage
    start_time = time.time()
    logger.info("Inserting documents sequentially into MongoDB")
    for doc in tqdm(docs, desc="Sequential insertion"):
        collection.insert_one(doc)
    timings["sequential_storage"] = time.time() - start_time
    logger.info(f"Sequential Document Storage Duration: {timings['sequential_storage']:.2f}s")
    
    stats["docs_stored_sequential"] = collection.count_documents({})
    logger.info(f"Stored {stats['docs_stored_sequential']} documents sequentially.")
    
    # Time database clearing (before batch)
    start_time = time.time()
    collection.delete_many({})
    timings["db_clearing_batch"] = time.time() - start_time
    logger.info(f"Database Clearing (before batch) Duration: {timings['db_clearing_batch']:.2f}s")
    
    # Time batch document storage
    start_time = time.time()
    logger.info("Inserting documents in batch into MongoDB")
    collection.insert_many(docs, ordered=False)
    timings["batch_storage"] = time.time() - start_time
    logger.info(f"Batch Document Storage Duration: {timings['batch_storage']:.2f}s")
    
    timings["embedding_storage_total"] = evaluator.end_monitoring("Embedding and Storage")[0]
    
    # Time document indexing
    logger.info("Setting up vector index")
    timings["document_indexing"] = setup_vector_index(collection, embedding_size)
    
    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})
    stats["docs_stored_batch"] = len(docs)
    stats["db_entries"] = total_docs
    
    logger.info(f"Stored {stats['docs_stored_batch']} documents in batch, {stats['db_entries']} entries in MongoDB.")
    
    return collection, combined_dataset, timings, stats