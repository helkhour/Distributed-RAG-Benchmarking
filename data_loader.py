from datasets import load_dataset, concatenate_datasets
from config import DATASET_NAME, SUBSET_NAME
from db_utils import get_db_connection, setup_vector_index
from system_evaluation import SystemEvaluator
from tqdm import tqdm
import torch
import numpy as np
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

def load_and_store_data(limit=None, embedding_generator=None, embedding_size=None, use_batch=False):
    """Load hotpotqa and pubmedqa test splits, store in MongoDB with embeddings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    evaluator = SystemEvaluator()
    if embedding_generator is None or embedding_size is None:
        raise ValueError("Error: Embedding generator and size are required.")
    
    timings = {"query_preprocessing": 0.0, "query_encoding": 0.0}
    stats = {}
        
    start_time = time.time()
    split = "test" if limit is None else f"test[:{limit}]"
    logger.info(f"Loading HotpotQA dataset with split: {split}")
    hotpotqa_dataset = load_dataset(DATASET_NAME, name=SUBSET_NAME, split=split)
    logger.info(f"Loading PubMedQA dataset with split: {split}")
    pubmedqa_dataset = load_dataset(DATASET_NAME, name="pubmedqa", split=split)
    timings["dataset_load"] = time.time() - start_time
    logger.info(f"Dataset Loading Duration: {timings['dataset_load']:.2f}s")
    
    logger.debug(f"Sample HotpotQA Entry: {hotpotqa_dataset[0]}")
    logger.debug(f"Sample PubMedQA Entry: {pubmedqa_dataset[0]}")
    
    evaluator.start_monitoring()
    
    hotpotqa_doc_count, hotpotqa_mb = compute_doc_stats(hotpotqa_dataset)
    pubmedqa_doc_count, pubmedqa_mb = compute_doc_stats(pubmedqa_dataset)
    combined_dataset = concatenate_datasets([hotpotqa_dataset, pubmedqa_dataset])
    total_doc_count = hotpotqa_doc_count + pubmedqa_doc_count
    total_mb = hotpotqa_mb + pubmedqa_mb
    
    stats["hotpotqa_entries"] = len(hotpotqa_dataset)
    stats["pubmedqa_entries"] = len(pubmedqa_dataset)
    stats["total_entries"] = len(combined_dataset)
    stats["hotpotqa_docs"] = hotpotqa_doc_count
    stats["pubmedqa_docs"] = pubmedqa_doc_count
    stats["total_docs"] = total_doc_count
    stats["hotpotqa_size_mb"] = hotpotqa_mb
    stats["pubmedqa_size_mb"] = pubmedqa_mb
    stats["total_size_mb"] = total_mb
    stats["embedding_mode"] = "batch" if use_batch else "sequential"
    
    logger.info(f"HotpotQA Test Split: {stats['hotpotqa_entries']} entries, {hotpotqa_doc_count} docs, {hotpotqa_mb:.2f} MB")
    logger.info(f"PubMedQA Test Split: {stats['pubmedqa_entries']} entries, {pubmedqa_doc_count} docs, {pubmedqa_mb:.2f} MB")
    logger.info(f"Combined Dataset: {stats['total_entries']} entries, {total_doc_count} docs, {total_mb:.2f} MB")
    
    if stats["total_entries"] < 100:
        logger.warning(f"Dataset size is very small ({stats['total_entries']} entries). Ensure 'limit' is None or dataset split is correct.")
    
    # Collect all documents
    start_time = time.time()
    all_docs = []
    for entry in combined_dataset:
        for doc in entry["documents"]:
            all_docs.append({
                "text": doc,
                "question_ids": [entry["id"]],
                "source": "test"
            })
    timings["doc_collection"] = time.time() - start_time
    logger.info(f"Document Collection Duration: {timings['doc_collection']:.2f}s")
    logger.info(f"Total Documents to Embed: {len(all_docs)}")
    
    # Embedding generation
    evaluator.start_monitoring()
    start_time = time.time()
    docs = []
    embedding_norms = []
    
    if use_batch:
        logger.info("Using batch embedding mode")
        batch_size = 8
        items = []
        for item in tqdm(all_docs, desc="Generating batch embeddings", disable=True):
            items.append(item)
            if len(items) >= batch_size:
                texts = [item["text"] for item in items]
                embeddings, batch_timings = embedding_generator.generate_embedding(texts)
                timings["query_preprocessing"] += batch_timings["query_preprocessing"]
                timings["query_encoding"] += batch_timings["query_encoding"]
                for embedding, item in zip(embeddings, items):
                    norm = np.linalg.norm(embedding)
                    embedding_norms.append(norm)
                    if norm < 1e-6:
                        logger.warning(f"Zero or near-zero embedding norm for text: {item['text'][:50]}...")
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
            embeddings, batch_timings = embedding_generator.generate_embedding(texts)
            timings["query_preprocessing"] += batch_timings["query_preprocessing"]
            timings["query_encoding"] += batch_timings["query_encoding"]
            for embedding, item in zip(embeddings, items):
                norm = np.linalg.norm(embedding)
                embedding_norms.append(norm)
                if norm < 1e-6:
                    logger.warning(f"Zero or near-zero embedding norm for text: {item['text'][:50]}...")
                docs.append({
                    "text": item["text"],
                    "embedding": embedding,
                    "question_ids": item["question_ids"],
                    "source": item["source"]
                })
            torch.cuda.empty_cache()
        timings["batch_embedding"] = time.time() - start_time
        logger.info(f"Batch Embedding Duration: {timings['batch_embedding']:.2f}s")
    else:
        logger.info("Using sequential embedding mode")
        for item in tqdm(all_docs, desc="Generating sequential embeddings", disable=True):
            embedding, seq_timings = embedding_generator.generate_embedding(item["text"])
            embedding = embedding[0]  # Single text input
            norm = np.linalg.norm(embedding)
            embedding_norms.append(norm)
            if norm < 1e-6:
                logger.warning(f"Zero or near-zero embedding norm for text: {item['text'][:50]}...")
            timings["query_preprocessing"] += seq_timings["query_preprocessing"]
            timings["query_encoding"] += seq_timings["query_encoding"]
            docs.append({
                "text": item["text"],
                "embedding": embedding,
                "question_ids": item["question_ids"],
                "source": item["source"]
            })
            torch.cuda.empty_cache()
        timings["sequential_embedding"] = time.time() - start_time
        logger.info(f"Sequential Embedding Duration: {timings['sequential_embedding']:.2f}s")
    
    logger.info(f"Total Query Preprocessing: {timings['query_preprocessing']:.2f}s")
    logger.info(f"Total Query Encoding: {timings['query_encoding']:.2f}s")
    logger.info(f"Average Embedding Norm: {np.mean(embedding_norms):.4f} ± {np.std(embedding_norms):.4f}")
    
    timings["embedding_generation"], _ = evaluator.end_monitoring("Embedding Generation")
    
    # Database operations
    evaluator.start_monitoring()
    
    start_time = time.time()
    collection = get_db_connection()
    timings["db_connection"] = time.time() - start_time
    
    start_time = time.time()
    collection.delete_many({})
    timings["db_clearing"] = time.time() - start_time
    logger.info(f"Database Clearing Duration: {timings['db_clearing']:.2f}s")
    
    start_time = time.time()
    logger.info("Inserting documents in batch into MongoDB")
    collection.insert_many(docs, ordered=False)
    timings["batch_storage"] = time.time() - start_time
    logger.info(f"Batch Document Storage Duration: {timings['batch_storage']:.2f}s")
    
    logger.info("Setting up vector index")
    timings["document_indexing"] = setup_vector_index(collection, embedding_size)
    
    timings["embedding_storage_total"], _ = evaluator.end_monitoring("Embedding and Storage")
    
    evaluator.log_resources("After Index Setup")
    total_docs = collection.count_documents({})
    stats["docs_stored_batch"] = len(docs)
    stats["db_entries"] = total_docs
    
    logger.info(f"Stored {stats['docs_stored_batch']} documents in batch, {stats['db_entries']} entries in MongoDB.")
    
    return collection, combined_dataset, timings, stats