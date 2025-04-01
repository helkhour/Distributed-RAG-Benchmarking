# main.py
from data_loader import load_and_store_test_data
from vector_util import generate_embedding
from retrieval import retrieve_top_k
import time 

def evaluate_retrieval_performance(dataset_test, collection):
    """Calculate retrieval success, average precision, and database size."""
    total_queries = len(dataset_test)
    queries_with_results = 0
    total_relevant = 0
    total_retrieved = 0
    
    for entry in dataset_test:
        query = entry["question"]
        query_embedding = generate_embedding(query)
        results = retrieve_top_k(query_embedding, collection)
        
        # Existing output
        print(f"\nQuery: {query}")
        print(f"Top {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text'][:100]}...")
        
        # Existing relevance check
        relevant_docs = entry["documents"]
        retrieved_texts = [r["text"] for r in results]
        print("Retrieved relevant? (manual check):")
        for i, text in enumerate(retrieved_texts, 1):
            is_relevant = text in relevant_docs
            print(f"{i}. {is_relevant}")
        
        # Metrics calculation
        if results:
            queries_with_results += 1
        relevant_count = sum(text in relevant_docs for text in retrieved_texts)
        total_relevant += relevant_count
        total_retrieved += len(retrieved_texts)
    
    # Calculate and print metrics
    retrieval_success = queries_with_results / total_queries if total_queries > 0 else 0
    avg_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    db_size = collection.count_documents({})
    
    print("\n=== Retrieval Performance Summary ===")
    print(f"Database Size: {db_size} documents")
    print(f"Retrieval Success: {retrieval_success:.2%} ({queries_with_results}/{total_queries} queries returned results)")
    print(f"Average Precision: {avg_precision:.2%} ({total_relevant}/{total_retrieved} relevant documents retrieved)")

def main():
    # Load test data and store in MongoDB
    collection, dataset_test = load_and_store_test_data(limit=None)
    
    # Evaluate retrieval performance
    evaluate_retrieval_performance(dataset_test, collection)

if __name__ == "__main__":
    main()