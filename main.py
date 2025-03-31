# main.py
from data_loader import load_and_store_test_data
from vector_util import generate_embedding
from retrieval import retrieve_top_k

def main():
    # Load test data and store in MongoDB
    collection, dataset_test = load_and_store_test_data(limit=10)
    
    # Test retrieval with each test question
    for entry in dataset_test:
        query = entry["question"]
        query_embedding = generate_embedding(query)
        results = retrieve_top_k(query_embedding, collection)
        
        print(f"\nQuery: {query}")
        print(f"Top {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['text'][:100]}...")
        
        # Basic evaluation (optional)
        relevant_docs = entry["documents"]  # Full list of candidate docs
        retrieved_texts = [r["text"] for r in results]
        print("Retrieved relevant? (manual check):")
        for i, text in enumerate(retrieved_texts, 1):
            is_relevant = text in relevant_docs  # Simple check
            print(f"{i}. {is_relevant}")

if __name__ == "__main__":
    main()