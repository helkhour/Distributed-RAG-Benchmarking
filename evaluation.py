from vector_util import EmbeddingGenerator
from retrieval import retrieve_top_k

class Evaluator:
    def __init__(self, collection, verbose=True):
        self.collection = collection
        self.embedding_generator = EmbeddingGenerator()
        self.verbose = verbose

    def evaluate(self, dataset):
        total_queries = len(dataset)
        queries_with_results = 0
        total_relevant = 0
        total_retrieved = 0

        for entry in dataset:
            query = entry["question"]
            query_embedding = self.embedding_generator.generate_embedding(query)
            results = retrieve_top_k(query_embedding, self.collection)
            
            if self.verbose:
                print(f"\nQuery: {query}")
                print(f"Top {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['text'][:100]}...")

            relevant_docs = entry["documents"]
            retrieved_texts = [r["text"] for r in results]
            relevant_count = sum(text in relevant_docs for text in retrieved_texts)

            if self.verbose:
                print("Retrieved relevant? (manual check):")
                for i, text in enumerate(retrieved_texts, 1):
                    print(f"{i}. {text in relevant_docs}")

            if results:
                queries_with_results += 1
            total_relevant += relevant_count
            total_retrieved += len(retrieved_texts)

        metrics = {
            "retrieval_success": queries_with_results / total_queries if total_queries > 0 else 0,
            "avg_precision": total_relevant / total_retrieved if total_retrieved > 0 else 0,
            "db_size": self.collection.count_documents({})
        }

        if self.verbose:
            print("\n=== Retrieval Performance Summary ===")
            print(f"Database Size: {metrics['db_size']} documents")
            print(f"Retrieval Success: {metrics['retrieval_success']:.2%} ({queries_with_results}/{total_queries})")
            print(f"Average Precision: {metrics['avg_precision']:.2%} ({total_relevant}/{total_retrieved})")

        return metrics