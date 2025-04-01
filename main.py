import argparse
from data_loader import load_and_store_test_data
from evaluation import Evaluator

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of test samples")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed output")
    args = parser.parse_args()

    collection, dataset_test = load_and_store_test_data(limit=args.limit)
    evaluator = Evaluator(collection, verbose=args.verbose)
    evaluator.evaluate(dataset_test)

if __name__ == "__main__":
    main()