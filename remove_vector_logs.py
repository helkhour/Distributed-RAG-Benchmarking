import re

def clean_log_file(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    patterns_to_remove = [
        r"^Vector Similarity Search Duration: 0\.",
        r"^Batches:.*",
        r"^Query Encoding Duration:.*",
        r"^Sequential insertion:.*",
        r"^Generating embeddings:.*"
    ]

    with open(output_path, 'w') as outfile:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue  # Remove blank lines
            if any(re.match(pattern, stripped) for pattern in patterns_to_remove):
                continue
            outfile.write(line)

    print(f"✅ Cleaned log written to: {output_path}")


# Example usage
if __name__ == "__main__":
    input_file = "rag_log.log"            # Replace with your actual input file
    output_file = "cleaned_log.txt"   # Output file
    clean_log_file(input_file, output_file)
