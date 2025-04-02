# RAG System Performance Report

## Overview
This report evaluates a Retrieval-Augmented Generation (RAG) system built with a modular Python codebase, leveraging MongoDB for vector storage and SentenceTransformers for embeddings. The system was tested with the full `hotpotqa` dataset from `rungalileo/ragbench` (1557 documents, 390 queries) under two studies:

1. **Study 1: Query Capacity with Parallelism** — Assesses how many queries the system supports under varying levels of parallelism (1 to 16 workers).
2. **Study 2: Embedding Model Comparison** — Compares performance between two embedding models: `all-mpnet-base-v2` (768 dimensions) and `all-MiniLM-L6-v2` (384 dimensions).

I am running the first test on my local machine 

## Setup

- **Dataset**: Full `hotpotqa` test split (1557 documents, 390 queries)
- **Database**: MongoDB with a vector index (`knnVector`, cosine similarity, `K=3` results)
- **Embedding Models**:
  - `all-mpnet-base-v2`: Larger, more accurate (768 dimensions)
  - `all-MiniLM-L6-v2`: Smaller, faster (384 dimensions)
- **Evaluation Metrics**:
  - **Database Size**: Storage in MB
  - **Retrieval Success**: % of queries returning results
  - **Average Precision**: % of retrieved documents that are relevant
  - **Average Latency**: Seconds per query
  - **Throughput**: Queries per second (q/s)
  - **Embedding Time**: Time to embed and store all data
- **Parallelism**: Tested with 1 (sequential), 2, 4, 8, and 16 workers using `ThreadPoolExecutor`

---

## Findings

### Study 1: Query Capacity with Parallelism

#### `all-mpnet-base-v2`

| Workers | Latency (s/query) | Throughput (q/s) | Notes                          |
|---------|-------------------|------------------|--------------------------------|
| 1       | 0.1264            | 7.78             | Baseline, sequential processing |
| 2       | 0.1094            | 18.23            | Peak efficiency, latency improves |
| 4       | 0.1935            | 20.61            | Throughput nears peak, latency rises |
| 8       | 0.3836            | 20.76            | Throughput plateaus, latency doubles |
| 16      | 0.7244            | 21.86            | Max throughput, severe latency hit |

- **Embedding Time**: 390.21 seconds (~0.25 s/doc)  
- **Database Size**: 15.56 MB  
- **Precision**: 74.27% (869/1170)

#### `all-MiniLM-L6-v2`

| Workers | Latency (s/query) | Throughput (q/s) | Notes                          |
|---------|-------------------|------------------|--------------------------------|
| 1       | 0.0529            | 18.10            | Faster baseline than mpnet     |
| 2       | 0.0455            | 43.75            | Best latency, strong scaling   |
| 4       | 0.0764            | 52.16            | Near-peak throughput           |
| 8       | 0.1514            | 52.55            | Throughput stabilizes          |
| 16      | 0.2991            | 52.90            | Max throughput, latency triples|

- **Embedding Time**: 93.82 seconds (~0.06 s/doc)  
- **Database Size**: 8.14 MB  
- **Precision**: 73.93% (865/1170)

**Key Observations**:

- **`all-mpnet-base-v2`**:
  - Peak throughput: ~21-22 q/s (4–16 workers)
  - Latency degrades significantly beyond 4 workers (0.1935s → 0.7244s)
  - Throughput scales well up to 4 workers, then plateaus

- **`all-MiniLM-L6-v2`**:
  - Peak throughput: ~52-53 q/s (4–16 workers), over 2x higher than mpnet
  - Latency remains low up to 4 workers, tripling at 16 workers
  - Scales efficiently due to smaller model size

**Bottlenecks**:

- CPU contention for embedding generation limits parallelism (GIL impact)
- MongoDB vector search scales but slows with worker contention

---

### Study 2: Embedding Model Comparison

Sequential execution (1 worker)

| Metric                | `all-mpnet-base-v2` | `all-MiniLM-L6-v2` | Difference |
|-----------------------|---------------------|---------------------|------------|
| Embedding Time (s)    | 390.21              | 93.82               | -76%       |
| Database Size (MB)    | 15.56               | 8.14                | -48%       |
| Precision (%)         | 74.27               | 73.93               | -0.34%     |
| Latency (s/query)     | 0.1264              | 0.0529              | -58%       |
| Throughput (q/s)      | 7.78                | 18.10               | +133%      |

**Key Observations**:

- **`all-mpnet-base-v2`**:
  - Higher precision (74.27%) due to richer embeddings
  - Slower embedding and query processing
  - Larger storage footprint

- **`all-MiniLM-L6-v2`**:
  - Slightly lower precision (73.93%) is a fair trade-off
  - 4x faster embedding, 2.4x faster query latency
  - Smaller storage and 2x higher throughput

**Bottlenecks**:

- Embedding generation is a major bottleneck for larger models
- Latency differences highlight computational overhead

---

Bottlenecks Identified:

    Embedding generation (CPU-bound, worse for larger models).
    Query latency under parallelism (contention in embedding or MongoDB).
    Throughput ceiling (hardware-limited beyond 4-8 workers).
    Smaller model size -> better for scalability. 


    Let's check on VM 