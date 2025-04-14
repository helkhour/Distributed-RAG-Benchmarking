# Summary of Results and Refined Data Loader

Below is a high-level overview of performance metrics, system usage observations, and a streamlined version of the `data_loader.py` script. These findings focus on the scaled dataset (HotpotQA + PubMedQA test splits, **13,807 documents**) using **sentence-transformers/all-MiniLM-L6-v2** embeddings. 

## 1. System Context

- **Machine**: Local setup (likely multi-core, ~8+ CPUs).
- **Dataset**: 
  - **HotpotQA**: 1,557 documents
  - **PubMedQA**: 12,250 documents
  - **Total**: 13,807 documents, up from 1,557 in prior tests.
  - **Queries**: 2,840 queries (up from 390).
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings, considered lightweight among transformer-based encoders).

## 2. Key Metrics

| **Metric**                | **Value**   | **Comparison (Prior Test on 1,557 Docs)** |
|---------------------------|-------------|-------------------------------------------|
| **Embedding Time (s)**    | 611.59      | 64.36 (~9.5× increase)                   |
| **Latency (s/query)**     | 0.1175      | 0.0852 (~1.4× increase)                  |
| **Throughput (q/s)**      | 8.36        | 11.42 (~0.73×)                           |
| **Precision (%)**         | 93.56       | 73.93 (~1.26×)                           |
| **DB Size (MB)**          | 70.63       | 8.14 (~8.7×)                              |
| **Peak Memory (MB)**      | 1123.99     | 665.31 (~1.7×)                            |
| **CPU Peak (%)**          | 51.00       | 45.00                                     |
| **CPU Time Total (s)**    | 9537.55     | 1071.66                                   |

**Note**: The document count increased by ~8.9× (1,557 → 13,807), which drove near-linear growth in embedding time, DB storage size, and memory usage.

## 3. Observations and Insights

1. **Dataset Size**  
   - **Text Volume**: ~0.73 MB from HotpotQA, ~4.31 MB from PubMedQA, total ~5.04 MB.  
   - **Embedded DB**: 70.63 MB for 13,807 docs (~5.1 KB/doc). Each embedding (384-dim float array) is ~1.5 KB/doc, plus metadata overhead.  
   - **Scaling**: The database size scaled almost linearly with document count.

2. **CPU Usage**  
   - **Embedding**: 31% peak usage, totaling ~7,258s CPU time vs. 609s wall time (~12×) ⇒ indicates multi-threading (likely >8 threads).
   - **Retrieval**: 39–51% CPU spikes, 2,264s CPU time for 340s wall time (~6.7×).

3. **Memory Usage**  
   - Peak at ~1,124 MB, ~1.7× previous peak of ~665 MB. Reasonable scaling given ~8.9× more documents.

4. **Timing and Throughput**  
   - **Embedding**: 609.45s vs. 64.13s previously (~9.5×).
   - **Storage**: 2.14s vs. 0.23s (~9.3×).
   - **Indexing**: 3.43s vs. 1.08s (~3.2×). Some non-linear overhead with a larger index.
   - **Latency**: 0.1175s/query vs. 0.0852s (~1.4×). Larger DB (8.14 MB → 70.63 MB) and `numCandidates=300` likely cause the increase.

5. **Performance**  
   - **Precision**: Jumped from 73.93% to 93.56%. Possibly due to PubMedQA queries aligning well with `all-MiniLM-L6-v2` embeddings.
   - **Throughput**: Decreased from 11.42 q/s to 8.36 q/s. The larger dataset and single-worker retrieval slow down processing slightly.

## 4. Inconsistencies to Note

- **CPU Time Deltas**: High CPU-to-wall-time ratio (e.g., ~11.9×). Likely due to multiple threads on multi-core CPU.  
- **Precision Jump**: From ~74% to ~94%. The new dataset composition (PubMedQA) may be inherently more “friendly” to this model, or better matched queries.  
- **Latency Increase**: From 0.0852s to 0.1175s, expected with a ~9× jump in documents and index size.