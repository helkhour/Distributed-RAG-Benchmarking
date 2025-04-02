# RAG System Performance Comparison: EC2 vs Local Machine

## Recap of Setup

- **Dataset**: `hotpotqa` (1557 documents, 390 queries)
- **Models**:
  - `all-mpnet-base-v2`: 768-dimensional embeddings
  - `all-MiniLM-L6-v2`: 384-dimensional embeddings
- **Metrics**:
  - **Embedding Time (s)**: Time to embed and store all documents
  - **Database Size (MB)**: Storage footprint
  - **Precision (%)**: Relevant documents retrieved (K=3, 1170 total retrieved)
  - **Latency (s/query)**: Average query time
  - **Throughput (q/s)**: Queries per second
- **Parallelism**: Tested with 1, 2, 4, 8, and 16 workers
- **Instance type**: c6id.xlarge

---

## Findings on EC2 Instance (Ubuntu VM)

### `all-mpnet-base-v2`

- **Embedding Time**: 163.14s (~0.105 s/doc)
- **Database Size**: 15.56 MB
- **Precision**: 74.27% (869/1170)

| Workers | Latency (s/query) | Throughput (q/s) |
|---------|-------------------|------------------|
| 1       | 0.0533            | 18.57            |
| 2       | 0.0846            | 23.59            |
| 4       | 0.1868            | 21.37            |
| 8       | 0.3923            | 20.30            |
| 16      | 0.8581            | 18.52            |

---

### `all-MiniLM-L6-v2`

- **Embedding Time**: 36.89s (~0.024 s/doc)
- **Database Size**: 8.14 MB
- **Precision**: 73.93% (865/1170)

| Workers | Latency (s/query) | Throughput (q/s) |
|---------|-------------------|------------------|
| 1       | 0.0141            | 68.41            |
| 2       | 0.0233            | 85.58            |
| 4       | 0.0492            | 81.08            |
| 8       | 0.1220            | 65.33            |
| 16      | 0.2436            | 65.16            |

---

## Comparison and Interpretation

### 1. **Embedding Time**

| Model               | Local         | EC2           | Difference      |
|---------------------|---------------|---------------|-----------------|
| mpnet               | 390.21s       | 163.14s       | -58% (2.4× faster) |
| MiniLM              | 93.82s        | 36.89s        | -61% (2.5× faster) |

**Interpretation**: EC2's faster CPU or memory optimizations drastically reduce embedding time. (Need to check CPU and Memory of each)

---

### 2. **Database Size**

| Model               | Local         | EC2           |
|---------------------|---------------|---------------|
| mpnet               | 15.56 MB      | 15.56 MB      |
| MiniLM              | 8.14 MB       | 8.14 MB       |


---

### 3. **Precision**

| Model               | Local         | EC2           |
|---------------------|---------------|---------------|
| mpnet               | 74.27%        | 74.27%        |
| MiniLM              | 73.93%        | 73.93%        |

**Interpretation**: No hardware dependence.

---

### 4. **Latency (Sequential, 1 Worker)**

| Model               | Local         | EC2           | Difference        |
|---------------------|---------------|---------------|-------------------|
| mpnet               | 0.1264s       | 0.0533s       | -58%              |
| MiniLM              | 0.0529s       | 0.0141s       | -73%              |

---

### 5. **Throughput (Sequential, 1 Worker)**

| Model               | Local         | EC2           | Difference        |
|---------------------|---------------|---------------|-------------------|
| mpnet               | 7.78 q/s      | 18.57 q/s     | +139%             |
| MiniLM              | 18.10 q/s     | 68.41 q/s     | +278%             |

**Interpretation**: Always better performance on smaller models. 

---

### 6. **Parallelism Scaling**

#### `mpnet`

- **Local**: Peaks at 21.86 q/s (16 workers)
- **EC2**: Peaks early at 23.59 q/s (2 workers), then drops

#### `MiniLM`

- **Local**: Peaks at 52.90 q/s (16 workers)
- **EC2**: Peaks early at 85.58 q/s (2 workers), then declines

**Interpretation**:
- EC2 scales poorly beyond 2 workers — likely fewer effective cores or higher contention.
- Local machine scales better with 8–16 workers, suggesting more CPU cores.

---

## Key Bottlenecks and Hardware Influence

- **Embedding Speed**:
  - EC2 dramatically faster, but still CPU-bound.
  - Maybe move to more powerful instance ? GPU. 

- **Sequential Performance**:
  - EC2 outperforms in latency and throughput at low parallelism.

- **Parallel Scaling**:
  - EC2: limited by 4 vCPUs.
  - Local: more cores enable better multi-threaded performance. I have 16 CPUs locally. 

- **Latency Degradation**:
  - EC2 shows sharper latency increases under parallelism, likely due to resource saturation.

---

## Conclusion

- **EC2 Advantages**:
  - Faster embedding
  - Excellent single-thread performance (ideal for low-parallelism)

- **Local Machine Advantages**:
  - Better parallel scalability
  - Suitable for high-throughput tasks with many workers

- **Hardware Bottlenecks**:
  - EC2 instance likely has 1–2 vCPUs → limits scaling
  - MongoDB performance and CPU-bound embedding are system-wide constraints