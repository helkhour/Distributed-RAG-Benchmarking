Test 3 : 

Multimodal testing running on an AWS c6id.xlarge VM (4 vCPUs, 8 GiB RAM, NVMe SSD)

Normal system - no concurrent or multiple runs
just testing the effect of the embedding model choice on the system. 

---

## Same Parameters, Different Dimensions (using mxbai - 335M parms - flexible output size)

general-purpose embedding tasks (semantic similarity, retrieval, clustering, etc.) :
mixedbread-ai/mxbai-embed-large-v1-{256,512,1024}
335M params
supports variable vector lengths (256, 512, 1024)

| Model                                | Embedding Size | Precision (%) | Latency (s/query) | Throughput (q/s) | DB Size (MB) | Embedding Time (s) |
|-------------------------------------|----------------|----------------|-------------------|------------------|--------------|---------------------|
| mxbai-embed-large-v1-256            | 256            | 79.49          | 0.1588            | 6.28             | 5.67         | 537.05              |
| mxbai-embed-large-v1-512            | 512            | 81.71          | 0.1552            | 6.42             | 10.61        | 534.73              |
| mxbai-embed-large-v1-1024           | 1024           | **82.39**      | **0.1543**        | **6.46**         | 20.53        | **531.44**          |

**Interpretation**:
- Larger embedding sizes improve precision marginally.
- Latency and throughput remain stable across sizes.
- Storage requirements and embedding time increase linearly.

---

## Different Parameters, Same Dimension (384)

all-MiniLM-L6-v2:
    Fast, lightweight embeddings for semantic similarity and sentence classification

e5-small-v2: 
    ecifically trained for information retrieval
    ained on RAG-style datasets

gte-base-384:
    trained for tasks like : semantic similarity, classification, clustering

| Model                                | Embedding Size | Precision (%) | Latency (s/query) | Throughput (q/s) | DB Size (MB) | Embedding Time (s) |
|-------------------------------------|----------------|----------------|-------------------|------------------|--------------|---------------------|
| all-MiniLM-L6-v2                    | 384            | 73.93          | **0.0138**        | **69.80**        | 8.14         | **37.03**           |
| e5-small-v2                         | 384            | **82.99**      | 0.0261            | 37.55            | 8.14         | 62.91               |
| gte-base-384                        | 384            | 78.89          | 0.0521            | 19.02            | 8.14         | 164.62              |

**Interpretation**:
- `e5-small-v2` has the highest precision in this group.
- `MiniLM` offers exceptional speed but meh precision.

---

## Same Parameters, Same Dimension (768)

sentence-transformers/all-mpnet-base-v2
    high-quality general-purpose sentence embeddings - trained for semantic tasks
BAAI/bge-base-en-v1.5
    trained explicitly for retrieval tasks
thenlper/gte-base :
    all purpose embedding 

| Model                                | Embedding Size | Precision (%) | Latency (s/query) | Throughput (q/s) | DB Size (MB) | Embedding Time (s) |
|-------------------------------------|----------------|----------------|-------------------|------------------|--------------|---------------------|
| all-mpnet-base-v2                   | 768            | 74.27          | 0.0566            | 17.49            | 15.56        | 167.84              |
| bge-base-en-v1.5                    | 768            | 80.60          | **0.0547**        | **18.11**        | 15.56        | 161.67              |
| gte-base                            | 768            | **82.48**      | 0.0554            | 17.88            | 15.56        | **163.59**          |

**Interpretation**:
- All models offer comparable performance in terms of latency and throughput.
- `gte-base` outperforms others slightly in precision.
- Trade-offs are subtle; model architecture plays a role in retrieval success.

---



Overall Insights
- Precision improves with larger embedding dimensions **when using the same model family**.
- Latency and throughput are strongly tied to model architecture rather than size.
- **Best balance of speed and precision**: `e5-small-v2`
- **Best overall precision**: `gte-base` (768) and `mxbai-1024`



- From 256 → 512 → 1024 (mxbai), precision only increased ~3%. Latency and storage costs increased linearly.

- mxbai-* models took ~530 seconds to embed ~1500 documents. In contrast, MiniLM did it in just ~37 seconds.

- Model Architecture Matters More Than Size. All models had 384 or 768 dimensions. Yet, precision ranged from 73.93% (MiniLM) to 82.99% (e5).

- retrieval Latency varies greatly : MiniLM: ~0.01s/query --- GTE-384: ~0.05s/query --- mxbai-1024: ~0.15s/query




identified bottlenecks : 
embedding time for large models 
augmented latency for high dim vectors
precision lower in non-retrieval-optimized models

next focus on : 
long wait for index readiness
Large vectors increase DB size