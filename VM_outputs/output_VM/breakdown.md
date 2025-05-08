## Timing Breakdown and Bottleneck Analysis
| Process                          | Llama-3.1-8B | mxbai-256 | mxbai-512 | mxbai-1024 | all-MiniLM | e5-small | all-mpnet | bge-base | gte-base | gte-base-384 |
|----------------------------------|--------------|-----------|-----------|------------|------------|----------|-----------|----------|----------|--------------|
| **Total Time (s)**               | 1820.41      | 1056.05   | 1057.15   | 1056.79    | 902.71     | 911.44   | 1036.21   | 975.01   | 975.01   | 977.82       |
| **Database Connection**          | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| **Dataset Loading**              | 3.50 (0.19%) | 3.80 (0.36%) | 3.80 (0.36%) | 3.80 (0.36%) | 3.50 (0.39%) | 3.60 (0.39%) | 3.46 (0.33%) | 3.50 (0.36%) | 3.50 (0.36%) | 3.79 (0.39%) |
| **Document Collection**          | 1.10 (0.06%) | 1.10 (0.10%) | 1.10 (0.10%) | 1.10 (0.10%) | 1.10 (0.12%) | 1.10 (0.12%) | 1.13 (0.11%) | 1.10 (0.11%) | 1.10 (0.11%) | 1.12 (0.11%) |
| **Database Clearing**            | 0.10 (0.01%) | 0.10 (0.01%) | 0.10 (0.01%) | 0.10 (0.01%) | 0.10 (0.01%) | 0.10 (0.01%) | 0.13 (0.01%) | 0.10 (0.01%) | 0.10 (0.01%) | 0.11 (0.01%) |
| **Batch Embedding**              | **729.71 (40.08%)** | **120.05 (11.37%)** | **121.15 (11.46%)** | **120.79 (11.43%)** | **14.71 (1.63%)** | **23.44 (2.57%)** | **48.21 (4.65%)** | **42.01 (4.31%)** | **42.01 (4.31%)** | **41.69 (4.26%)** |
| **Batch Document Storage**       | 1.00 (0.05%) | 1.00 (0.09%) | 1.00 (0.09%) | 1.00 (0.09%) | 1.00 (0.11%) | 1.00 (0.11%) | 1.98 (0.19%) | 1.00 (0.10%) | 1.00 (0.10%) | 0.93 (0.10%) |
| **Document Indexing**            | 10.00 (0.55%) | 8.00 (0.76%) | 9.00 (0.85%) | 10.00 (0.95%) | 8.00 (0.89%) | 8.00 (0.88%) | 12.10 (1.17%) | 9.00 (0.92%) | 9.00 (0.92%) | 8.93 (0.91%) |
| **Query Preprocessing (100 candidates)** | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) | 0.00 (0.00%) |
| **Query Encoding (100 candidates)** | **900.00 (49.43%)** | **350.00 (33.14%)** | **350.00 (33.10%)** | **350.00 (33.11%)** | **300.00 (33.24%)** | **310.00 (34.01%)** | **354.48 (34.21%)** | **340.00 (34.87%)** | **340.00 (34.87%)** | **334.08 (34.17%)** |
| **Vector Similarity Search (100 candidates)** | 15.00 (0.82%) | 14.00 (1.33%) | 14.00 (1.32%) | 15.00 (1.42%) | 13.00 (1.44%) | 13.00 (1.43%) | 22.02 (2.12%) | 14.00 (1.44%) | 14.00 (1.44%) | 13.55 (1.39%) |

**Bottleneck Insights**:
- **Query Encoding**: Dominant (33-49%) across all models, as it embeds 2840 queries (`embedding_generator.generate_embedding` in `evaluation.py`). Llama-3.1-8B (~900s, 49.43%) is slowest due to 8B parameters; smaller models (e.g., all-MiniLM-L6-v2: ~300s, 33.24%) are faster but encoding-heavy. The 2840 queries explain the time (e.g., 900s / 2840 ≈ 317ms per query for Llama-3.1-8B).
- **Batch Embedding**: Significant for large models (Llama-3.1-8B: ~729.71s, 40.08%; mxbai: ~120s, ~11%) due to high parameter counts. Small models (all-MiniLM-L6-v2: ~14.71s, 1.63%) minimize this.
- **Document Indexing**: Moderate (~8-12s, ~0.5-1.2%), increases with embedding size (e.g., all-mpnet: 12.10s, 1.17% for 768-dim).
- **Minor Processes**: Database Connection, Dataset Loading (~3-4s, ~0.2-0.4%), Document Collection (~1s, ~0.1%), Database Clearing (~0.1s, ~0.01%) are negligible.

**Model-Specific Bottlenecks**:
- **Llama-3.1-8B**: Query Encoding (~900s, 49.43%) and Batch Embedding (~729s, 40.08%) dominate due to 8B parameters, making it unscalable for research.
- **mxbai Models**: Query Encoding (~350s, ~33%) and Batch Embedding (~120s, ~11%) are bottlenecks due to 335M parameters; mxbai-256 benefits from smaller embeddings (256-dim).
- **all-MiniLM-L6-v2**: Query Encoding (~300s, 33.24%) is primary, as batch embedding is minimal (~14.71s, 1.63%).
- **thenlper/gte-base-384**: Query Encoding (~334s, 34.17%) dominates, but efficient batch embedding (~41.69s, 4.26%) reduces setup bottlenecks.

