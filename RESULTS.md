# Results — EN↔RU↔HY Benchmark

## Ranked by Cross-Language Mean Score

| # | Backend | Model | Cross-lang Mean | HY-HY Mean | Time/text (s) |
|---|---------|-------|:--------------:|:----------:|:-------------:|
| 1  | st | `intfloat/multilingual-e5-large-instruct` | **0.900** | **0.976** | 0.51 |
| 2  | st | `intfloat/multilingual-e5-large` | 0.881 | 0.964 | 0.49 |
| 3  | st | `intfloat/multilingual-e5-base` | 0.876 | 0.958 | 0.36 |
| 4  | st | `sentence-transformers/LaBSE` | 0.800 | 0.934 | 0.45 |
| 5  | st | `Metric-AI/armenian-text-embeddings-1` | 0.798 | 0.910 | 0.30 |
| 6  | st | `intfloat/e5-large-v2` | 0.776 | 0.833 | 0.32 |
| 7  | flag | `BAAI/bge-m3` | 0.767 | 0.849 | **0.27** |
| 8  | st | `BAAI/bge-m3` | 0.767 | 0.849 | 0.42 |
| 9  | st | `intfloat/e5-large` | 0.751 | 0.863 | 0.31 |
| 10 | st | `paraphrase-multilingual-mpnet-base-v2` | 0.709 | 0.762 | 0.36 |
| 11 | st | `distiluse-base-multilingual-cased` | 0.708 | 0.749 | 0.28 |
| 12 | st | `paraphrase-multilingual-MiniLM-L12-v2` | 0.668 | 0.752 | 0.32 |
| 13 | openai | `text-embedding-3-large` | 0.338 | 0.597 | 0.18 |
| 14 | st | `all-MiniLM-L6-v2` | 0.141 | 0.460 | 0.41 |

_st = sentence-transformers · flag = FlagEmbedding · Hardware: M2 Max, 32 GB RAM_

Full per-title data: [`results/benchmark_results.csv`](results/benchmark_results.csv)
