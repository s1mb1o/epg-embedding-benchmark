# Results — EN↔RU↔HY Benchmark

## Ranked by Cross-Language Mean Score

| #  | Backend | Model | Cross-lang Mean | HY-HY Mean | Time/text (s) |
|----|---------|-------|:--------------:|:----------:|:-------------:|
| 1  | st    | `intfloat/multilingual-e5-large-instruct`     | **0.900** | **0.976** | 0.51 |
| 2  | st    | `intfloat/multilingual-e5-large`              | 0.881 | 0.964 | 0.49 |
| 3  | st    | `intfloat/multilingual-e5-base`               | 0.876 | 0.958 | 0.36 |
| 4  | st    | `sentence-transformers/LaBSE`                 | 0.800 | 0.934 | 0.45 |
| 5  | st    | `Metric-AI/armenian-text-embeddings-1`        | 0.798 | 0.910 | 0.30 |
| 6  | st    | `intfloat/e5-large-v2`                        | 0.776 | 0.833 | 0.32 |
| 7  | flag  | `BAAI/bge-m3`                                 | 0.767 | 0.849 | **0.27** |
| 8  | st    | `BAAI/bge-m3`                                 | 0.767 | 0.849 | 0.42 |
| 9  | st    | `intfloat/e5-large`                           | 0.751 | 0.863 | 0.31 |
| 10 | st    | `paraphrase-multilingual-mpnet-base-v2`        | 0.709 | 0.762 | 0.36 |
| 11 | st    | `distiluse-base-multilingual-cased`            | 0.708 | 0.749 | 0.28 |
| 12 | st    | `paraphrase-multilingual-MiniLM-L12-v2`        | 0.668 | 0.752 | 0.32 |
| 13 | openai | `text-embedding-3-large`                     | 0.338 | 0.597 | 0.18 |
| 14 | st    | `all-MiniLM-L6-v2`                            | 0.141 | 0.460 | 0.41 |

_st = sentence-transformers · flag = FlagEmbedding · Hardware: M2 Max, 32 GB RAM_

Full per-title data: [`results/benchmark_results.csv`](results/benchmark_results.csv)

---

## Per-Entry Breakdown: Selected Models

### intfloat/multilingual-e5-large-instruct — rank 1

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.9271   0.8827   0.9007   0.9035
morning_show               0.9236   0.8881   0.9252   0.9123
documentary_premiere       0.9292   0.8722   0.9175   0.9063
live_football              0.9077   0.8768   0.9289   0.9045
cooking_competition        0.9271   0.8844   0.8997   0.9038
movie_road_home            0.9033   0.8335   0.8873   0.8747
movie_secret_ararat        0.8976   0.8644   0.9158   0.8926
------------------------------------------------------------
overall_mean                                       0.8997

hy-hy:  0.9811 / 0.9773 / 0.9606 / 0.9828  ->  mean 0.9755
```

Highest and most consistent across all title types including domain-specific abbreviations (`к/ф`, `ֆ/ֆ`).

### intfloat/multilingual-e5-base — rank 3

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.8800   0.8694   0.8821   0.8772
morning_show               0.8749   0.8631   0.8883   0.8754
documentary_premiere       0.9115   0.8510   0.8602   0.8742
live_football              0.8781   0.8178   0.8830   0.8596
cooking_competition        0.8516   0.8980   0.8837   0.8778
movie_road_home            0.8842   0.8408   0.9016   0.8755
movie_secret_ararat        0.8986   0.8644   0.9230   0.8953
------------------------------------------------------------
overall_mean                                       0.8764

hy-hy:  0.9900 / 0.9489 / 0.9475 / 0.9472  ->  mean 0.9584
```

Recommended for production: 28% faster than the large-instruct variant with only -2.6% cross-language accuracy.

### openai/text-embedding-3-large — rank 13

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.6134   0.1500   0.1947   0.3194
morning_show               0.5808   0.1869   0.2250   0.3309
documentary_premiere       0.6072   0.1406   0.1995   0.3158
live_football              0.5920   0.1851   0.2016   0.3262
cooking_competition        0.7337   0.0786   0.1753   0.3292
movie_road_home            0.5598   0.1144   0.1959   0.2900
movie_secret_ararat        0.6689   0.3226   0.3659   0.4525
------------------------------------------------------------
overall_mean                                       0.3377

hy-hy:  0.6353 / 0.7817 / 0.4403 / 0.5300  ->  mean 0.5968
```

EN-RU alignment is serviceable (0.56–0.73), but Armenian vectors collapse to near-random (en-hy: 0.08–0.32).
Cost and brand do not predict performance on low-resource languages.
