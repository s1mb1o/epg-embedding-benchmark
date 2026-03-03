# Results — EN↔RU↔HY Benchmark

## Ranked by Cross-Language Mean Score

| #  | Backend | Model | Cross-lang Mean | HY-HY Mean | Time/text (s) |
|----|---------|-------|:--------------:|:----------:|:-------------:|
| 1  | st    | `intfloat/multilingual-e5-large-instruct`     | **0.900** | **0.975** | 0.51 |
| 2  | st    | `intfloat/multilingual-e5-large`              | 0.881 | 0.964 | 0.54 |
| 3  | st    | `intfloat/multilingual-e5-base`               | 0.876 | 0.958 | 0.42 |
| 4  | st    | `sentence-transformers/LaBSE`                 | 0.800 | 0.934 | 0.51 |
| 5  | st    | `Metric-AI/armenian-text-embeddings-1`        | 0.798 | 0.910 | 0.33 |
| 6  | st    | `intfloat/e5-large-v2`                        | 0.776 | 0.833 | 0.37 |
| 7  | flag  | `BAAI/bge-m3`                                 | 0.767 | 0.849 | 0.28 |
| 8  | st    | `BAAI/bge-m3`                                 | 0.767 | 0.849 | 0.51 |
| 9  | st    | `intfloat/e5-large`                           | 0.751 | 0.863 | 0.39 |
| 10 | st    | `paraphrase-multilingual-mpnet-base-v2`        | 0.709 | 0.762 | 0.45 |
| 11 | st    | `distiluse-base-multilingual-cased`            | 0.708 | 0.749 | 0.39 |
| 12 | st    | `paraphrase-multilingual-MiniLM-L12-v2`        | 0.668 | 0.752 | 0.39 |
| 13 | openai | `text-embedding-3-large`                     | 0.338 | 0.666 | **0.18** |
| 14 | st    | `all-MiniLM-L6-v2`                            | 0.141 | 0.460 | 0.33 |

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
movie_road_home            0.9033   0.8335   0.8872   0.8747
movie_secret_ararat        0.8976   0.8644   0.9157   0.8925
------------------------------------------------------------
overall_mean                                       0.8996

hy-hy:  0.9811 / 0.9773 / 0.9606 / 0.9827  ->  mean 0.9754
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

### sentence-transformers/LaBSE — rank 4

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.9478   0.8980   0.8923   0.9127
morning_show               0.8931   0.9106   0.9246   0.9094
documentary_premiere       0.8949   0.8215   0.8215   0.8460
live_football              0.8945   0.8986   0.9500   0.9144
cooking_competition        0.7413   0.9093   0.7042   0.7849
movie_road_home            0.5361   0.3798   0.7742   0.5633
movie_secret_ararat        0.5918   0.6177   0.7902   0.6666
------------------------------------------------------------
overall_mean                                       0.7996

hy-hy:  0.9018 / 0.9641 / 0.9178 / 0.9512  ->  mean 0.9337
```

Strong on common news/sports titles; drops sharply on movie entries with proper nouns and domain prefixes (`к/ф`, `ֆ/ֆ`).
Best non-e5 fallback when the intfloat family is unavailable.

### paraphrase-multilingual-mpnet-base-v2 — rank 10

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.9433   0.5020   0.6292   0.6915
morning_show               0.7748   0.9333   0.8596   0.8559
documentary_premiere       0.6391   0.2330   0.6501   0.5074
live_football              0.8291   0.8318   0.9589   0.8732
cooking_competition        0.7764   0.9574   0.8191   0.8509
movie_road_home            0.5258   0.5835   0.6289   0.5794
movie_secret_ararat        0.5413   0.5558   0.7073   0.6015
------------------------------------------------------------
overall_mean                                       0.7085

hy-hy: mean 0.7622
```

Inconsistent: some pairs score very high (cooking_competition en-hy 0.96, live_football ru-hy 0.96) while others collapse (documentary_premiere en-hy 0.23).
Usable as a lightweight baseline but not reliable enough for production EPG matching.

### openai/text-embedding-3-large — rank 13

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.6133   0.1500   0.1946   0.3193
morning_show               0.5811   0.1871   0.2251   0.3311
documentary_premiere       0.6072   0.1406   0.1996   0.3158
live_football              0.5890   0.1850   0.2013   0.3251
cooking_competition        0.7337   0.0782   0.1746   0.3288
movie_road_home            0.5598   0.1144   0.1959   0.2900
movie_secret_ararat        0.6690   0.3226   0.3664   0.4527
------------------------------------------------------------
overall_mean                                       0.3375

hy-hy:  0.6355 / 0.7819 / 0.4403 / 0.8079  ->  mean 0.6664
```

en-ru scores hover around 0.6 — OpenAI aligns Russian and English reasonably well. en-hy and ru-hy stay near 0.15–0.20:
Armenian is effectively drifting in the embedding space; the overall mean of 0.34 reflects that imbalance.

> text-embedding-3-large is very strong overall, but Armenian is a low-resource language
> in OpenAI's training mix. Without much HY-only or HY↔EN parallel data in their corpus,
> the model just can't anchor Armenian phrases near their English/Russian counterparts.
> Prefix cases (к/ф, ֆ/ֆ) add extra tokens that many models — including OpenAI's — may
> have rarely seen, which drags the HY vectors further away.
> — ChatGPT

### all-MiniLM-L6-v2 — rank 14

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.0827   0.1045   0.1827   0.1233
morning_show               0.1195   0.1232   0.2626   0.1684
documentary_premiere       0.0949   0.0625   0.1532   0.1035
live_football              0.0931   0.1274   0.1599   0.1268
cooking_competition        0.0600   0.1110   0.3180   0.1630
movie_road_home            0.0150  -0.0043   0.2328   0.0812
movie_secret_ararat        0.1223   0.2080   0.3214   0.2173
------------------------------------------------------------
overall_mean                                       0.1405

hy-hy:  0.9031 / 0.2893 / 0.1948 / 0.4532  ->  mean 0.4601
```

> all-MiniLM-L6-v2 isn't tuned for cross-lingual tasks. Armenian especially is
> underrepresented, so the embeddings drift apart despite true semantic matches.
> ru-hy scores consistently beat en-ru/en-hy, suggesting the model catches a bit of
> regional similarity but still lacks solid multilingual understanding. Movie-prefixed
> rows (к/ф, ֆ/ֆ) drop even further; the model likely hasn't seen those prefixes, so
> it treats the strings as unrelated.
> — ChatGPT

Included as a baseline only. Do not use for multilingual EPG matching.
