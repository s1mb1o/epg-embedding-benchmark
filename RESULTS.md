# Results — EN↔RU↔HY similarity

## Summary

| # | Model | Cross-lang Mean | HY-HY Mean |
|---|-------|:--------------:|:----------:|
| 1 | `intfloat/multilingual-e5-base` | **0.876** | **0.958** |
| 2 | `sentence-transformers/LaBSE` | 0.800 | 0.934 |
| 3 | `Metric-AI/armenian-text-embeddings-1` | 0.798 | 0.910 |
| 4 | `paraphrase-multilingual-MiniLM-L12-v2` | 0.668 | 0.752 |
| 5 | `openai/text-embedding-3-large` | 0.338 | 0.597 |
| 6 | `all-MiniLM-L6-v2` | 0.141 | 0.460 |

---

## st/intfloat/multilingual-e5-base

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

Consistent across all title types — highest overall cross-language alignment.

## st/sentence-transformers/LaBSE

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

Strong on common news/sports titles; drops on Armenian movie titles with proper nouns.

## st/Metric-AI/armenian-text-embeddings-1

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.8140   0.8309   0.8576   0.8342
morning_show               0.7879   0.8332   0.8464   0.8225
documentary_premiere       0.8774   0.7471   0.7678   0.7975
live_football              0.8313   0.7638   0.9168   0.8373
cooking_competition        0.7747   0.9308   0.8120   0.8392
movie_road_home            0.7119   0.6218   0.7194   0.6844
movie_secret_ararat        0.7599   0.7145   0.8275   0.7673
------------------------------------------------------------
overall_mean                                       0.7975

hy-hy:  0.9764 / 0.8778 / 0.9013 / 0.8833  ->  mean 0.9097
```

Armenian-tuned model shows strong HY alignment. Trails e5-base on EN↔RU but comparable cross-language.

## st/paraphrase-multilingual-MiniLM-L12-v2

```
ID                          en-ru    en-hy    ru-hy     mean
------------------------------------------------------------
evening_news               0.9483   0.5201   0.6052   0.6912
morning_show               0.9179   0.8434   0.9029   0.8881
documentary_premiere       0.9409   0.2385   0.2749   0.4848
live_football              0.7698   0.6436   0.7711   0.7282
cooking_competition        0.6724   0.8823   0.7368   0.7638
movie_road_home            0.5210   0.4204   0.7277   0.5564
movie_secret_ararat        0.5032   0.2966   0.8989   0.5662
------------------------------------------------------------
overall_mean                                       0.6684

hy-hy:  0.9768 / 0.5491 / 0.6601 / 0.8228  ->  mean 0.7522
```

Good EN↔RU but HY alignment is inconsistent — documentary and proper noun titles drop badly.

## openai/text-embedding-3-large

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

EN↔RU is decent (0.58–0.73) but Armenian vectors are near-random (0.08–0.19). This is the key finding:
**a flagship commercial model fails on a low-resource language.**

## st/all-MiniLM-L6-v2

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

English-only baseline — completely fails cross-lingual matching. Included only for reference.
