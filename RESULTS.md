# Results — EN↔RU similarity

## st/all-MiniLM-L6-v2

```
EN                        RU                                      sim
Evening News              Вечерние новости                     0.0827
Morning Show              Утреннее шоу                         0.1195
Live Football             Прямая трансляция футбола            0.0931
Documentary Premiere      Премьера документального фильма      0.0949
Cooking Competition       Кулинарное состязание                0.0600
mean: 0.0900
```

English-centric model completely fails cross-lingual alignment — scores near zero.

## st/paraphrase-multilingual-MiniLM-L12-v2

```
EN                        RU                                      sim
Evening News              Вечерние новости                     0.9483
Morning Show              Утреннее шоу                         0.9179
Live Football             Прямая трансляция футбола            0.7698
Documentary Premiere      Премьера документального фильма      0.9409
Cooking Competition       Кулинарное состязание                0.6724
mean: 0.8498
```

## st/paraphrase-multilingual-mpnet-base-v2

```
EN                        RU                                      sim
Evening News              Вечерние новости                     0.9433
Morning Show              Утреннее шоу                         0.7748
Live Football             Прямая трансляция футбола            0.8291
Documentary Premiere      Премьера документального фильма      0.6391
Cooking Competition       Кулинарное состязание                0.7764
mean: 0.7925
```

## openai/text-embedding-3-small

```
EN                        RU                                      sim
Evening News              Вечерние новости                     0.6923
Morning Show              Утреннее шоу                         0.5886
Live Football             Прямая трансляция футбола            0.5911
Documentary Premiere      Премьера документального фильма      0.6893
Cooking Competition       Кулинарное состязание                0.6058
mean: 0.6334
```

---

## Summary

`all-MiniLM-L6-v2` is useless for cross-lingual matching (mean 0.09) — English-only model has no Russian alignment at all. `paraphrase-multilingual-MiniLM-L12-v2` wins at 0.85. OpenAI `text-embedding-3-small` underperforms open-source multilingual models (0.63). Next: add Armenian.
