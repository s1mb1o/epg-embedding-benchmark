# Multilingual Embedding Benchmark: EN · RU · HY

> Evaluating sentence embedding models for cross-lingual TV program guide matching across **English**, **Russian**, and **Armenian** — a production RecSys problem with a low-resource language twist.

`hy` = Armenian (`հայ.`), not Hungarian. This matters because Armenian is a genuinely low-resource language that most commercial models handle poorly.

---

## TL;DR

| Winner | Cross-lang score | HY score |
|--------|:---------------:|:--------:|
| `intfloat/multilingual-e5-large-instruct` | **0.900** | **0.976** |

**Surprise:** `openai/text-embedding-3-large` scores only **0.34** — below all dedicated multilingual models in this benchmark — because Armenian is severely underrepresented in its training data.

---

## Background

Built while developing a content recommendation system for an IPTV/OTT operator whose platform serves TV program guides (EPG) in three languages: English, Russian, and Armenian.

Every IPTV operator ingests EPG data from multiple sources. The same program arrives with different titles, different transliterations, and in different languages — all needing to be matched together. A RecSys that can't do cross-lingual matching produces poor recommendations for non-English content.

Armenian (`hy`) is particularly challenging:
- Non-Latin, non-Cyrillic script (unique Armenian alphabet: Հ, Ա, Յ, Ե, ...)
- Low-resource language: underrepresented in most embedding model training data
- Domain-specific abbreviations: `ֆ/ֆ` (Feature Film), `ու.` (Documentary)

This benchmark measures how well each model handles **semantic alignment** of the same program title across all three languages.

---

## Evaluation Setup

### Test Dataset

7 representative TV EPG title triplets, covering news, sports, entertainment, and movies:

| ID | English | Russian | Armenian (հայ.) |
|----|---------|---------|-----------------|
| `evening_news` | Evening News | Вечерние новости | Երեկոյան լուրեր |
| `morning_show` | Morning Show | Утреннее шоу | Առավոտյան շոու |
| `documentary_premiere` | Documentary Premiere | Премьера документального фильма | Վավերագրական ֆիլմի պրեմիերա |
| `live_football` | Live Football | Прямая трансляция футбола | Ֆուտբոլի ուղիղ հեռարձակում |
| `cooking_competition` | Cooking Competition | Кулинарное состязание | Խոհարարական մրցույթ |
| `movie_road_home` | Feature Film: The Road Home | к/ф Дорога домой | ֆ/ֆ Տուն վերադարձ |
| `movie_secret_ararat` | Feature Film: Secret of Ararat | к/ф Тайна Арарата | ֆ/ֆ Արարատի գաղտնիքը |

The `movie_*` entries include domain-specific abbreviations (`к/ф` in Russian, `ֆ/ֆ` in Armenian for "Feature Film") that many models have never seen during training — a practical pain point in real EPG pipelines.

### Armenian Synonym Test

4 intra-Armenian synonym pairs to test `HY↔HY` consistency — same concept, two common phrasings:

| ID | Variant A | Variant B |
|----|-----------|-----------|
| `hy_news_bulletin` | Լուրերի թողարկում | Նորությունների թողարկում |
| `hy_live_broadcast` | Ուղիղ հեռարձակում | Ուղիղ եթեր |
| `hy_sports_wrap` | Մարզական ամփոփում | Սպորտային ամփոփագիր |
| `hy_family_movie` | Ընտանեկան ֆիլմ | Ընտանիքի համար նախատեսված ֆիլմ |

### Metric

**Mean cosine similarity** across all three cross-lingual pairs (EN↔RU, EN↔HY, RU↔HY). Score range:
- `1.0` = model places all three translations at the same point in embedding space
- `0.0` = model sees no relationship between semantically identical titles in different languages

### Hardware

MacBook M2 Max, 32 GB RAM — all models run locally via CPU/Metal, no GPU cluster required.

---

## Results

See [RESULTS.md](RESULTS.md).

---

## Key Findings

### 1. The multilingual-e5 family is the clear winner for EN+RU+HY

`intfloat/multilingual-e5-large-instruct` achieves 0.90 cross-language mean and 0.98 HY-HY consistency. The entire e5 family performs significantly better than all alternatives. The base model offers the best accuracy/cost trade-off for production use.

### 2. OpenAI text-embedding-3-large fails on Armenian

Despite being a flagship commercial model, it scores **0.34 overall** — below all dedicated multilingual models tested. Armenian is a low-resource language; without sufficient HY parallel training data, the model cannot anchor Armenian phrases near their EN/RU counterparts.

**Lesson:** cost and brand do not predict performance on low-resource languages. Always benchmark for your specific language mix.

### 3. Domain-specific abbreviations hurt all models

`к/ф` (Russian) and `ֆ/ֆ` (Armenian) for "Feature Film" are rare tokens that most models have seen infrequently. Even top-performing models score slightly lower on `movie_*` entries. Pre-processing to expand abbreviations would likely improve all scores.

### 4. LaBSE is the best non-e5 fallback

Google's Language-agnostic BERT Sentence Embedding scores 0.80 cross-language with solid HY performance (0.93). It's the most reliable choice if the intfloat/e5 family is unavailable.

### 5. Armenian-specialized model doesn't win overall

`Metric-AI/armenian-text-embeddings-1` (based on multilingual-e5) raises HY scores slightly but underperforms the full e5 family on EN↔RU alignment. Specialization for one language comes at a cost to others.

### 6. FlagEmbedding is the fastest local inference path

`BAAI/bge-m3` via FlagEmbedding runs at 0.27s/text — 35% faster than the same model through sentence-transformers — with identical embedding quality. Useful for latency-sensitive production deployments.

---

## Recommendations

For a production EN+RU+HY IPTV/OTT RecSys:

| Priority | Model | Rationale |
|----------|-------|-----------|
| **Default** | `intfloat/multilingual-e5-base` | Best accuracy/size/speed trade-off |
| **Max accuracy** | `intfloat/multilingual-e5-large-instruct` | Top scores, +40% latency |
| **EN+RU only** | `openai/text-embedding-3-large` | Viable if Armenian is absent |
| **Resource-constrained** | `BAAI/bge-m3` via FlagEmbedding | Fastest local inference, 0.27s/text |

**Avoid for HY:** `all-MiniLM-L6-v2` (English-centric), `openai/text-embedding-3-large` (Armenian alignment collapses to noise)

---

## Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single model
python benchmark.py --api st --model intfloat/multilingual-e5-base

# Run the full benchmark suite
./run_benchmark.sh

# With OpenAI models
OPENAI_API_KEY=sk-... ./run_benchmark.sh

# With custom phrase dataset
python benchmark.py --api st --model intfloat/multilingual-e5-base --phrases data/epg_phrases.json
```

### Backends

| Flag | Package | Notes |
|------|---------|-------|
| `--api st` | `sentence-transformers` | Local inference, most models |
| `--api flag` | `FlagEmbedding` | Faster local inference for BAAI/bge-m3 |
| `--api openai` | `openai` | Requires `OPENAI_API_KEY` env var |
| `--api ollama` | `requests` | Requires local `ollama serve` + `TEST_EMB_OLLAMA=1` |

---

## Repository Structure

```
.
├── benchmark.py          # Main evaluation script
├── run_benchmark.sh      # Harness to run all models sequentially
├── requirements.txt
├── data/
│   └── epg_phrases.json  # Test dataset: 7 EN/RU/HY triplets + 4 HY synonym pairs
└── results/
    └── benchmark_results.csv  # Pre-computed results from M2 Max
```

---

## About

Developed as part of building a content recommendation system for an IPTV/OTT platform serving multi-language EPG data in English, Russian, and Armenian. The results directly informed the production model selection.

The test dataset contains common TV program categories only; no proprietary EPG data is included.

---

_Benchmarked on Apple M2 Max, 32 GB RAM · Models from [Hugging Face Hub](https://huggingface.co)_
