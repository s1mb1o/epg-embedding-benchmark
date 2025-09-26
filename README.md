# Multilingual Embedding Benchmark: EN · RU · HY

> Evaluating sentence embedding models for cross-lingual TV program guide matching across **English**, **Russian**, and **Armenian** — a production RecSys problem with a low-resource language twist.

`hy` = Armenian (`հայ.`), not Hungarian. Armenian is a genuinely low-resource language that most commercial models handle poorly.

---

## TL;DR

| Winner | Cross-lang score | HY score |
|--------|:---------------:|:--------:|
| `intfloat/multilingual-e5-large-instruct` | **0.900** | **0.976** |

**Surprise:** `openai/text-embedding-3-large` scores only **0.34** — below all dedicated multilingual models in this benchmark — because Armenian is severely underrepresented in its training data.

---

## Background

Built while developing a content recommendation system for an IPTV/OTT operator whose platform serves TV program guides in three languages: English, Russian, and Armenian.

Every IPTV operator ingests EPG data from multiple sources. The same program arrives with different titles, different transliterations, and in different languages — all needing to be matched together. A RecSys that can't do cross-lingual matching produces poor recommendations for non-English content.

Armenian (`hy`) is particularly challenging:
- Non-Latin, non-Cyrillic script (unique Armenian alphabet)
- Low-resource language: underrepresented in most embedding model training data
- Domain-specific abbreviations: `ֆ/ֆ` (Feature Film), `ու.` (Documentary)

---

## Evaluation Setup

### Test Dataset

7 representative TV EPG title triplets (EN/RU/HY) + 4 Armenian synonym pairs.

### Metric

**Mean cosine similarity** across all three cross-lingual pairs (EN↔RU, EN↔HY, RU↔HY).

### Hardware

MacBook M2 Max, 32 GB RAM — all models run locally via CPU/Metal.

---

## Results

See [RESULTS.md](RESULTS.md).

---

## Key Findings

### 1. The multilingual-e5 family is the clear winner for EN+RU+HY

`intfloat/multilingual-e5-large-instruct` achieves 0.90 cross-language mean and 0.98 HY-HY consistency. The base model offers the best accuracy/cost trade-off for production use.

### 2. OpenAI text-embedding-3-large fails on Armenian

Despite being a flagship commercial model, it scores **0.34 overall** — below all dedicated multilingual models tested. Armenian is a low-resource language; without sufficient HY parallel training data, the model cannot anchor Armenian phrases near their EN/RU counterparts.

**Lesson:** cost and brand do not predict performance on low-resource languages. Always benchmark for your specific language mix.

### 3. LaBSE is the best non-e5 fallback

Google's Language-agnostic BERT Sentence Embedding scores 0.80 cross-language with solid HY performance (0.93).

### 4. FlagEmbedding is the fastest local inference path

`BAAI/bge-m3` via FlagEmbedding runs at 0.27s/text — 35% faster than the same model through sentence-transformers — with identical embedding quality.

---

## Recommendations

For a production EN+RU+HY IPTV/OTT RecSys:

| Priority | Model | Rationale |
|----------|-------|-----------|
| **Default** | `intfloat/multilingual-e5-base` | Best accuracy/size/speed trade-off |
| **Max accuracy** | `intfloat/multilingual-e5-large-instruct` | Top scores, +40% latency |
| **EN+RU only** | `openai/text-embedding-3-large` | Viable if Armenian is absent |
| **Resource-constrained** | `BAAI/bge-m3` via FlagEmbedding | Fastest local inference, 0.27s/text |

---

## Reproduce

```bash
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

## About

Developed as part of building a content recommendation system for an IPTV/OTT platform serving multi-language EPG data in English, Russian, and Armenian. The results directly informed the production model selection.

---

_Benchmarked on Apple M2 Max, 32 GB RAM · Models from [Hugging Face Hub](https://huggingface.co)_
