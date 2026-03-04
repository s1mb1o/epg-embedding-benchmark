# Multilingual Embedding Benchmark: EN · RU · HY

> Evaluating sentence embedding models for cross-lingual TV program guide matching across **English**, **Russian**, and **Armenian** — a production RecSys problem with a low-resource language twist.

`hy` = Armenian (`հայ.`). This matters because Armenian is a genuinely low-resource language that most commercial models handle poorly.

---

## TL;DR

| Winner | Cross-lang score | HY score |
|--------|:---------------:|:--------:|
| `intfloat/multilingual-e5-large-instruct` | **0.900** | **0.975** |

**Surprise:** OpenAI `text-embedding-3-large` scores only **0.34** — below all dedicated multilingual models in this benchmark — because Armenian is severely underrepresented in its training data.

---

## Background

Built while developing a content recommendation system for an IPTV/OTT operator whose platform serves TV program guides (EPG) in three languages: English, Russian, and Armenian.

### Why these three languages?

Armenian IPTV/OTT EPG feeds use three languages simultaneously: **Armenian** (HY), **Russian** (RU), and **English** (EN). Armenian is the national language. Russian is widely spoken and dominates post-Soviet broadcast metadata. English is the lingua franca of international content. All three coexist in the same EPG feed, making trilingual matching a hard practical requirement rather than an academic exercise.

A 14-day snapshot of a production Armenian EPG feed (239 channels, 35,170 entries, 2025-09-18 to 2025-10-02) shows the language mix:

| Language | Entries | Share |
|----------|--------:|------:|
| Russian only | 18,876 | 53.7% |
| Armenian only | 10,519 | 29.9% |
| English only | 4,236 | 12.0% |
| Mixed (two languages) | 1,519 | 4.3% |

No single entry contained all three languages — each title is monolingual or bilingual, so a RecSys must align them purely through embedding similarity.

Every IPTV operator ingests EPG data from multiple sources. The same program arrives with different titles, different transliterations, and in different languages — all needing to be matched together. A RecSys that can't do cross-lingual matching produces poor recommendations for non-English content.

Armenian (`hy`) is particularly challenging:
- Non-Latin, non-Cyrillic script (unique Armenian alphabet: Հ, Ա, Յ, Ե, ...)
- Low-resource language: underrepresented in most embedding model training data
- Domain-specific abbreviations
- Mix of Armenian, Russian and English in EPG titles

### Domain-specific abbreviations

To quantify how prevalent this issue is, an analysis of a 14-day snapshot of a production EPG feed (unmapped raw titles, 2025-09-18 to 2025-10-02) was performed. Out of 11,298 Armenian program titles broadcasted, **29.2%** (3,297 titles) contained at least one domain-specific abbreviation. The most common were:

1. `Հ/Ն` (Հեռուստանովել / Herustanoevel - Telenovela/Soap Opera) — 1,971 occurrences
2. `Հ/Ս` (Հեռուստասերիալ / Herustaserial - TV Series) — 598 occurrences
3. `Գ/Ֆ` (Գեղարվեստական ֆիլմ / Gegharvestakan film - Feature Film) — 112 occurrences
4. `Հ/Շ` (Հեռուստաշոու / Herustashou - TV Show) — 88 occurrences
5. `Մ/Ս` (Մուլտսերիալ / Multserial - Animated Series) — 84 occurrences

This proves that failing to embed abbreviations correctly isn't an edge case; it actively degrades search and recommendation quality for a massive portion of daily linear TV content.

Other common abbreviations include:

  Feature Film
   * `ֆ/ֆ` (ֆիլմ / film) - Less formal, but sometimes used.

  Animations / Cartoons
   * `Մ/Ֆ` (Մուլտիպլիկացիոն ֆիլմ or Մուլտֆիլմ / Multiplikatsion film or Multfilm) - Standard for animated movies.

  Documentary
   * `Փ/Ֆ` (Փաստավավերագրական ֆիլմ / Phastavaveragrakan film) - Standard.
   * `Վ/Ֆ` (Վավերագրական ֆիլմ / Vaveragrakan film) - Standard (often used interchangeably with Փ/Ֆ).

  Show/Entertainment
   * `Ժ/Ծ` (Ժամանցային ծրագիր / Zhamantsayin tsragir) - Standard for entertainment programs.
   * `Թ/Շ` (Թոք շոու / Tok shou) - Standard for talk shows.

  News
   * `Լ/Ծ` (Լրատվական ծրագիր / Lratvakan tsragir) - Standard for news programs/broadcasts.

  Live Broadcasts
   * `Ու/Ե` (Ուղիղ եթեր / Ughigh yeter) — Live Broadcast / Live Air (Very common for news, sports, and events).
   * `Ու/Հ` (Ուղիղ հեռարձակում / Ughigh herardzakum) — Live Broadcast (Alternative to Ու/Ե).

  Specific Film Genres
   * `Կ/Ֆ` (Կարճամետրաժ ֆիլմ / Karchametrazh film) — Short Film.
   * `Պ/Ֆ` (Պատմական ֆիլմ / Patmakan film) — Historical Film.
   * `Ուս/Ֆ` (Ուսումնական ֆիլմ / Usumnakan film) — Educational Film.

  Specific Programs & Shows
   * `Մ/Հ` or `Մ/Ծ` (Մանկական հաղորդում / Mankakan haghordum or tsragir) — Children's Program.
   * `Գ/Հ` or `Գ/Ծ` (Գիտահանրամատչելի հաղորդում / Gitahanramatcheli haghordum) — Scientific / Educational Program.
   * `Ե/Ծ` (Երաժշտական ծրագիր / Yerazhshtakan tsragir) — Musical Program / Concert.
   * `Ս/Հ` (Սպորտային հաղորդում / Sportayin haghordum) — Sports Program.

This benchmark measures how well each model handles **semantic alignment** of the same program title across all three languages.

---

## Evaluation Setup

### Test Dataset

253 trilingual title triplets: 7 hand-crafted TV EPG entries covering news, sports, entertainment, and movies, plus 246 real Armenian titles (movies, animated films, TV series) from [TMDB](https://www.themoviedb.org/) (see [DATASET.md](DATASET.md)).

Hand-crafted EPG triplets:

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

MacBook M2 Max, 32 GB RAM — local models run via CPU/Metal; paid models (OpenAI, Cohere, Jina, Voyage) use remote APIs.

---

## Results

| # | Backend | Model | Year | Paid | Cross-lang | EN↔RU | EN↔HY | RU↔HY | HY↔HY | Abbrev | s/text |
|---|---------|-------|:----:|:----:|:----------:|:-----:|:-----:|:-----:|:-----:|:------:|:------:|
| 1 | st | `intfloat/multilingual-e5-large-instruct` | 2024 | | **0.900** | 0.916 | 0.872 | 0.911 | **0.975** | **0.975** | 0.48 |
| 2 | jina | `jina-embeddings-v3` | 2024 | $$ | 0.881 | 0.899 | 0.860 | 0.884 | 0.913 | 0.899 | **0.05** |
| 3 | st | `intfloat/multilingual-e5-large` | 2023 | | 0.881 | 0.880 | 0.864 | 0.899 | 0.964 | 0.940 | 0.59 |
| 4 | st | `intfloat/multilingual-e5-base` | 2023 | | 0.876 | 0.883 | 0.858 | 0.889 | 0.958 | 0.948 | 0.42 |
| 5 | st | `sentence-transformers/LaBSE` | 2022 | | 0.800 | 0.786 | 0.776 | 0.837 | 0.934 | 0.794 | 0.51 |
| 6 | st | `Metric-AI/armenian-text-embeddings-1` | 2024 | | 0.797 | 0.794 | 0.777 | 0.821 | 0.910 | 0.875 | 0.30 |
| 7 | st | `intfloat/e5-large-v2` | 2023 | | 0.776 | 0.754 | 0.767 | 0.808 | 0.833 | 0.920 | 0.39 |
| 8 | st | `jinaai/jina-embeddings-v3` | 2024 | | 0.771 | 0.821 | 0.700 | 0.792 | 0.826 | 0.928 | 0.66 |
| 9 | st | `BAAI/bge-m3` | 2024 | | 0.766 | 0.797 | 0.712 | 0.791 | 0.849 | 0.831 | 0.47 |
| 10 | flag | `BAAI/bge-m3` | 2024 | | 0.766 | 0.797 | 0.712 | 0.791 | 0.849 | 0.831 | 0.31 |
| 11 | cohere | `embed-multilingual-v3.0` | 2023 | $$ | 0.764 | 0.802 | 0.683 | 0.806 | 0.951 | 0.911 | 0.05 |
| 12 | st | `intfloat/e5-large` | 2022 | | 0.751 | 0.723 | 0.729 | 0.802 | 0.863 | 0.969 | 0.35 |
| 13 | voyage | `voyage-multilingual-2` | 2024 | $$ | 0.729 | 0.814 | 0.660 | 0.713 | 0.783 | 0.889 | 0.07 |
| 14 | st | `Alibaba-NLP/gte-multilingual-base` | 2024 | | 0.724 | 0.802 | 0.655 | 0.716 | 0.737 | 0.899 | 0.40 |
| 15 | st | `paraphrase-multilingual-mpnet-base-v2` | 2021 | | 0.709 | 0.719 | 0.657 | 0.750 | 0.762 | 0.793 | 0.46 |
| 16 | st | `distiluse-base-multilingual-cased` | 2020 | | 0.708 | 0.766 | 0.618 | 0.739 | 0.749 | 0.749 | 0.39 |
| 17 | st | `paraphrase-multilingual-MiniLM-L12-v2` | 2021 | | 0.668 | 0.753 | 0.549 | 0.703 | 0.752 | 0.796 | 0.42 |
| 18 | cohere | `embed-v4.0` | 2025 | $$ | 0.404 | 0.668 | 0.238 | 0.305 | 0.572 | 0.695 | 0.05 |
| 19 | openai | `text-embedding-3-large` | 2024 | $$ | 0.338 | 0.622 | 0.168 | 0.222 | 0.667 | 0.774 | 0.06 |
| 20 | st | `all-MiniLM-L6-v2` | 2021 | | 0.141 | 0.084 | 0.105 | 0.233 | 0.460 | 0.837 | 0.34 |

_253 triplets · 783 abbreviation duplets · st = sentence-transformers · flag = FlagEmbedding · $$ = paid API · Hardware: M2 Max, 32 GB RAM_

Rankings are identical to the original 7-triplet benchmark — the small hand-crafted dataset was representative after all. Scores are remarkably stable, confirming the original findings hold at 253 triplets.

Raw data: [results/benchmark_results.csv](results/benchmark_results.csv)

---

## Analysis

[analysis.ipynb](analysis.ipynb) generates five charts from the benchmark results:

### Models ranked by cross-lingual score

![Models ranked by cross-lingual score](results/scores_ranked.png)

### Accuracy vs Speed

![Accuracy vs Speed](results/accuracy_vs_speed.png)

### Cross-language score vs Armenian consistency

![Cross-language vs HY consistency](results/cross_lang_vs_hy.png)

### Per language-pair heatmap

![Per language-pair heatmap](results/heatmap.png)

### Abbreviation robustness

![Abbreviation robustness](results/abbrev_scores.png)

---

## Key Findings

### 1. The multilingual-e5 family is the clear winner for EN+RU+HY

`intfloat/multilingual-e5-large-instruct` achieves 0.90 cross-lingual mean and 0.975 HY↔HY consistency. The three multilingual-e5 variants (base, large, large-instruct) all score above 0.87 — significantly better than all other models. The base model offers the best accuracy/cost trade-off for production use.

### 2. Commercial API models underperform on Armenian

OpenAI `text-embedding-3-large` scores **0.34 overall** and Cohere `embed-v4.0` scores **0.40** — both below all dedicated multilingual models tested, despite being flagship commercial offerings. Armenian is a low-resource language; without sufficient HY parallel training data, these models cannot anchor Armenian phrases near their EN/RU counterparts. Cohere's older `embed-multilingual-v3.0` (0.76) fares much better, suggesting the newer v4 model may have regressed on low-resource languages.

**Lesson:** cost and brand do not predict performance on low-resource languages. Always benchmark for your specific language mix.

### 3. Domain-specific abbreviations hurt all models

EPG data is full of abbreviations like `к/ф` (Russian) and `Գ/Ֆ`, `ֆ/ֆ`, `Հ/Ս` (Armenian). Even top-performing models score lower on `movie_*` entries. This happens for several reasons:

- **Sparse tokens.** Abbreviations are rare in training data, so their vectors are noisy or under-trained.
- **Tokenization artifacts.** `Գ/Ֆ` gets split into subpieces (`Գ`, `/`, `Ֆ`) — the embedding reflects punctuation + letters, not "feature film".
- **No compositional meaning.** Models can't infer that `Հ/Ս` expands to `Հեռուստասերիալ` without having seen that mapping frequently.
- **Script fragmentation.** Armenian + punctuation + mixed case increases subword fragmentation in multilingual tokenizers.

Pre-processing to expand abbreviations before embedding would likely improve all scores.

### 4. Abbreviation robustness benchmark confirms e5 dominance

A dedicated abbreviation test (duplets of `"Title"` vs `"prefix Title"` using `к/ф`, `т/с`, `м/ф` for Russian and `ֆ/ֆ`, `հ/ս`, `մ/ֆ` for Armenian) measures how much an abbreviation prefix shifts the embedding. `e5-large-instruct` leads at **0.975** mean, meaning abbreviations barely move the embedding vector. The entire e5 family scores above 0.94 — well above the 0.90 threshold. Notably, `e5-large (legacy)` and `e5-large-v2` score 0.97 and 0.92 respectively despite being English-centric models, suggesting their strong subword representations handle abbreviation prefixes gracefully. At the bottom, `embed-v4.0 [cohere]` (0.70) and `distiluse` (0.75) are most disrupted by abbreviation prefixes.

### 5. LaBSE is the best non-e5 fallback

Google's Language-agnostic BERT Sentence Embedding scores 0.80 cross-lingual with solid HY performance (0.93). It's the most reliable choice if the intfloat/e5 family is unavailable.

### 6. Armenian-specialized model doesn't win overall

`Metric-AI/armenian-text-embeddings-1` (based on multilingual-e5) scores 0.91 HY↔HY — below LaBSE (0.93) and the multilingual-e5 variants (0.96–0.98). It also underperforms on EN↔RU alignment. Specialization for one language doesn't guarantee better scores even in that language.

### 7. FlagEmbedding is the fastest local inference path

`BAAI/bge-m3` via FlagEmbedding runs at 0.31s/text — 35% faster than the same model through sentence-transformers (0.47s) — with identical embedding quality. Useful for latency-sensitive production deployments.

---

## Recommendations

For a production EN+RU+HY IPTV/OTT RecSys:

| Priority | Model | Rationale |
|----------|-------|-----------|
| **Default** | `intfloat/multilingual-e5-base` | Best accuracy/size/speed trade-off |
| **Max accuracy** | `intfloat/multilingual-e5-large-instruct` | Top scores, +16% latency vs base |
| **Resource-constrained** | `BAAI/bge-m3` via FlagEmbedding | Fastest local inference, 0.31s/text |

**Avoid for HY:** `all-MiniLM-L6-v2` (English-centric), OpenAI `text-embedding-3-large` (Armenian alignment collapses to noise)

---

## Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single model
python benchmark.py --api st --model intfloat/multilingual-e5-base

# Run the full benchmark suite
./run_benchmark.sh

# With OpenAI models (paid)
OPENAI_API_KEY=sk-... ./run_benchmark.sh

# With Cohere models (paid)
COHERE_API_KEY=... python benchmark.py --api cohere --model embed-v4.0

# With Jina API (paid)
JINA_API_KEY=... python benchmark.py --api jina --model jina-embeddings-v3

# With Voyage AI (paid)
VOYAGE_API_KEY=... python benchmark.py --api voyage --model voyage-multilingual-2

# GTE (Alibaba) — local, free
python benchmark.py --api st --model Alibaba-NLP/gte-multilingual-base --trust-remote-code

# Jina v3 — local, free (same model as API, runs via SentenceTransformers)
python benchmark.py --api st --model jinaai/jina-embeddings-v3 --trust-remote-code

# With custom phrase dataset
python benchmark.py --api st --model intfloat/multilingual-e5-base --phrases data/epg_phrases.json
```

### Backends

| Flag | Package | Notes |
|------|---------|-------|
| `--api st` | `sentence-transformers` | Local inference, most models |
| `--api flag` | `FlagEmbedding` | Faster local inference for BAAI/bge-m3 |
| `--api openai` | `openai` | Requires `OPENAI_API_KEY` env var |
| `--api cohere` | `cohere` | Requires `COHERE_API_KEY` env var |
| `--api jina` | `requests` | Requires `JINA_API_KEY` env var |
| `--api voyage` | `voyageai` | Requires `VOYAGE_API_KEY` env var |
| `--api ollama` | `requests` | Requires local `ollama serve` + `TEST_EMB_OLLAMA=1` |

---

## Repository Structure

```
.
├── benchmark.py          # Main evaluation script
├── run_benchmark.sh      # Harness to run all models sequentially
├── analysis.ipynb        # Visualization notebook (5 charts)
├── DATASET.md            # Dataset preparation notes
├── requirements.txt
├── data/
│   ├── epg_phrases.json          # Test dataset: 253 EN/RU/HY triplets + 4 HY synonym pairs
│   ├── abbrev_duplets.json       # Abbreviation robustness test: plain vs prefixed title duplets
│   └── tmdb_armenian_movies.json # 523 Armenian movies from TMDB (HY/RU/EN titles)
├── scripts/
│   ├── fetch_tmdb_armenian.py    # Fetch Armenian movie titles from TMDB API
│   ├── merge_tmdb_to_phrases.py  # Merge TMDB movies into epg_phrases.json
│   └── generate_abbrev_dataset.py # Generate abbreviation robustness duplets
└── results/
    ├── benchmark_results.csv  # Pre-computed results from M2 Max
    ├── scores_ranked.png
    ├── accuracy_vs_speed.png
    ├── cross_lang_vs_hy.png
    ├── heatmap.png
    └── abbrev_scores.png
```

---

## About

Developed as part of building a content recommendation system for an IPTV/OTT platform serving multi-language EPG data in English, Russian, and Armenian. The results directly informed the production model selection.

The initial test dataset (7 triplets + 4 synonym pairs) is intentionally small — sufficient for a quick draft estimation of model quality, then extended with more titles and genres for production-grade evaluation. See [DATASET.md](DATASET.md) for dataset preparation and extension plans. No proprietary EPG data is included.

---

_Benchmarked on Apple M2 Max, 32 GB RAM · Models from [Hugging Face Hub](https://huggingface.co)_
