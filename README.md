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

Digital television EPG data predominantly uses three languages: **English** (EN), **Russian** (RU), and **Armenian** (HY). English is the lingua franca of international content metadata. Russian is the primary language of post-Soviet broadcast markets where many IPTV/OTT operators serve. Armenian is the national language for operators in Armenia — a market where all three languages coexist in the same EPG feed, making trilingual matching a hard practical requirement rather than an academic exercise.

Every IPTV operator ingests EPG data from multiple sources. The same program arrives with different titles, different transliterations, and in different languages — all needing to be matched together. A RecSys that can't do cross-lingual matching produces poor recommendations for non-English content.

Armenian (`hy`) is particularly challenging:
- Non-Latin, non-Cyrillic script (unique Armenian alphabet: Հ, Ա, Յ, Ե, ...)
- Low-resource language: underrepresented in most embedding model training data
- Domain-specific abbreviations
- Mix of armenian, russian and english in EPG titles

### Domain-specific abbreviations
  Feature Film
   * `Գ/Ֆ` (Գեղարվեստական ֆիլմ / Gegharvestakan film) - Most common and standard.
   * `ֆ/ֆ` (ֆիլմ / film) - Less formal, but sometimes used.

  TV Series
   * `Հ/Ս` (Հեռուստասերիալ / Herustaserial) - Standard.

  Animations / Cartoons
   * `Մ/Ֆ` (Մուլտիպլիկացիոն ֆիլմ or Մուլտֆիլմ / Multiplikatsion film or Multfilm) - Standard for animated movies.
   * `Մ/Ս` (Մուլտսերիալ / Multserial) - Standard for animated series.

  Documentary
   * `Փ/Ֆ` (Փաստավավերագրական ֆիլմ / Phastavaveragrakan film) - Standard.
   * `Վ/Ֆ` (Վավերագրական ֆիլմ / Vaveragrakan film) - Standard (often used interchangeably with Փ/Ֆ).

  Show/Entertainment
   * `Հ/Շ` (Հեռուստաշոու / Herustashou) - Standard for TV shows.
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

250 trilingual title triplets: 7 hand-crafted TV EPG entries covering news, sports, entertainment, and movies, plus 243 real Armenian movie titles from [TMDB](https://www.themoviedb.org/) (see [DATASET.md](DATASET.md)).

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

MacBook M2 Max, 32 GB RAM — local models run via CPU/Metal; OpenAI uses remote API.

---

## Results

| # | Backend | Model | Cross-lang | EN↔RU | EN↔HY | RU↔HY | HY↔HY | s/text |
|---|---------|-------|:----------:|:-----:|:-----:|:-----:|:-----:|:------:|
| 1 | st | `intfloat/multilingual-e5-large-instruct` | **0.900** | 0.917 | 0.872 | 0.911 | **0.975** | 0.46 |
| 2 | st | `intfloat/multilingual-e5-large` | 0.881 | 0.880 | 0.864 | 0.899 | 0.964 | 0.52 |
| 3 | st | `intfloat/multilingual-e5-base` | 0.876 | 0.883 | 0.858 | 0.889 | 0.958 | 0.41 |
| 4 | st | `sentence-transformers/LaBSE` | 0.800 | 0.786 | 0.777 | 0.837 | 0.934 | 0.47 |
| 5 | st | `Metric-AI/armenian-text-embeddings-1` | 0.798 | 0.794 | 0.777 | 0.821 | 0.910 | 0.32 |
| 6 | st | `intfloat/e5-large-v2` | 0.776 | 0.754 | 0.767 | 0.808 | 0.833 | 0.37 |
| 7 | flag | `BAAI/bge-m3` | 0.767 | 0.797 | 0.712 | 0.791 | 0.849 | **0.27** |
| 8 | st | `BAAI/bge-m3` | 0.767 | 0.797 | 0.712 | 0.791 | 0.849 | 0.45 |
| 9 | st | `intfloat/e5-large` | 0.751 | 0.723 | 0.729 | 0.803 | 0.863 | 0.39 |
| 10 | st | `paraphrase-multilingual-mpnet-base-v2` | 0.709 | 0.719 | 0.657 | 0.750 | 0.762 | 0.42 |
| 11 | st | `distiluse-base-multilingual-cased` | 0.708 | 0.766 | 0.618 | 0.739 | 0.749 | 0.37 |
| 12 | st | `paraphrase-multilingual-MiniLM-L12-v2` | 0.668 | 0.753 | 0.549 | 0.703 | 0.752 | 0.38 |
| 13 | openai | `text-embedding-3-large` | 0.342 | 0.622 | 0.168 | 0.222 | 0.665 | 0.01 |
| 14 | st | `all-MiniLM-L6-v2` | 0.141 | 0.084 | 0.105 | 0.233 | 0.460 | 0.32 |

_250 triplets · st = sentence-transformers · flag = FlagEmbedding · Hardware: M2 Max, 32 GB RAM_

Rankings are identical to the original 7-triplet benchmark — the small hand-crafted dataset was representative after all. Scores are remarkably stable, confirming the original findings hold at 250 triplets.

Raw data: [results/benchmark_results.csv](results/benchmark_results.csv)

---

## Analysis

[analysis.ipynb](analysis.ipynb) generates four charts from the benchmark results:

### Models ranked by cross-lingual score

![Models ranked by cross-lingual score](results/scores_ranked.png)

### Accuracy vs Speed

![Accuracy vs Speed](results/accuracy_vs_speed.png)

### Cross-language score vs Armenian consistency

![Cross-language vs HY consistency](results/cross_lang_vs_hy.png)

### Per language-pair heatmap

![Per language-pair heatmap](results/heatmap.png)

---

## Key Findings

### 1. The multilingual-e5 family is the clear winner for EN+RU+HY

`intfloat/multilingual-e5-large-instruct` achieves 0.90 cross-lingual mean and 0.975 HY↔HY consistency. The three multilingual-e5 variants (base, large, large-instruct) all score above 0.87 — significantly better than all other models. The base model offers the best accuracy/cost trade-off for production use.

### 2. OpenAI `text-embedding-3-large` underperforms on Armenian

Despite being a flagship commercial model, it scores **0.34 overall** — below all dedicated multilingual models tested. Armenian is a low-resource language; without sufficient HY parallel training data, the model cannot anchor Armenian phrases near their EN/RU counterparts.

**Lesson:** cost and brand do not predict performance on low-resource languages. Always benchmark for your specific language mix.

### 3. Domain-specific abbreviations hurt all models

EPG data is full of abbreviations like `к/ф` (Russian) and `Գ/Ֆ`, `ֆ/ֆ`, `Հ/Ս` (Armenian). Even top-performing models score lower on `movie_*` entries. This happens for several reasons:

- **Sparse tokens.** Abbreviations are rare in training data, so their vectors are noisy or under-trained.
- **Tokenization artifacts.** `Գ/Ֆ` gets split into subpieces (`Գ`, `/`, `Ֆ`) — the embedding reflects punctuation + letters, not "feature film".
- **No compositional meaning.** Models can't infer that `Հ/Ս` expands to `Հեռուստասերիալ` without having seen that mapping frequently.
- **Script fragmentation.** Armenian + punctuation + mixed case increases subword fragmentation in multilingual tokenizers.

Pre-processing to expand abbreviations before embedding would likely improve all scores.

### 4. LaBSE is the best non-e5 fallback

Google's Language-agnostic BERT Sentence Embedding scores 0.80 cross-lingual with solid HY performance (0.93). It's the most reliable choice if the intfloat/e5 family is unavailable.

### 5. Armenian-specialized model doesn't win overall

`Metric-AI/armenian-text-embeddings-1` (based on multilingual-e5) scores 0.91 HY↔HY — below LaBSE (0.93) and the multilingual-e5 variants (0.96–0.98). It also underperforms on EN↔RU alignment. Specialization for one language doesn't guarantee better scores even in that language.

### 6. FlagEmbedding is the fastest local inference path

`BAAI/bge-m3` via FlagEmbedding runs at 0.29s/text — 42% faster than the same model through sentence-transformers (0.50s) — with identical embedding quality. Useful for latency-sensitive production deployments.

---

## Recommendations

For a production EN+RU+HY IPTV/OTT RecSys:

| Priority | Model | Rationale |
|----------|-------|-----------|
| **Default** | `intfloat/multilingual-e5-base` | Best accuracy/size/speed trade-off |
| **Max accuracy** | `intfloat/multilingual-e5-large-instruct` | Top scores, +17% latency vs base |
| **Resource-constrained** | `BAAI/bge-m3` via FlagEmbedding | Fastest local inference, 0.29s/text |

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
├── analysis.ipynb        # Visualization notebook (4 charts)
├── DATASET.md            # Dataset preparation notes
├── requirements.txt
├── data/
│   ├── epg_phrases.json          # Test dataset: 7 EN/RU/HY triplets + 4 HY synonym pairs
│   └── tmdb_armenian_movies.json # 515 Armenian movies from TMDB (HY/RU/EN titles)
├── scripts/
│   └── fetch_tmdb_armenian.py    # Fetch Armenian movie titles from TMDB API
└── results/
    ├── benchmark_results.csv  # Pre-computed results from M2 Max
    ├── scores_ranked.png
    ├── accuracy_vs_speed.png
    ├── cross_lang_vs_hy.png
    └── heatmap.png
```

---

## About

Developed as part of building a content recommendation system for an IPTV/OTT platform serving multi-language EPG data in English, Russian, and Armenian. The results directly informed the production model selection.

The initial test dataset (7 triplets + 4 synonym pairs) is intentionally small — sufficient for a quick draft estimation of model quality, then extended with more titles and genres for production-grade evaluation. See [DATASET.md](DATASET.md) for dataset preparation and extension plans. No proprietary EPG data is included.

---

_Benchmarked on Apple M2 Max, 32 GB RAM · Models from [Hugging Face Hub](https://huggingface.co)_
