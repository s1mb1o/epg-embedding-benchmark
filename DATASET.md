# Dataset Preparation

## Problem

The initial test dataset (7 triplets + 4 synonym pairs) was intentionally small — sufficient for a quick draft estimation of model quality, but needed to be extended with more titles and genres for production-grade evaluation.

A robust benchmark needs hundreds of real multilingual title pairs, not hand-crafted examples. Manually creating large datasets is tedious and introduces author bias.

## Solution: TMDB Armenian Movies

[The Movie Database (TMDB)](https://www.themoviedb.org/) has a public API with multilingual metadata. Armenian-language content (`original_language=hy`) provides a natural source of real-world title triplets in HY, RU, and EN.

**Script:** [scripts/fetch_tmdb_armenian.py](scripts/fetch_tmdb_armenian.py)

The script:
1. Discovers movies, animated films, and TV series with `original_language=hy` via the TMDB `/discover/movie` and `/discover/tv` endpoints
2. For each entry, fetches the title in three languages: `hy-HY`, `ru-RU`, `en-US`
3. Tags each entry with `type` (`movie`, `animation_movie`, `series`) and `genres`
4. Saves the result to `data/tmdb_armenian_movies.json`

Usage:
```bash
TMDB_API_KEY=your_key python scripts/fetch_tmdb_armenian.py
```

## Results

Fetched **523** Armenian-language entries from TMDB (447 movies, 68 animated films, 8 TV series).

| Metric | Count |
|--------|------:|
| Total movies | 523 |
| Have HY title | 523 |
| Have EN title | 500 |
| Have RU title | 260 |
| Have all 3 languages | 246 |
| With at least 2 distinct titles | 246 |

Only **246 out of 523** movies (47%) have titles in all three languages — the rest are missing Russian translations. This is expected: TMDB relies on community contributions, and Russian metadata for Armenian films is sparse.

After deduplication (removing 8 entries that appeared under different TMDB IDs with identical or overlapping titles in any language), **238 TMDB triplets** are integrated into the benchmark alongside the 7 hand-crafted triplets (245 total). The merge script (`scripts/merge_tmdb_to_phrases.py`) detects duplicates by checking if any single non-empty language title already exists under a different ID.

## Data Format

Each entry in `data/tmdb_armenian_movies.json`:

```json
{
  "id": 1588816,
  "type": "movie",
  "year": "2025",
  "genres": ["Drama"],
  "hy": "Հիմա մեր հերթն է",
  "ru": "Теперь наша очередь.",
  "en": "Now it's our turn"
}
```

## Abbreviation Robustness Dataset

**File:** `data/abbrev_duplets.json`
**Script:** [scripts/generate_abbrev_dataset.py](scripts/generate_abbrev_dataset.py)

EPG titles in Russian and Armenian commonly use content-type abbreviation prefixes (`к/ф`, `ֆ/ֆ`, etc.) that may shift the embedding away from the plain title. This dataset measures how much.

Each entry is a **duplet**: a plain title and the same title with an abbreviation prefix prepended.

| Abbreviation | Type | Russian | Armenian |
|:------------:|------|---------|----------|
| movie | Feature Film | `к/ф` | `ֆ/ֆ` |
| animation_movie | Animated Film | `м/ф` | `մ/ֆ` |
| series | TV Series | `т/с` | `հ/ս` |

### Stats

| Metric | Count |
|--------|------:|
| Total duplets | 783 |
| HY duplets | 523 |
| RU duplets | 260 |
| Movie type | 649 |
| Animation type | 123 |
| Series type | 11 |

### Data Format

```json
{
  "id": "tmdb_1601611_hy",
  "type": "movie",
  "lang": "hy",
  "plain": "Ընտանեկան Հարաման (Undanegan Haraman)",
  "abbreviated": "ֆ/ֆ Ընտանեկան Հարաման (Undanegan Haraman)"
}
```

The benchmark computes cosine similarity between embeddings of `plain` and `abbreviated` for each duplet. A perfect model (1.0) would embed both identically — the abbreviation prefix adds no semantic shift. Results are reported as `abbrev_ru_mean`, `abbrev_hy_mean`, and `abbrev_mean` in `benchmark_results.csv`.

## Next Steps

- Add more content types (TV series, documentaries, sports) beyond movies
- Source titles from additional databases beyond TMDB
