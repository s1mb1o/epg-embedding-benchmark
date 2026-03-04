# Dataset Preparation

## Problem

The test dataset (7 triplets + 4 synonym pairs) is intentionally small — sufficient for a quick draft estimation of model quality, but should be extended with more titles and genres for production-grade evaluation.

A robust benchmark needs hundreds of real multilingual title pairs, not hand-crafted examples. Manually creating large datasets is tedious and introduces author bias.

## Solution: TMDB Armenian Movies

[The Movie Database (TMDB)](https://www.themoviedb.org/) has a public API with multilingual metadata. Armenian-language movies (`original_language=hy`) provide a natural source of real-world title triplets in HY, RU, and EN.

**Script:** [scripts/fetch_tmdb_armenian.py](scripts/fetch_tmdb_armenian.py)

The script:
1. Discovers all movies with `original_language=hy` via the TMDB `/discover/movie` endpoint
2. For each movie, fetches the title in three languages: `hy-HY`, `ru-RU`, `en-US`
3. Saves the result to `data/tmdb_armenian_movies.json`

Usage:
```bash
TMDB_API_KEY=your_key python scripts/fetch_tmdb_armenian.py
```

## Results

Fetched **515** Armenian-language movies from TMDB.

| Metric | Count |
|--------|------:|
| Total movies | 515 |
| Have HY title | 515 |
| Have EN title | 494 |
| Have RU title | 256 |
| Have all 3 languages | 243 |
| With at least 2 distinct titles | 243 |

Only **243 out of 515** movies (47%) have titles in all three languages — the rest are missing Russian translations. This is expected: TMDB relies on community contributions, and Russian metadata for Armenian films is sparse.

The 243 complete triplets provide a solid foundation for extending the benchmark with real movie titles.

## Data Format

Each entry in `data/tmdb_armenian_movies.json`:

```json
{
  "id": 1588816,
  "year": "2025",
  "hy": "Հիմա մեր հերթն է",
  "ru": "Теперь наша очередь.",
  "en": "Now it's our turn"
}
```

## Next Steps

- Filter to movies with all 3 titles and at least 2 distinct titles
- Integrate as an extended test set in the benchmark pipeline
