# epg-similarity

Research for a RecSys being developed for an IPTV operator. Testing two models on English title pairs:

- `all-MiniLM-L6-v2` — the default sentence-transformers model, fast and lightweight, good starting point to verify the cosine similarity logic works at all
- `openai/text-embedding-3-small` — cheapest OpenAI embedding model, easy cloud baseline to compare against local inference

## Results

See [RESULTS.md](RESULTS.md).

## Run

```bash
pip install sentence-transformers openai numpy
OPENAI_API_KEY=sk-... python epg_similarity.py
```
