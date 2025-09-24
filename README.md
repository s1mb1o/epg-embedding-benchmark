# EPG Embedding Similarity — EN + RU

Research for a content recommendation system being developed for an IPTV operator.

The platform receives EPG data from multiple providers. The same TV program arrives
with different titles in different languages — it needs to be de-duplicated and matched
for the RecSys to work properly. This script measures how well embedding models can align
semantically identical titles across English and Russian.

## Models tested

- `all-MiniLM-L6-v2` — English-centric, baseline (poor cross-lingual)
- `paraphrase-multilingual-MiniLM-L12-v2` — lightweight multilingual
- `paraphrase-multilingual-mpnet-base-v2` — stronger multilingual baseline
- `openai/text-embedding-3-small` — cloud baseline

## Results

See [RESULTS.md](RESULTS.md).

## Run

```bash
pip install -r requirements.txt
python epg_similarity.py --api st --model paraphrase-multilingual-MiniLM-L12-v2

# OpenAI backend requires an API key
OPENAI_API_KEY=sk-... python epg_similarity.py --api openai --model text-embedding-3-small
```
