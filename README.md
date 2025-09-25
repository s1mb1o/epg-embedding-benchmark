# EPG Embedding Benchmark — EN + RU + HY

Research for a content recommendation system being developed for an IPTV operator.
The platform serves EPG data in three languages: **English**, **Russian**, and **Armenian**.
Same TV programs arrive from multiple sources with inconsistent titles — the RecSys needs to
match and de-duplicate them cross-lingually.

This script measures cosine similarity between semantically identical titles across all three
language pairs (EN↔RU, EN↔HY, RU↔HY), and includes an Armenian synonym diagnostic.

---

## Key Finding

`openai/text-embedding-3-large` scores **0.34 overall** — dramatically lower than every
open-source multilingual model. EN↔RU alignment is decent (0.57–0.73), but Armenian vectors
drift to near-random: `en-hy` and `ru-hy` hover around **0.15–0.19**.

Armenian is a low-resource language severely underrepresented in OpenAI's training data.
Open-source multilingual models trained on broader parallel corpora handle it far better.

## Models tested

- `intfloat/multilingual-e5-base` — strong multilingual baseline, best open-source EN+RU+HY performer at this stage
- `sentence-transformers/LaBSE` — Google's language-agnostic BERT, broad multilingual coverage
- `Metric-AI/armenian-text-embeddings-1` — Armenian-tuned model, sanity-check for HY specialization
- `paraphrase-multilingual-MiniLM-L12-v2` — lightweight multilingual (carried over from day 2)
- `openai/text-embedding-3-large` — flagship commercial model, included to test Armenian coverage
- `all-MiniLM-L6-v2` — English-only reference baseline

---

## Results

See [RESULTS.md](RESULTS.md).

---

## Run

```bash
pip install -r requirements.txt

python epg_similarity.py --api st --model intfloat/multilingual-e5-base
python epg_similarity.py --api st --model sentence-transformers/LaBSE
python epg_similarity.py --api st --model Metric-AI/armenian-text-embeddings-1
python epg_similarity.py --api openai --model text-embedding-3-large
```
