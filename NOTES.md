# Notes — Model Selection

Consulted Grok and ChatGPT on best embedding models for EN/RU/HY EPG domain.
Quotes below record which suggestions I acted on and why.

---

## 2025-09-24: paraphrase-multilingual-mpnet-base-v2, paraphrase-multilingual-MiniLM-L12-v2

### ChatGPT

> sentence-transformers/paraphrase-multilingual-mpnet-base-v2: strong multilingual ST
> baseline; does well on EN/RU and covers Armenian. Fast to run, easy to fine-tune on
> EPG pairs.

### Grok

> paraphrase-multilingual-mpnet-base-v2 (from Sentence Transformers): Trained on 50+
> languages with strong STS (79.8 in MMTEB), including EN and RU. Armenian support is
> indirect (via multilingual training), but it may underperform on HY compared to E5 or
> specialized models.
