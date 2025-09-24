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

---

## 2025-09-24: intfloat/multilingual-e5-base, LaBSE, Metric-AI/armenian-text-embeddings-1, openai/text-embedding-3-large

### Grok

> armenian-text-embeddings-1 (from Metric-AI on Hugging Face): Specifically optimized
> for Armenian while retaining strong multilingual capabilities from its base
> (multilingual-e5-base). Outperforms OpenAI and other general models on Armenian
> benchmarks, making it excellent for HY-EN-RU mixed content like international TV
> guides. Good for STS in media domains due to its retrieval focus.
> Pros: Best-in-class for HY; preserves EN/RU performance.
> Cons: Newer model with less widespread testing outside Armenian.

> LaBSE (Language-agnostic BERT Sentence Embedding, from Sentence Transformers):
> Supports 109 languages, including EN, RU, and HY (via mBERT heritage). MMTEB
> average ~52.1, strong in bitext mining (76.4) for cross-lingual similarity. Good
> baseline for EPG if compute is limited.

> Alternatives if needed: For English-dominant EPG, OpenAI text-embedding-3-large is
> strong but weaker on HY.

### ChatGPT

> intfloat/multilingual-e5-base (or -large): modern multilingual embedding family;
> higher recall on retrieval tasks, multilingual training includes HY. Needs vector
> normalization but works well for ranking.

> LaBSE: Google's multilingual sentence encoder with excellent cross-language
> alignment; slower but reliable zero-shot option when HY data is sparse.

> text-embedding-3-large (OpenAI): top quality if API usage is acceptable; handles
> low-resource languages better than smaller OpenAI models and supports domain
> fine-tuning via prompt engineering.
