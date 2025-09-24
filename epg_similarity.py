#!/usr/bin/env python3
"""Cosine similarity for multilingual EPG title pairs (EN + RU).

Research for a content recommendation system being developed for an IPTV
operator. The platform ingests EPG data in English and Russian from multiple
providers; the same program often arrives with different titles. This script
checks how well embedding models can align semantically identical titles
across languages.

Usage:
    python epg_similarity.py --api st --model paraphrase-multilingual-MiniLM-L12-v2
    python epg_similarity.py --api openai --model text-embedding-3-small
"""

import argparse
import os

import numpy as np


# EN+RU phrase pairs: same TV program, two languages
PHRASE_PAIRS = [
    {"en": "Evening News",           "ru": "Вечерние новости"},
    {"en": "Morning Show",           "ru": "Утреннее шоу"},
    {"en": "Live Football",          "ru": "Прямая трансляция футбола"},
    {"en": "Documentary Premiere",   "ru": "Премьера документального фильма"},
    {"en": "Cooking Competition",    "ru": "Кулинарное состязание"},
]

DEFAULT_MODELS = {
    "st": "paraphrase-multilingual-MiniLM-L12-v2",
    "openai": "text-embedding-3-small",
}


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_st(texts, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def embed_openai(texts, model_name):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=model_name, input=list(texts))
    arr = np.array([item.embedding for item in resp.data], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def main():
    parser = argparse.ArgumentParser(description="EPG title similarity — EN+RU")
    parser.add_argument("--api", choices=("st", "openai"), default="st")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model_name = args.model or DEFAULT_MODELS[args.api]
    print(f"api={args.api}  model={model_name}\n")

    en_texts = [p["en"] for p in PHRASE_PAIRS]
    ru_texts = [p["ru"] for p in PHRASE_PAIRS]

    if args.api == "st":
        vecs = embed_st(en_texts + ru_texts, model_name)
    else:
        vecs = embed_openai(en_texts + ru_texts, model_name)

    n = len(PHRASE_PAIRS)
    print(f"{'EN':<35} {'RU':<40} {'sim':>6}")
    print("-" * 85)
    scores = []
    for i, pair in enumerate(PHRASE_PAIRS):
        score = cosine_similarity(vecs[i], vecs[n + i])
        scores.append(score)
        print(f"{pair['en']:<35} {pair['ru']:<40} {score:6.4f}")
    print("-" * 85)
    print(f"{'mean':<76} {sum(scores)/len(scores):6.4f}")


if __name__ == "__main__":
    main()
