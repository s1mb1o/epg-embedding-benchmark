#!/usr/bin/env python3
"""Evaluate cosine similarities for multilingual TV EPG titles (EN + RU + HY).

Research for a content recommendation system being developed for an IPTV
operator. The platform serves EPG data in English, Russian, and Armenian.
This script benchmarks how well embedding models align semantically identical
titles across all three languages.

Usage:
    python epg_similarity.py --api st --model intfloat/multilingual-e5-base
    python epg_similarity.py --api openai --model text-embedding-3-large
"""

import argparse
import os
from time import perf_counter

import numpy as np


PHRASE_TRIPLETS = [
    {
        "id": "evening_news",
        "en": "Evening News",
        "ru": "Вечерние новости",
        "hy": "Երեկոյան լուրեր",
    },
    {
        "id": "morning_show",
        "en": "Morning Show",
        "ru": "Утреннее шоу",
        "hy": "Առավոտյան շոու",
    },
    {
        "id": "documentary_premiere",
        "en": "Documentary Premiere",
        "ru": "Премьера документального фильма",
        "hy": "Վավերագրական ֆիլմի պրեմիերա",
    },
    {
        "id": "live_football",
        "en": "Live Football",
        "ru": "Прямая трансляция футбола",
        "hy": "Ֆուտբոլի ուղիղ հեռարձակում",
    },
    {
        "id": "cooking_competition",
        "en": "Cooking Competition",
        "ru": "Кулинарное состязание",
        "hy": "Խոհարարական մրցույթ",
    },
    {
        "id": "movie_road_home",
        "en": "Feature Film: The Road Home",
        "ru": "к/ф Дорога домой",
        "hy": "ֆ/ֆ Տուն վերադարձ",
    },
    {
        "id": "movie_secret_ararat",
        "en": "Feature Film: Secret of Ararat",
        "ru": "к/ф Тайна Арарата",
        "hy": "ֆ/ֆ Արարատի գաղտնիքը",
    },
]


HY_SYNONYM_PAIRS = [
    {
        "id": "hy_news_bulletin",
        "hy_a": "Լուրերի թողարկում",
        "hy_b": "Նորությունների թողարկում",
    },
    {
        "id": "hy_live_broadcast",
        "hy_a": "Ուղիղ հեռարձակում",
        "hy_b": "Ուղիղ եթեր",
    },
    {
        "id": "hy_sports_wrap",
        "hy_a": "Մարզական ամփոփում",
        "hy_b": "Սպորտային ամփոփագիր",
    },
    {
        "id": "hy_family_movie",
        "hy_a": "Ընտանեկան ֆիլմ",
        "hy_b": "Ընտանիքի համար նախատեսված ֆիլմ",
    },
]


DEFAULT_MODELS = {
    "st": "intfloat/multilingual-e5-base",
    "openai": "text-embedding-3-large",
}


def l2_normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_st(texts, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def embed_openai(texts, model_name):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model_name, input=list(texts))
    arr = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return l2_normalize(arr)


def get_embeddings(texts, api, model_name):
    start = perf_counter()
    if api == "st":
        vecs = embed_st(texts, model_name)
    else:
        vecs = embed_openai(texts, model_name)
    duration = perf_counter() - start
    return vecs, duration


def report_similarities(triplets, embeddings):
    header = f"{'ID':<24} {'en-ru':>8} {'en-hy':>8} {'ru-hy':>8} {'mean':>8}"
    print(header)
    print("-" * len(header))

    totals = []
    for idx, entry in enumerate(triplets):
        vec_en = embeddings[idx * 3]
        vec_ru = embeddings[idx * 3 + 1]
        vec_hy = embeddings[idx * 3 + 2]

        sim_en_ru = cosine_similarity(vec_en, vec_ru)
        sim_en_hy = cosine_similarity(vec_en, vec_hy)
        sim_ru_hy = cosine_similarity(vec_ru, vec_hy)
        mean_val = (sim_en_ru + sim_en_hy + sim_ru_hy) / 3.0
        totals.append(mean_val)

        print(
            f"{entry.get('id', f'phrase_{idx}'):<24} "
            f"{sim_en_ru:8.4f} {sim_en_hy:8.4f} {sim_ru_hy:8.4f} {mean_val:8.4f}"
        )

    if totals:
        overall = sum(totals) / len(totals)
        print("-" * len(header))
        print(f"{'overall_mean':<24} {overall:>32.4f}")


def report_hy_synonyms(synonyms, api, model_name):
    if not synonyms:
        return

    texts = []
    for entry in synonyms:
        texts.append(entry["hy_a"])
        texts.append(entry["hy_b"])

    vecs, duration = get_embeddings(texts, api, model_name)
    per_text = duration / len(texts) if texts else 0.0
    print(f"\nhy-hy | total_time={duration:.4f}s | per_text={per_text:.4f}s")

    header = f"{'ID':<24} {'hy-hy':>8}"
    print(header)
    print("-" * len(header))

    totals = []
    for idx, entry in enumerate(synonyms):
        vec_a = vecs[idx * 2]
        vec_b = vecs[idx * 2 + 1]
        score = cosine_similarity(vec_a, vec_b)
        totals.append(score)
        print(f"{entry.get('id', f'hy_pair_{idx}'):<24} {score:8.4f}")

    if totals:
        mean_score = sum(totals) / len(totals)
        print("-" * len(header))
        print(f"{'overall_mean':<24} {mean_score:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="EPG title similarity — EN+RU+HY")
    parser.add_argument("--api", choices=("st", "openai"), required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model_name = args.model or DEFAULT_MODELS[args.api]

    texts = []
    for entry in PHRASE_TRIPLETS:
        texts.extend([entry["en"], entry["ru"], entry["hy"]])

    vecs, duration = get_embeddings(texts, args.api, model_name)
    total = len(texts)
    per_text = duration / total if total else 0.0

    print(f"api={args.api} | model={model_name} | texts={total} | total_time={duration:.4f}s | per_text={per_text:.4f}s")
    report_similarities(PHRASE_TRIPLETS, vecs)
    report_hy_synonyms(HY_SYNONYM_PAIRS, args.api, model_name)


if __name__ == "__main__":
    main()
