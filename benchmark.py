#!/usr/bin/env python3
"""Evaluate cosine similarities for multilingual TV EPG phrases.

Usage example:
    python benchmark.py --api st --model intfloat/multilingual-e5-base
    python benchmark.py --api flag --model BAAI/bge-m3
    python benchmark.py --api openai --model text-embedding-3-large --skip-synonyms
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_MODELS: Dict[str, str] = {
    "st": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-large",
    "flag": "BAAI/bge-m3",
}


PHRASE_TRIPLETS: Sequence[Dict[str, str]] = (
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
)


HY_SYNONYM_PAIRS: Sequence[Dict[str, str]] = (
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
)

def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingClient:
    """Thin wrapper over supported embedding APIs."""

    api: str
    model_name: str
    trust_remote_code: bool = False
    _st_model: object = None
    _flag_model: object = None
    _openai_client: object = None
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding")
        text_list = list(texts)
        missing = [t for t in dict.fromkeys(text_list) if t not in self._cache]
        if missing:
            new_vectors = self._embed_backend(missing)
            for text, vec in zip(missing, new_vectors):
                self._cache[text] = np.asarray(vec, dtype=np.float32)
        stacked = [self._cache[t] for t in text_list]
        return np.vstack(stacked)

    def _embed_backend(self, texts: Sequence[str]) -> np.ndarray:
        if self.api == "st":
            return self._embed_sentence_transformer(texts)
        if self.api == "openai":
            return self._embed_openai(texts)
        if self.api == "ollama":
            return self._embed_ollama(texts)
        if self.api == "flag":
            return self._embed_flag_embedding(texts)
        raise EmbeddingError(f"Unsupported API '{self.api}'")

    def _embed_sentence_transformer(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("sentence-transformers package is required for --api st") from exc
        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model_name, trust_remote_code=self.trust_remote_code)
        embeddings = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_openai(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError("openai package is required for --api openai") from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY environment variable is not set")
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=api_key)
        resp = self._openai_client.embeddings.create(model=self.model_name, input=list(texts))
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
        return _l2_normalize(np.vstack(vectors))

    def _embed_ollama(self, texts: Sequence[str]) -> np.ndarray:
        try:
            import requests
        except ImportError as exc:
            raise EmbeddingError("requests package is required for --api ollama") from exc
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        url = base_url.rstrip("/") + "/api/embeddings"
        vectors: List[np.ndarray] = []
        for text in texts:
            response = requests.post(url, json={"model": self.model_name, "input": text}, timeout=60)
            if response.status_code != 200:
                raise EmbeddingError(f"Ollama returned status {response.status_code}: {response.text}")
            data = response.json()
            if "embedding" not in data:
                raise EmbeddingError(f"Unexpected Ollama response: {data}")
            vectors.append(np.asarray(data["embedding"], dtype=np.float32))
        return _l2_normalize(np.vstack(vectors))

    def _embed_flag_embedding(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise EmbeddingError("FlagEmbedding package is required for --api flag") from exc
        if self._flag_model is None:
            self._flag_model = BGEM3FlagModel(self.model_name, use_fp16=True)
        encoded = self._flag_model.encode(texts, batch_size=8)
        dense_vectors = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
        if dense_vectors is None:
            raise EmbeddingError("FlagEmbedding encode() did not return dense vectors")
        return _l2_normalize(np.asarray(dense_vectors, dtype=np.float32))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.shape != vec_b.shape:
        raise ValueError("Vectors must share the same dimensionality")
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate cosine similarity of multilingual TV EPG titles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api", choices=("st", "ollama", "openai", "flag"), required=True)
    parser.add_argument("--model", help="Embedding model identifier for the selected backend")
    parser.add_argument("--phrases", help="Optional path to JSON file with custom phrase triplets")
    parser.add_argument("--skip-synonyms", action="store_true", help="Skip the Armenian synonym diagnostic")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow loading models that require trust_remote_code")
    return parser


def load_phrases_from_json(path: str) -> Sequence[Dict[str, str]]:
    import json
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("triplets", payload.get("phrases", []))
    if not isinstance(payload, list):
        raise ValueError("JSON file must contain a list of phrase objects or an object with a 'triplets' key")
    required_keys = {"en", "ru", "hy"}
    validated = []
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict) or not required_keys.issubset(entry):
            raise ValueError(f"Phrase entry at index {idx} must provide keys: {sorted(required_keys)}")
        validated.append(entry)
    return validated


def iterate_phrases(args: argparse.Namespace) -> Sequence[Dict[str, str]]:
    if args.phrases:
        return load_phrases_from_json(args.phrases)
    return PHRASE_TRIPLETS


def prepare_embeddings(client: EmbeddingClient, phrases: Sequence[Dict[str, str]]) -> Tuple[Dict[Tuple[int, str], np.ndarray], Dict[str, float]]:
    texts: List[str] = []
    index_map: List[Tuple[int, str]] = []
    for idx, entry in enumerate(phrases):
        for lang in ("en", "ru", "hy"):
            texts.append(entry[lang])
            index_map.append((idx, lang))
    embed_start = perf_counter()
    matrix = client.embed_batch(texts)
    embed_duration = perf_counter() - embed_start
    embedded: Dict[Tuple[int, str], np.ndarray] = {}
    for row_idx, (phrase_idx, lang) in enumerate(index_map):
        embedded[(phrase_idx, lang)] = matrix[row_idx]
    stats = {"count": float(len(index_map)), "duration": embed_duration}
    return embedded, stats


def report_similarities(phrases: Sequence[Dict[str, str]], embeddings: Dict[Tuple[int, str], np.ndarray]) -> None:
    header = f"{'ID':<24} {'en-ru':>8} {'en-hy':>8} {'ru-hy':>8} {'mean':>8}"
    print(header)
    print("-" * len(header))
    totals = []
    for idx, entry in enumerate(phrases):
        vec_en = embeddings[(idx, "en")]
        vec_ru = embeddings[(idx, "ru")]
        vec_hy = embeddings[(idx, "hy")]
        sim_en_ru = cosine_similarity(vec_en, vec_ru)
        sim_en_hy = cosine_similarity(vec_en, vec_hy)
        sim_ru_hy = cosine_similarity(vec_ru, vec_hy)
        mean_value = (sim_en_ru + sim_en_hy + sim_ru_hy) / 3.0
        totals.append(mean_value)
        print(
            f"{entry.get('id', f'phrase_{idx}'):<24} "
            f"{sim_en_ru:8.4f} {sim_en_hy:8.4f} {sim_ru_hy:8.4f} {mean_value:8.4f}"
        )
    if totals:
        overall = sum(totals) / len(totals)
        print("-" * len(header))
        print(f"{'overall_mean':<24} {overall:>32.4f}")


def report_hy_synonyms(client: EmbeddingClient, synonyms: Sequence[Dict[str, str]]) -> None:
    if not synonyms:
        return
    texts: List[str] = []
    for entry in synonyms:
        texts.append(entry["hy_a"])
        texts.append(entry["hy_b"])
    start = perf_counter()
    matrix = client.embed_batch(texts)
    duration = perf_counter() - start
    total_texts = len(texts)
    per_text = duration / total_texts if total_texts else 0.0
    print(f"hy-hy api={client.api} | model={client.model_name} | texts={total_texts} | total_time={duration:.4f}s | per_text={per_text:.4f}s")
    header = f"{'ID':<24} {'hy-hy':>8}"
    print(header)
    print("-" * len(header))
    totals = []
    for idx, entry in enumerate(synonyms):
        vec_a = matrix[idx * 2]
        vec_b = matrix[idx * 2 + 1]
        score = cosine_similarity(vec_a, vec_b)
        totals.append(score)
        ident = entry.get("id", f"hy_pair_{idx}")
        print(f"{ident:<24} {score:8.4f}")
    if totals:
        mean_score = sum(totals) / len(totals)
        print("-" * len(header))
        print(f"{'overall_mean':<24} {mean_score:>8.4f}")


def main(argv: Sequence[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    api = args.api
    model_name = args.model or DEFAULT_MODELS[api]
    phrases = iterate_phrases(args)
    client = EmbeddingClient(api=api, model_name=model_name, trust_remote_code=args.trust_remote_code)
    try:
        embeddings, stats = prepare_embeddings(client, phrases)
    except EmbeddingError as exc:
        parser.error(str(exc))
    total_texts = int(stats["count"])
    total_time = stats["duration"]
    per_text = total_time / total_texts if total_texts else 0.0
    print(f"api={client.api} | model={client.model_name} | texts={total_texts} | total_time={total_time:.4f}s | per_text={per_text:.4f}s")
    report_similarities(phrases, embeddings)
    if not args.skip_synonyms:
        report_hy_synonyms(client, HY_SYNONYM_PAIRS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
