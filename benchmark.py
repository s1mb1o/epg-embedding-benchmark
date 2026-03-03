#!/usr/bin/env python3
"""Evaluate cosine similarities for multilingual TV EPG phrases.

Usage example:
    python benchmark.py --api st --model paraphrase-multilingual-MiniLM-L12-v2
    python benchmark.py --api flag --model BAAI/bge-m3
    python benchmark.py --api openai --model text-embedding-3-large --skip-synonyms
    python benchmark.py --api st --model jinaai/jina-embeddings-v3 --trust-remote-code
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np


# Default embedding models per backend; can be overridden with --model.
DEFAULT_MODELS: Dict[str, str] = {
    "st": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-large",
    "flag": "BAAI/bge-m3",
}


# Representative TV EPG titles in English, Russian, and Armenian.
# Non-ASCII text is intentional here because we evaluate multilingual phrases.
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
    """Normalize rows to unit length, protecting against divide-by-zero."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails."""


@dataclass
class EmbeddingClient:
    """Thin wrapper over supported embedding APIs."""

    api: str
    model_name: str
    trust_remote_code: bool = False
    _st_model: object = None
    _flag_models: Dict[str, object] = field(default_factory=dict, init=False)
    _openai_client: object = None
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _dimension: int | None = None

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding")
        text_list = list(texts)

        # Use cache to avoid duplicate embedding calls for identical strings.
        missing = [t for t in dict.fromkeys(text_list) if t not in self._cache]
        if missing:
            new_vectors = self._embed_backend(missing)
            for text, vec in zip(missing, new_vectors):
                arr = np.asarray(vec, dtype=np.float32)
                self._cache[text] = arr
                if self._dimension is None:
                    self._dimension = arr.shape[-1]

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
            self._st_model = SentenceTransformer(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
        model = self._st_model
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
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
        client = self._openai_client
        resp = client.embeddings.create(model=self.model_name, input=list(texts))
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
        stacked = np.vstack(vectors)
        return _l2_normalize(stacked)

    def _embed_ollama(self, texts: Sequence[str]) -> np.ndarray:
        try:
            import requests
        except ImportError as exc:
            raise EmbeddingError("requests package is required for --api ollama") from exc

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        url = base_url.rstrip("/") + "/api/embeddings"
        vectors: List[np.ndarray] = []

        for text in texts:
            try:
                response = requests.post(url, json={"model": self.model_name, "input": text}, timeout=60)
            except requests.RequestException as exc:  # pragma: no cover - depends on runtime connectivity
                raise EmbeddingError(f"Failed to reach Ollama at {url}: {exc}") from exc
            if response.status_code != 200:
                raise EmbeddingError(f"Ollama returned status {response.status_code}: {response.text}")

            data = response.json()
            if "embedding" not in data:
                raise EmbeddingError(f"Unexpected Ollama response: {data}")

            vectors.append(np.asarray(data["embedding"], dtype=np.float32))

        stacked = np.vstack(vectors)
        return _l2_normalize(stacked)

    def _embed_flag_embedding(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from FlagEmbedding import BGEM3FlagModel, FlagModel
        except ImportError as exc:
            raise EmbeddingError("FlagEmbedding package is required for --api flag (pip install FlagEmbedding)") from exc

        model_name = self.model_name
        is_m3 = model_name.lower().endswith("m3")

        model = self._flag_models.get(model_name)
        if model is None:
            if is_m3:
                model = BGEM3FlagModel(model_name, use_fp16=True)
            else:
                model = FlagModel(model_name, use_fp16=True)
            self._flag_models[model_name] = model

        if is_m3:
            encoded = model.encode(texts, batch_size=8)
            dense_vectors = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
            if dense_vectors is None:
                raise EmbeddingError("FlagEmbedding encode() did not return dense vectors; check model compatibility")
        else:
            dense_vectors = model.encode(texts, batch_size=8)

        embeddings = np.asarray(dense_vectors, dtype=np.float32)
        return _l2_normalize(embeddings)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
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
    parser.add_argument(
        "--api",
        choices=("st", "ollama", "openai", "flag"),
        required=True,
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--model",
        help="Embedding model identifier for the selected backend",
    )
    parser.add_argument(
        "--phrases",
        help="Optional path to JSON file with custom phrase triplets and synonym pairs",
    )
    parser.add_argument(
        "--skip-synonyms",
        action="store_true",
        help="Skip the Armenian synonym diagnostic"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading SentenceTransformer models that require trust_remote_code"
    )
    parser.add_argument(
        "--csv-file",
        help="Append a CSV result row to this file (created with header if missing)",
    )
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


def load_synonyms_from_json(path: str) -> Sequence[Dict[str, str]]:
    import json
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "hy_synonym_pairs" not in payload:
        return []
    pairs = payload["hy_synonym_pairs"]
    required_keys = {"hy_a", "hy_b"}
    validated = []
    for idx, entry in enumerate(pairs):
        if not isinstance(entry, dict) or not required_keys.issubset(entry):
            raise ValueError(f"Synonym entry at index {idx} must provide keys: {sorted(required_keys)}")
        validated.append(entry)
    return validated


def iterate_phrases(args: argparse.Namespace) -> Sequence[Dict[str, str]]:
    if args.phrases:
        return load_phrases_from_json(args.phrases)
    return PHRASE_TRIPLETS


def iterate_synonyms(args: argparse.Namespace) -> Sequence[Dict[str, str]]:
    if args.phrases:
        return load_synonyms_from_json(args.phrases)
    return HY_SYNONYM_PAIRS


def prepare_embeddings(
    client: EmbeddingClient, phrases: Sequence[Dict[str, str]]
) -> Tuple[Dict[Tuple[int, str], np.ndarray], Dict[str, float]]:
    texts: List[str] = []
    index_map: List[Tuple[int, str]] = []

    for idx, entry in enumerate(phrases):
        for lang in ("en", "ru", "hy"):
            texts.append(entry[lang])
            index_map.append((idx, lang))

    embed_start = perf_counter()
    matrix = client.embed_batch(texts)
    embed_duration = perf_counter() - embed_start
    if matrix.shape[0] != len(index_map):
        raise EmbeddingError("Embedding count mismatch")

    embedded: Dict[Tuple[int, str], np.ndarray] = {}
    for row_idx, (phrase_idx, lang) in enumerate(index_map):
        embedded[(phrase_idx, lang)] = matrix[row_idx]
    stats = {
        "count": float(len(index_map)),
        "duration": embed_duration,
    }
    return embedded, stats


def report_similarities(phrases: Sequence[Dict[str, str]], embeddings: Dict[Tuple[int, str], np.ndarray]) -> float:
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

    overall = sum(totals) / len(totals) if totals else 0.0
    if totals:
        print("-" * len(header))
        print(f"{'overall_mean':<24} {overall:>32.4f}")
    return overall


def report_hy_synonyms(client: EmbeddingClient, synonyms: Sequence[Dict[str, str]]) -> float:
    if not synonyms:
        return 0.0

    texts: List[str] = []
    pair_indices: List[int] = []
    for idx, entry in enumerate(synonyms):
        texts.append(entry["hy_a"])
        texts.append(entry["hy_b"])
        pair_indices.append(idx)

    start = perf_counter()
    matrix = client.embed_batch(texts)
    duration = perf_counter() - start

    total_texts = len(texts)
    per_text = duration / total_texts if total_texts else 0.0
    print(
        f"hy-hy api={client.api} | model={client.model_name} | dim={client._dimension or 0} | texts={total_texts} "
        f"| total_time={duration:.4f}s | per_text={per_text:.4f}s"
    )

    header = f"{'ID':<24} {'hy-hy':>8}"
    print(header)
    print("-" * len(header))

    totals = []
    for idx in pair_indices:
        vec_a = matrix[(idx * 2)]
        vec_b = matrix[(idx * 2) + 1]
        score = cosine_similarity(vec_a, vec_b)
        totals.append(score)
        ident = synonyms[idx].get("id", f"hy_pair_{idx}")
        print(f"{ident:<24} {score:8.4f}")

    mean_score = sum(totals) / len(totals) if totals else 0.0
    if totals:
        print("-" * len(header))
        print(f"{'overall_mean':<24} {mean_score:>8.4f}")
    return mean_score


def _append_csv_row(
    csv_path: str,
    backend: str,
    model: str,
    cross_lang_mean: float,
    hy_hy_mean: float,
    total_time_s: float,
    time_per_text_s: float,
) -> None:
    """Append a single result row to *csv_path*, writing the header if the file is new or empty."""
    header = ["backend", "model", "cross_lang_mean", "hy_hy_mean", "total_time_s", "time_per_text_s"]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            backend,
            model,
            f"{cross_lang_mean:.4f}",
            f"{hy_hy_mean:.4f}",
            f"{total_time_s:.4f}",
            f"{time_per_text_s:.4f}",
        ])


def main(argv: Sequence[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    api = args.api
    model_name = args.model or DEFAULT_MODELS[api]

    if api == "st" and model_name == "jinaai/jina-embeddings-v3" and not args.trust_remote_code:
        parser.error(
            "Model 'jinaai/jina-embeddings-v3' requires trust_remote_code; "
            "invoke with trust_remote_code enabled or use '--api flag --model BAAI/bge-m3'."
        )
    phrases = iterate_phrases(args)
    if not phrases:
        parser.error("No phrase triplets found; check your --phrases file or use the built-in dataset")

    client = EmbeddingClient(
        api=api,
        model_name=model_name,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        embeddings, stats = prepare_embeddings(client, phrases)
    except EmbeddingError as exc:
        parser.error(str(exc))

    total_texts = int(stats["count"])
    total_time = stats["duration"]
    per_text = total_time / total_texts if total_texts else 0.0
    dim = client._dimension or 0

    print(
        f"api={client.api} | model={client.model_name} | dim={dim} | texts={total_texts} "
        f"| total_time={total_time:.4f}s | per_text={per_text:.4f}s"
    )

    cross_lang_mean = report_similarities(phrases, embeddings)

    hy_hy_mean = 0.0
    if not args.skip_synonyms:
        hy_hy_mean = report_hy_synonyms(client, iterate_synonyms(args))

    if args.csv_file:
        _append_csv_row(args.csv_file, api, model_name, cross_lang_mean, hy_hy_mean, total_time, per_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
