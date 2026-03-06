#!/usr/bin/env python3
"""Retrieval inspection: show best cross-language match per triplet per model.

For each embedding model and each phrase triplet, queries every language
variant (EN/RU/HY) against the full corpus and shows the nearest neighbor.
Outputs a markdown table for manual inspection.

Usage:
    # Run with specific models (api:model_name format):
    python retrieval_inspect.py --phrases data/epg_phrases.json \
        --models st:sentence-transformers/LaBSE st:BAAI/bge-m3

    # Models requiring trust_remote_code — append :trust:
    python retrieval_inspect.py --phrases data/epg_phrases.json \
        --models "st:jinaai/jina-embeddings-v3:trust"

    # Limit to first N triplets for quick testing:
    python retrieval_inspect.py --phrases data/epg_phrases.json \
        --models st:sentence-transformers/LaBSE --limit 10

    # Custom output path:
    python retrieval_inspect.py --phrases data/epg_phrases.json \
        --models st:sentence-transformers/LaBSE --output results/my_inspect.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import EmbeddingClient

LANGS = ("en", "ru", "hy")


def load_triplets(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("triplets", [])
    return data


def parse_model_spec(spec: str) -> tuple[str, str, bool]:
    """Parse 'api:model' or 'api:model:trust' -> (api, model, trust_remote_code)."""
    parts = spec.split(":", maxsplit=2)
    if len(parts) < 2:
        raise ValueError(f"Model spec must be 'api:model_name', got: {spec}")
    api = parts[0]
    model = parts[1]
    trust = len(parts) > 2 and "trust" in parts[2].lower()
    return api, model, trust


def short_name(model: str) -> str:
    return model.split("/")[-1]


def escape_md(text: str) -> str:
    """Escape characters that break markdown tables."""
    return text.replace("|", "\\|").replace("\n", " ")


def find_best_matches(
    matrix: np.ndarray,
    text_meta: list[tuple[int, str]],
    triplets: list[dict],
) -> dict:
    """For each text find the best match excluding itself.

    Returns dict of (triplet_idx, lang) -> {text, sim, correct, matched_lang}.
    """
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, -2.0)

    results = {}
    for i, (tri_idx, lang) in enumerate(text_meta):
        best_j = int(np.argmax(sim[i]))
        best_sim = float(sim[i, best_j])
        best_tri_idx, best_lang = text_meta[best_j]
        best_text = triplets[best_tri_idx][best_lang]

        results[(tri_idx, lang)] = {
            "text": best_text,
            "sim": best_sim,
            "correct": best_tri_idx == tri_idx,
            "matched_lang": best_lang,
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phrases", required=True, help="Path to epg_phrases.json")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model specs as 'api:model_name' (e.g. st:sentence-transformers/LaBSE)",
    )
    parser.add_argument(
        "--output",
        default="results/retrieval_inspection.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of triplets (0 = all)",
    )
    args = parser.parse_args()

    triplets = load_triplets(args.phrases)
    if not triplets:
        print("No triplets found", file=sys.stderr)
        return 1
    if args.limit:
        triplets = triplets[: args.limit]

    n = len(triplets)

    # Build flat text list with metadata
    all_texts: list[str] = []
    text_meta: list[tuple[int, str]] = []
    for i, t in enumerate(triplets):
        for lang in LANGS:
            all_texts.append(t[lang])
            text_meta.append((i, lang))

    print(f"Loaded {n} triplets ({len(all_texts)} texts)")

    # Run each model
    model_results: dict[str, dict] = {}
    model_order: list[str] = []

    for spec in args.models:
        api, model, trust = parse_model_spec(spec)
        sname = short_name(model)
        print(f"\n[{sname}] Embedding {len(all_texts)} texts via {api}:{model} ...")

        client = EmbeddingClient(api=api, model_name=model, trust_remote_code=trust)
        t0 = perf_counter()
        try:
            matrix = client.embed_batch(all_texts)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        dt = perf_counter() - t0

        print(f"  dim={matrix.shape[1]}  time={dt:.1f}s  computing matches ...")
        results = find_best_matches(matrix, text_meta, triplets)

        correct = sum(1 for v in results.values() if v["correct"])
        total = len(results)
        print(f"  Accuracy: {correct}/{total} ({correct / total * 100:.1f}%)")

        model_results[sname] = results
        model_order.append(sname)

    if not model_results:
        print("No models produced results.", file=sys.stderr)
        return 1

    # ---- Generate markdown ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Retrieval Inspection\n\n")
        f.write(
            "Each cell shows the nearest-neighbor match when querying "
            "with the EN / RU / HY variant of that triplet.\n\n"
        )
        f.write("- **checkmark** = matched another language of the *same* title (correct)\n")
        f.write("- **x** = matched a *different* title (wrong)\n\n")

        # Header
        cols = ["#", "Triplet (EN / RU / HY)"] + model_order
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")

        # Sort rows: best average matching first, worst last
        def triplet_sort_key(idx: int) -> tuple[float, float]:
            correct_sum = 0
            sim_sum = 0
            count = 0
            for mn in model_order:
                results = model_results[mn]
                for lang in LANGS:
                    r = results[(idx, lang)]
                    correct_sum += int(r["correct"])
                    sim_sum += r["sim"]
                    count += 1
            return (correct_sum / count, sim_sum / count)

        sorted_indices = sorted(range(n), key=triplet_sort_key, reverse=True)

        for rank, i in enumerate(sorted_indices, 1):
            t = triplets[i]
            en = escape_md(t["en"])
            ru = escape_md(t["ru"])
            hy = escape_md(t["hy"])
            triplet_cell = f"**{en}**<br>{ru}<br>{hy}"

            model_cells = []
            for mn in model_order:
                results = model_results[mn]
                lines = []
                for lang in LANGS:
                    r = results[(i, lang)]
                    marker = "✓" if r["correct"] else "✗"
                    matched = escape_md(r["text"])
                    if len(matched) > 40:
                        matched = matched[:37] + "…"
                    lines.append(f'{marker} "{matched}" {r["sim"]:.3f}')
                model_cells.append("<br>".join(lines))

            f.write(f"| {rank} | {triplet_cell} | " + " | ".join(model_cells) + " |\n")

        # Summary row
        f.write("\n## Accuracy Summary\n\n")
        f.write("| Model | Correct | Total | Accuracy |\n")
        f.write("|---|---|---|---|\n")
        for mn in model_order:
            results = model_results[mn]
            correct = sum(1 for v in results.values() if v["correct"])
            total = len(results)
            f.write(f"| {mn} | {correct} | {total} | {correct / total * 100:.1f}% |\n")

    print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
