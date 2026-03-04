#!/usr/bin/env python3
"""Merge TMDB Armenian movies into epg_phrases.json as additional triplets."""

import json

TMDB_FILE = "data/tmdb_armenian_movies.json"
PHRASES_FILE = "data/epg_phrases.json"


def main():
    with open(TMDB_FILE, encoding="utf-8") as f:
        movies = json.load(f)

    with open(PHRASES_FILE, encoding="utf-8") as f:
        phrases = json.load(f)

    existing_ids = {t["id"] for t in phrases["triplets"]}

    added = 0
    for m in movies:
        if not m["hy"] or not m["ru"] or not m["en"]:
            continue

        tid = f"tmdb_{m['id']}"
        if tid in existing_ids:
            continue

        phrases["triplets"].append({
            "id": tid,
            "en": m["en"],
            "ru": m["ru"],
            "hy": m["hy"],
        })
        existing_ids.add(tid)
        added += 1

    with open(PHRASES_FILE, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)
        f.write("\n")

    total = len(phrases["triplets"])
    print(f"Added {added} TMDB triplets. Total triplets: {total}")


if __name__ == "__main__":
    main()
