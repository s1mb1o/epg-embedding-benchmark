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

    # Build text indexes for dedup: skip movies whose titles already exist
    # under a different TMDB ID (same movie, different DB entry).
    existing_en = {t["en"]: t["id"] for t in phrases["triplets"]}
    existing_ru = {t["ru"]: t["id"] for t in phrases["triplets"]}
    existing_hy = {t["hy"]: t["id"] for t in phrases["triplets"]}

    added = 0
    skipped_dup = 0
    for m in movies:
        if not m["hy"] or not m["ru"] or not m["en"]:
            continue

        tid = f"tmdb_{m['id']}"
        if tid in existing_ids:
            continue

        # Skip if ANY non-empty title already exists under a different ID.
        # Identical texts produce identical embeddings, causing false
        # negatives in retrieval benchmarks.
        dup_en = existing_en.get(m["en"])
        dup_ru = existing_ru.get(m["ru"])
        dup_hy = existing_hy.get(m["hy"])

        if dup_en:
            print(f"  skip {tid}: EN dup of {dup_en} (\"{m['en']}\")")
            skipped_dup += 1
            continue
        if dup_ru:
            print(f"  skip {tid}: RU dup of {dup_ru} (\"{m['ru']}\")")
            skipped_dup += 1
            continue
        if dup_hy:
            print(f"  skip {tid}: HY dup of {dup_hy} (\"{m['hy']}\")")
            skipped_dup += 1
            continue

        triplet = {"id": tid, "en": m["en"], "ru": m["ru"], "hy": m["hy"]}
        phrases["triplets"].append(triplet)
        existing_ids.add(tid)
        existing_en[m["en"]] = tid
        existing_ru[m["ru"]] = tid
        existing_hy[m["hy"]] = tid
        added += 1

    with open(PHRASES_FILE, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)
        f.write("\n")

    total = len(phrases["triplets"])
    print(f"Added {added} TMDB triplets, skipped {skipped_dup} duplicates. Total triplets: {total}")


if __name__ == "__main__":
    main()
