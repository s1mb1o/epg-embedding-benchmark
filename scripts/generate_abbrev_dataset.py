#!/usr/bin/env python3
"""Generate abbreviation duplets from TMDB Armenian movies dataset.

For each movie with a non-empty title in a given language, creates a duplet:
  (plain_title, abbreviated_title)

This tests whether embedding models understand that
"к/ф Domestic Demon" ≈ "Domestic Demon".
"""

from __future__ import annotations

import json
from pathlib import Path

# Standard EPG content-type abbreviations by language.
ABBREVIATIONS = {
    "movie": {"ru": "к/ф", "hy": "ֆ/ֆ"},
    "animation_movie": {"ru": "м/ф", "hy": "մ/ֆ"},
    "series": {"ru": "т/с", "hy": "հ/ս"},
}


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    src = data_dir / "tmdb_armenian_movies.json"
    dst = data_dir / "abbrev_duplets.json"

    with open(src, "r", encoding="utf-8") as f:
        movies = json.load(f)

    duplets = []
    for movie in movies:
        movie_type = movie.get("type", "movie")
        abbrevs = ABBREVIATIONS.get(movie_type, ABBREVIATIONS["movie"])
        movie_id = movie.get("id", "unknown")

        for lang in ("ru", "hy"):
            title = movie.get(lang, "").strip()
            if not title:
                continue

            prefix = abbrevs[lang]
            duplets.append({
                "id": f"tmdb_{movie_id}_{lang}",
                "type": movie_type,
                "lang": lang,
                "plain": title,
                "abbreviated": f"{prefix} {title}",
            })

    output = {
        "description": "EPG abbreviation robustness test — duplets of titles with and without content-type prefixes",
        "abbreviations": ABBREVIATIONS,
        "duplets": duplets,
    }

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    by_type: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for d in duplets:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
        by_lang[d["lang"]] = by_lang.get(d["lang"], 0) + 1

    print(f"Generated {len(duplets)} abbreviation duplets → {dst}")
    print(f"  By type: {by_type}")
    print(f"  By lang: {by_lang}")


if __name__ == "__main__":
    main()
