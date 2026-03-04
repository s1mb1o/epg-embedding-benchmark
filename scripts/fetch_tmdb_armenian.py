#!/usr/bin/env python3
"""Fetch all Armenian-language movies from TMDB with titles in HY, RU, EN."""

import json
import os
import time
import urllib.request
import urllib.parse

API_KEY = os.environ.get("TMDB_API_KEY", "")
BASE = "https://api.themoviedb.org/3"
OUTPUT = "data/tmdb_armenian_movies.json"

DELAY = 0.05


def api_get(path, params=None):
    """Make a GET request to TMDB API."""
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_armenian_movies():
    """Get all movie IDs with original_language=hy via discover endpoint."""
    movies = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        data = api_get("/discover/movie", {
            "with_original_language": "hy",
            "sort_by": "primary_release_date.desc",
            "page": page,
        })
        total_pages = data["total_pages"]
        total_results = data["total_results"]

        for m in data["results"]:
            movies.append({
                "id": m["id"],
                "release_date": m.get("release_date", ""),
            })

        print(f"  Page {page}/{total_pages} — {len(data['results'])} movies (total: {total_results})")
        page += 1
        time.sleep(DELAY)

    return movies


def fetch_title(movie_id, language):
    """Fetch movie title in a specific language."""
    data = api_get(f"/movie/{movie_id}", {"language": language})
    return data.get("title", "")


def main():
    if not API_KEY:
        print("Error: set TMDB_API_KEY environment variable")
        return

    print("Discovering Armenian movies...")
    movies = discover_armenian_movies()
    print(f"\nFound {len(movies)} movies. Fetching titles in 3 languages...\n")

    results = []
    for i, m in enumerate(movies):
        mid = m["id"]
        year = m["release_date"][:4] if m["release_date"] else ""

        title_hy = fetch_title(mid, "hy-HY")
        time.sleep(DELAY)
        title_ru = fetch_title(mid, "ru-RU")
        time.sleep(DELAY)
        title_en = fetch_title(mid, "en-US")
        time.sleep(DELAY)

        entry = {
            "id": mid,
            "year": year,
            "hy": title_hy,
            "ru": title_ru,
            "en": title_en,
        }
        results.append(entry)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(movies)}] id={mid} year={year}")
            print(f"    HY: {title_hy}")
            print(f"    RU: {title_ru}")
            print(f"    EN: {title_en}")

    # Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} movies to {OUTPUT}")

    # Stats
    all_three = sum(1 for r in results if r["hy"] and r["ru"] and r["en"])
    has_hy = sum(1 for r in results if r["hy"])
    has_ru = sum(1 for r in results if r["ru"])
    has_en = sum(1 for r in results if r["en"])
    diff_titles = sum(1 for r in results if r["hy"] and r["ru"] and r["en"]
                      and len({r["hy"], r["ru"], r["en"]}) >= 2)
    print(f"  All 3 languages: {all_three}")
    print(f"  Has HY: {has_hy}, RU: {has_ru}, EN: {has_en}")
    print(f"  With at least 2 distinct titles: {diff_titles}")


if __name__ == "__main__":
    main()
