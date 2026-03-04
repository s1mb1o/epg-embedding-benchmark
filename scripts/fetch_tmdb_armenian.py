#!/usr/bin/env python3
"""Fetch all Armenian-language movies and TV shows from TMDB with titles in HY, RU, EN."""

import json
import os
import time
import urllib.request
import urllib.parse

API_KEY = os.environ.get("TMDB_API_KEY", "")
BASE = "https://api.themoviedb.org/3"
OUTPUT = "data/tmdb_armenian_movies.json"

DELAY = 0.05
ANIMATION_GENRE_ID = 16

# Armenian Unicode block: U+0530–U+058F
_ARMENIAN_RANGE = range(0x0530, 0x0590)


def has_armenian(text):
    """Check if text contains at least one Armenian character."""
    return any(ord(ch) in _ARMENIAN_RANGE for ch in text)


def api_get(path, params=None):
    """Make a GET request to TMDB API."""
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_items(media_type):
    """Get all item IDs with original_language=hy via discover endpoint.

    media_type: 'movie' or 'tv'
    """
    items = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        data = api_get(f"/discover/{media_type}", {
            "with_original_language": "hy",
            "sort_by": "primary_release_date.desc" if media_type == "movie" else "first_air_date.desc",
            "page": page,
        })
        total_pages = data["total_pages"]
        total_results = data["total_results"]

        date_key = "release_date" if media_type == "movie" else "first_air_date"
        title_key = "original_title" if media_type == "movie" else "original_name"
        for m in data["results"]:
            items.append({
                "id": m["id"],
                "date": m.get(date_key, ""),
                "genre_ids": m.get("genre_ids", []),
                "original_title": m.get(title_key, ""),
            })

        print(f"  Page {page}/{total_pages} — {len(data['results'])} items (total: {total_results})")
        page += 1
        time.sleep(DELAY)

    return items


def fetch_details(media_type, item_id, language):
    """Fetch item details in a specific language.

    Returns (title, genres) where genres is a list of genre name strings.
    For movies: title field is 'title'. For TV: title field is 'name'.
    """
    data = api_get(f"/{media_type}/{item_id}", {"language": language})
    title_key = "title" if media_type == "movie" else "name"
    title = data.get(title_key, "")
    genres = [g["name"] for g in data.get("genres", [])]
    return title, genres


def classify_type(media_type, genre_ids):
    """Classify content type based on media type and animation genre."""
    is_animation = ANIMATION_GENRE_ID in genre_ids
    if media_type == "movie":
        return "animation_movie" if is_animation else "movie"
    else:
        return "animation_series" if is_animation else "series"


def main():
    if not API_KEY:
        print("Error: set TMDB_API_KEY environment variable")
        return

    all_results = []

    for media_type in ("movie", "tv"):
        label = "movies" if media_type == "movie" else "TV shows"
        print(f"Discovering Armenian {label}...")
        items = discover_items(media_type)
        print(f"\nFound {len(items)} {label}. Fetching titles in 3 languages...\n")

        for i, item in enumerate(items):
            item_id = item["id"]
            year = item["date"][:4] if item["date"] else ""
            content_type = classify_type(media_type, item["genre_ids"])

            original_title = item["original_title"]

            # Fetch HY details (includes genres in Armenian)
            title_hy, _ = fetch_details(media_type, item_id, "hy-HY")
            time.sleep(DELAY)
            title_ru, _ = fetch_details(media_type, item_id, "ru-RU")
            time.sleep(DELAY)
            # Fetch EN details — use EN genre names for readability
            title_en, genres_en = fetch_details(media_type, item_id, "en-US")
            time.sleep(DELAY)

            # TMDB returns the original title as fallback when no translation
            # exists. Detect this and set untranslated titles to empty string.
            if not has_armenian(title_hy):
                title_hy = ""
            if title_ru == original_title:
                title_ru = ""
            if title_en == original_title:
                title_en = ""

            # Skip entries without Armenian title — they have no usable HY data
            if not title_hy:
                continue

            entry = {
                "id": item_id,
                "type": content_type,
                "year": year,
                "genres": genres_en,
                "hy": title_hy,
                "ru": title_ru,
                "en": title_en,
            }
            all_results.append(entry)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(items)}] {media_type} id={item_id} year={year} type={content_type}")
                print(f"    Genres: {', '.join(genres_en) if genres_en else '—'}")
                print(f"    HY: {title_hy}")
                print(f"    RU: {title_ru}")
                print(f"    EN: {title_en}")

    # Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_results)} items to {OUTPUT}")

    # Stats
    movies = [r for r in all_results if r["type"] in ("movie", "animation_movie")]
    tv = [r for r in all_results if r["type"] in ("series", "animation_series")]
    anim = [r for r in all_results if r["type"].startswith("animation")]
    all_three = sum(1 for r in all_results if r["hy"] and r["ru"] and r["en"])
    has_hy = sum(1 for r in all_results if r["hy"])
    has_ru = sum(1 for r in all_results if r["ru"])
    has_en = sum(1 for r in all_results if r["en"])

    print(f"\n  Movies: {len(movies)}, TV shows: {len(tv)}, Animation: {len(anim)}")
    print(f"  All 3 languages: {all_three}")
    print(f"  Has HY: {has_hy}, RU: {has_ru}, EN: {has_en}")


if __name__ == "__main__":
    main()
