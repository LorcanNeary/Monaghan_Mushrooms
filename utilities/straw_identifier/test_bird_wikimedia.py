import requests
from fastdownload import download_url
from PIL import Image

def search_wikimedia(term, max_images=5):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": term,
        "gsrlimit": max_images,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    headers = {
        # Use a descriptive UA per Wikimedia's API etiquette
        "User-Agent": "MonaghanMushroomsImageFetcher/0.1 (contact: your-email@example.com)"
    }

    r = requests.get(url, params=params, headers=headers, timeout=15)

    # Debug: if request fails or returns non-JSON, show it
    try:
        r.raise_for_status()
    except Exception as e:
        print(f"[search_wikimedia] HTTP error: {e}")
        print("[search_wikimedia] Response text (truncated):")
        print(r.text[:500])
        return []

    try:
        data = r.json()
    except Exception as e:
        print(f"[search_wikimedia] JSON decode error: {e}")
        print("[search_wikimedia] Raw response (truncated):")
        print(r.text[:500])
        return []

    if "query" not in data or "pages" not in data["query"]:
        print("[search_wikimedia] No 'query/pages' in response.")
