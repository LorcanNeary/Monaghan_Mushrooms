import requests
from fastdownload import download_url
from PIL import Image

def search_wikimedia(term, max_images=5):
    """
    Search Wikimedia Commons for images matching `term`,
    return a list of direct image URLs.
    """
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": term,
        "gsrlimit": max_images,
        "gsrnamespace": 6,          # 💡 Only search File: pages (images)
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    headers = {
        # Per Wikimedia API etiquette, use a descriptive User-Agent
        "User-Agent": "MonaghanMushroomsImageFetcher/0.1 (contact: your-email@example.com)"
    }

    r = requests.get(url, params=params, headers=headers, timeout=15)

    # Check HTTP status
    try:
        r.raise_for_status()
    except Exception as e:
        print(f"[search_wikimedia] HTTP error: {e}")
        print("[search_wikimedia] Response text (truncated):")
        print(r.text[:500])
        return []

    # Try to parse JSON
    try:
        data = r.json()
    except Exception as e:
        print(f"[search_wikimedia] JSON decode error: {e}")
        print("[search_wikimedia] Raw response (truncated):")
        print(r.text[:500])
        return []

    if "query" not in data or "pages" not in data["query"]:
        print("[search_wikimedia] No 'query/pages' in response.")
        print("Full JSON (truncated):", str(data)[:500])
        return []

    pages = data["query"]["pages"]
    print(f"[search_wikimedia] Found {len(pages)} pages")

    urls = []
    for page_id, page in pages.items():
        title = page.get("title", "<no title>")
        if "imageinfo" in page:
            img_url = page["imageinfo"][0]["url"]
            print(f"  - {title} -> {img_url}")
            urls.append(img_url)
        else:
            print(f"  - {title} (no imageinfo)")

    return urls[:max_images]


if __name__ == "__main__":
    # Test it
    urls = search_wikimedia("bird", max_images=5)
    print("URLs:", urls)

    if not urls:
        print("No URLs returned from Wikimedia – check the debug output above.")
    else:
        # Download one
        dest = "bird.jpg"
        print(f"Downloading first image to {dest} ...")
        download_url(urls[0], dest, show_progress=False)

        im = Image.open(dest)
        im.thumbnail((256, 256))
        im.show()
        print("Done – thumbnail opened.")
