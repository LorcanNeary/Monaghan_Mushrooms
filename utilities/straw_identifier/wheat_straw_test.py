# wheat_straw_test.py
import time
from ddgs import DDGS                  # <-- use ddgs (renamed package)
from fastdownload import download_url
from PIL import Image

def search_images(term, max_images=1, retries=3, delay=2):
    """Return a list of image URLs with simple retry/backoff."""
    for attempt in range(1, retries + 1):
        try:
            with DDGS() as ddgs:
                # safesearch can be: "off", "moderate", "strict"
                results = ddgs.images(keywords=term, max_results=max_images, safesearch="moderate")
                urls = [r["image"] for r in results if "image" in r]
                if urls:
                    return urls
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(delay * attempt)  # backoff and try again
    return []

term = "wheat straw photos"
urls = search_images(term, max_images=1)
print("Image URL:", urls[0])

# Download
dest = "wheat_straw.jpg"
download_url(urls[0], dest, show_progress=False)

# Make and save a thumbnail (avoid .show() subprocess warning)
thumb = "wheat_straw_thumb.jpg"
with Image.open(dest) as im:
    im.thumbnail((256, 256))
    im.save(thumb)
print("Saved thumbnail:", thumb)
