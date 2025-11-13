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

    r = requests.get(url, params=params)
    data = r.json()

    if "query" not in data or "pages" not in data["query"]:
        return []

    urls = []
    for _, page in data["query"]["pages"].items():
        if "imageinfo" in page:
            urls.append(page["imageinfo"][0]["url"])

    return urls[:max_images]


# Test it
urls = search_wikimedia("bird", max_images=5)
print(urls)

# Download one
download_url(urls[0], "bird.jpg")
im = Image.open("bird.jpg")
im.thumbnail((256, 256))
im.show()
