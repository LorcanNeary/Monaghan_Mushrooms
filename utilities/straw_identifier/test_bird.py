from ddgs import DDGS
from fastdownload import download_url
from fastai.vision.all import *
from PIL import Image

# Search for bird photos (DuckDuckGo)
def search_images(term, max_images=1):
    with DDGS() as ddgs:
        return [r["image"] for r in ddgs.images(term, max_results=max_images)]

urls = search_images('bird photos', max_images=1)
print("Image URL:", urls[0])

# Download the image
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

# Open and display a thumbnail
im = Image.open(dest)
im.thumbnail((256, 256))
im.show()
