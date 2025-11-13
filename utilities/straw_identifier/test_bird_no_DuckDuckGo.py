from fastdownload import download_url
from fastai.vision.all import *
from PIL import Image

def search_images(term, max_images=1):
    fixed_images = {
        "bird photos": [
            "https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_bird.jpg"
        ],
        "forest photos": [
            "https://upload.wikimedia.org/wikipedia/commons/a/a2/Forest_pine_trees.jpg"
        ]
    }
    return fixed_images.get(term, [])[:max_images]

urls = search_images("bird photos", max_images=1)
print("Image URL:", urls[0])

dest = "bird.jpg"
download_url(urls[0], dest, show_progress=False)

im = Image.open(dest)
im.thumbnail((256, 256))
im.show()