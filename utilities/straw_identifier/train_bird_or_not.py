"""
train_bird_or_not.py

Script to:
1. Download example bird/forest images
2. Build a 'bird_or_not' dataset using DuckDuckGo image search
3. Train a ResNet18 classifier using fastai
4. Export the trained model for later use
"""

from pathlib import Path
from time import sleep

from fastdownload import download_url
from fastai.vision.all import *

# -------------------------------------------------------------------------
# DuckDuckGo Image Search Helper (from fastbook)
# -------------------------------------------------------------------------

import time, random, json
from fastcore.all import L

def search_images(term, max_images=30):
    """
    Search DuckDuckGo for images matching `term`, return list of URLs.
    This is the same function used in fastbook but copied here so the
    script can run outside the fastbook environment.
    """

    # DuckDuckGo query URL
    url = 'https://duckduckgo.com/'
    params = {'q': term}

    # First request to obtain a token ("vqd")
    res = requests.get(url, params=params)
    res.raise_for_status()

    # Extract vqd token
    token_text = res.text
    start = token_text.find('vqd=')
    if start == -1:
        raise Exception("Could not find vqd token in DuckDuckGo response")
    start += 5
    end = token_text.find("'", start)
    vqd = token_text[start:end]

    # Second request: actual image search
    request_url = 'https://duckduckgo.com/i.js'
    params = {
        'l': 'us-en',
        'o': 'json',
        'q': term,
        'vqd': vqd,
        'p': '1',
        'f': ',,,'
    }

    urls = []
    while len(urls) < max_images:
        try:
            res = requests.get(request_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            urls.extend(L(data['results']).itemgot('image'))
            if 'next' not in data:
                break
            request_url = url + data['next']
            time.sleep(0.2 + random.random() / 10)
        except Exception:
            break

    return urls[:max_images]


# -----------------------------------------------------------------------------
# 0. Quick sanity check images (optional, can be removed if you like)
# -----------------------------------------------------------------------------

def download_example_images():
    """
    Download one bird image and one forest image and save them locally.
    This is mostly just a sanity check that search_images & download_url work.
    """
    # NB: search_images depends on duckduckgo.com, which doesn't always return
    # correct responses. If you get a JSON error, just try running it again.
    urls = search_images('bird photos', max_images=1)
    dest = Path('bird.jpg')
    download_url(urls[0], dest, show_progress=False)

    # Show thumbnail to visually confirm (only works in notebook, harmless in script)
    im = Image.open(dest)
    im.thumbnail((256, 256))

    forest_url = search_images('forest photos', max_images=1)[0]
    download_url(forest_url, 'forest.jpg', show_progress=False)
    forest_im = Image.open('forest.jpg')
    forest_im.thumbnail((256, 256))


# -----------------------------------------------------------------------------
# 1. Build dataset: download bird / forest images and resize
# -----------------------------------------------------------------------------

def build_dataset():
    """
    Create a dataset in the 'bird_or_not' folder with two subfolders:
        bird_or_not/bird
        bird_or_not/forest

    Each folder is populated with images downloaded via DuckDuckGo.
    """
    searches = ('forest', 'bird')
    path = Path('bird_or_not')

    for o in searches:
        dest = path / o
        dest.mkdir(exist_ok=True, parents=True)

        # NB: DuckDuckGo can be flaky – JSON errors are not uncommon.
        # If this fails, just run the script again.
        print(f"Downloading images for: {o}")
        download_images(dest, urls=search_images(f'{o} photo'))
        sleep(10)  # Pause between searches to avoid over-loading the server
        download_images(dest, urls=search_images(f'{o} sun photo'))
        sleep(10)
        download_images(dest, urls=search_images(f'{o} shade photo'))
        sleep(10)

        # Resize images to a manageable size
        print(f"Resizing images in {dest} ...")
        resize_images(dest, max_size=400, dest=dest)

    return path


# -----------------------------------------------------------------------------
# 2. Train the model
# -----------------------------------------------------------------------------

def train_model(data_path: Path, n_epochs: int = 3, bs: int = 32):
    """
    Train a ResNet18 classifier on images found in data_path.

    Args:
        data_path: Path containing the class folders (e.g. bird_or_not/bird, bird_or_not/forest)
        n_epochs:  Number of epochs to fine-tune
        bs:        Batch size
    Returns:
        Trained learner
    """

    # Quick sanity check: how many images did we actually get?
    all_files = get_image_files(data_path)
    print(f"Total images found: {len(all_files)} in {data_path}")

    # Remove corrupted/failed images
    print("Verifying images...")
    failed = verify_images(all_files)
    failed_count = len(failed)
    print(f"Found {failed_count} failed images.")
    failed.map(Path.unlink)

    # Re-scan after deleting bad ones
    all_files = get_image_files(data_path)
    print(f"Images remaining after cleanup: {len(all_files)}")

    # Create DataLoaders using the simple high-level API
    print("Creating DataLoaders (ImageDataLoaders.from_folder)...")
    dls = ImageDataLoaders.from_folder(
        data_path,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(192, method='squish'),
        bs=bs
    )

    # Optional: comment out if you don't care to see the batch / avoid matplotlib popping up
    # dls.show_batch(max_n=6)

    # Create and train the model
    print("Training model...")
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(n_epochs)

    return learn


# -----------------------------------------------------------------------------
# 3. Main entry point
# -----------------------------------------------------------------------------

def main():
    # Optional sanity check – can be commented out if not needed
    # download_example_images()

    # 1) Build dataset
    data_path = build_dataset()

    # 2) Train model
    learn = train_model(data_path, n_epochs=3, bs=32)

    # 3) Export trained model
    model_path = Path("bird_or_not_model.pkl")
    learn.export(model_path)
    print(f"Model exported to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
