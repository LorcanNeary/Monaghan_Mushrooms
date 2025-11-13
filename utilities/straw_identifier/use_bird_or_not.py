"""
use_bird_or_not.py

Script to:
1. Load the trained 'bird_or_not' fastai model (exported as bird_or_not_model.pkl)
2. Run predictions on one or more input images
"""

from pathlib import Path
import sys

from fastai.vision.all import *


def load_model(model_file: str = "bird_or_not_model.pkl"):
    """
    Load the exported fastai learner.

    Args:
        model_file: Path to the exported model (.pkl)
    Returns:
        fastai Learner
    """
    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_file}' not found. "
            f"Make sure you've run the training script first."
        )
    learn = load_learner(model_path)
    return learn


def predict_image(learn, img_path: str):
    """
    Run prediction on a single image.

    Args:
        learn:    Loaded fastai learner
        img_path: Path to the image file

    Prints:
        Predicted class and probabilities
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image '{img_path}' not found.")

    img = PILImage.create(img_path)
    pred_class, pred_idx, probs = learn.predict(img)

    print(f"Image: {img_path}")
    print(f"Predicted class: {pred_class}")
    print("Class probabilities:")
    for c, p in zip(learn.dls.vocab, probs):
        print(f"  {c:10s} : {p:.4f}")


def main():
    """
    Example usage:

        python use_bird_or_not.py path/to/image.jpg

    If no image path is provided, the script will prompt the user.
    """
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to image: ").strip()

    learn = load_model("bird_or_not_model.pkl")
    predict_image(learn, image_path)


if __name__ == "__main__":
    main()
