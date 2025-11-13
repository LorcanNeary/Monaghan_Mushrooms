import time
import os
import pathlib
from io import BytesIO

import streamlit as st
from ddgs import DDGS
from fastdownload import download_url
from PIL import Image

# -------------- Config --------------
DEFAULT_MAX_IMAGES = 8
DOWNLOAD_DIR = pathlib.Path("downloads")
THUMB_SIZE = (256, 256)

# A small negative/positive prompt to steer results
PROMPT_TEMPLATES = {
    "wheat":  "wheat straw bales close-up daylight consistent lighting -people -animal -art -illustration",
    "bean":   "bean straw bales close-up daylight consistent lighting -people -animal -art -illustration",
    "barley": "barley straw bales close-up daylight consistent lighting -people -animal -art -illustration",
    "oat":    "oat straw bales close-up daylight consistent lighting -people -animal -art -illustration",
}

# ------------------------------------


@st.cache_data(show_spinner=False)
def search_image_urls(query: str, max_images: int, retries: int = 4, delay: float = 2.0):
    """Return a list of image URLs using ddgs, with simple retry/backoff."""
    urls = []
    for attempt in range(1, retries + 1):
        try:
            with DDGS() as ddgs:
                # safesearch: "off" | "moderate" | "strict"
                results = ddgs.images(query, max_results=max_images, safesearch="moderate")
                urls = [r.get("image") for r in results if r.get("image")]
                if urls:
                    break
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(delay * attempt)
    return urls[:max_images]


def safe_filename(url: str, idx: int, straw_type: str) -> str:
    """Create a consistent local filename from URL."""
    stem = f"{straw_type}_straw_{idx:03d}"
    ext = ".jpg"
    # try to infer extension from url
    for cand in [".jpg", ".jpeg", ".png", ".webp"]:
        if cand in url.lower():
            ext = ".jpg" if cand in (".jpg", ".jpeg") else cand
            break
    return stem + ext


def create_thumbnail(image_path: pathlib.Path, thumb_size=THUMB_SIZE) -> Image.Image:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail(thumb_size)
        return im.copy()


def download_one(url: str, out_dir: pathlib.Path, filename: str) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    download_url(url, target, show_progress=False)
    return target


def draw_image_card(idx: int, img: Image.Image, local_path: pathlib.Path):
    col = st.container()
    col.image(img, caption=f"{local_path.name}", use_container_width=True)
    with open(local_path, "rb") as f:
        col.download_button(
            label="Download original",
            data=f.read(),
            file_name=local_path.name,
            mime="image/jpeg",
            key=f"dl_{local_path.name}_{idx}",
        )


def main():
    st.set_page_config(page_title="Straw Image Picker", page_icon="🌾", layout="wide")
    st.title("🌾 Straw Image Picker")
    st.write(
        "Pick a straw type and fetch close-up images of **straw bales** in consistent daylight. "
        "Results are downloaded locally and displayed as thumbnails."
    )

    # Controls
    straw_type = st.selectbox(
        "Straw type:",
        options=["wheat", "bean", "barley", "oat"],
        index=0,
    )

    n_images = st.slider("How many images?", 1, 24, DEFAULT_MAX_IMAGES, step=1)
    subdir = st.text_input("Save into folder:", value=f"{straw_type}_straw_bales")
    query_extra = st.text_input(
        "Optional extra query terms (advanced):",
        value="close-up bale texture uniform light field",
        help="Add extra constraints to steer the search (e.g. 'macro', 'daylight', 'field').",
    )

    col_go, col_clear = st.columns([1, 1])
    go = col_go.button("Search & Download")
    clear_cache = col_clear.button("Clear cached search results")

    if clear_cache:
        st.cache_data.clear()
        st.success("Cleared cached results.")

    # Action
    if go:
        base_query = PROMPT_TEMPLATES[straw_type]
        full_query = f"{base_query} {query_extra}".strip()
        st.info(f"Query: `{full_query}`")

        try:
            with st.spinner("Searching images…"):
                urls = search_image_urls(full_query, n_images)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

        if not urls:
            st.warning("No results returned. Try reducing the image count, changing the query, or running again (DDG can rate-limit).")
            st.stop()

        st.write(f"Found {len(urls)} URLs. Downloading…")

        out_dir = DOWNLOAD_DIR / subdir
        paths = []
        errors = []

        prog = st.progress(0.0, text="Downloading images…")
        for i, url in enumerate(urls, start=1):
            fname = safe_filename(url, i, straw_type)
            try:
                p = download_one(url, out_dir, fname)
                paths.append(p)
            except Exception as e:
                errors.append((url, str(e)))
            prog.progress(i / len(urls))

        if errors:
            with st.expander("Some downloads failed (click to expand)"):
                for u, msg in errors:
                    st.write(f"- {u} — {msg}")

        if not paths:
            st.error("All downloads failed. Try again (DuckDuckGo sometimes rate-limits).")
            st.stop()

        # Show thumbnails in a grid
        st.success(f"Downloaded {len(paths)} image(s) to `{out_dir.as_posix()}`")
        cols = st.columns(4)
        for idx, p in enumerate(paths):
            try:
                thumb = create_thumbnail(p)
                with cols[idx % 4]:
                    draw_image_card(idx, thumb, p)
            except Exception as e:
                st.write(f"Could not render {p.name}: {e}")

        # Zip all button
        import shutil, tempfile
        with tempfile.TemporaryDirectory() as tmpd:
            zip_path = pathlib.Path(tmpd) / f"{subdir}.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", out_dir)
            with open(zip_path, "rb") as f:
                st.download_button(
                    "Download all as ZIP",
                    f.read(),
                    file_name=f"{subdir}.zip",
                    mime="application/zip",
                )

    st.caption(
        "Tip: If you hit rate limits, run again later, reduce image count, or tweak the query. "
        "This app uses DuckDuckGo (`ddgs`) which can intermittently throttle."
    )


if __name__ == "__main__":
    main()