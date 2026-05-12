# module2_rag/construct_multimodal_vector_index.py
# Module 2 Lesson 1: Construct a Multimodal Vector Index
# Builds text and image vector databases using ChromaDB
# for the restaurant RAG retrieval system.

# ── Imports ────────────────────────────────────────────────────────────────────

import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# ── Constants ──────────────────────────────────────────────────────────────────

ZIP_URL  = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
ZIP_PATH = "synthetic-recipe-images.zip"
IMG_DIR  = "recipe_images"
DB_DIR   = str((Path.home() / "chroma_multimodal").resolve())

# ── Step 1: Download and Prepare Image Data ────────────────────────────────────

def download_images():
    """Download and extract recipe image dataset."""
    print("Downloading recipe images...")
    os.system(f"wget -q -O {ZIP_PATH} {ZIP_URL}")
    os.system(f"unzip -oq {ZIP_PATH} -d {IMG_DIR}")
    image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))
    print(f"✅ Images found: {len(image_paths)}")
    return image_paths

# ── Step 2: Load Structured Data from Module 1 ─────────────────────────────────

def load_data():
    """Load restaurant and recipe JSON files from Module 1."""
    print("Loading structured data...")
    with open("data/structured_restaurant_data.json", "r") as f:
        restaurants = json.load(f)
    with open("data/augmented_food_recipe.json", "r") as f:
        recipes = json.load(f)
    print(f"✅ Loaded restaurants: {len(restaurants)}")
    print(f"✅ Loaded recipes:     {len(recipes)}")
    return restaurants, recipes

# ── Step 3: Initialize Embedding Models ───────────────────────────────────────

def init_text_embedder():
    """Initialize sentence transformer for text embeddings (384-d)."""
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(texts, batch_size=64):
        return text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine-ready
        ).astype(np.float32)

    print("✅ Text embedder ready")
    return embed_texts


def init_image_embedder():
    """Initialize CLIP model for image embeddings (512-d)."""
    device = "cpu"
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()

    @torch.no_grad()
    def embed_images(paths, batch_size=16):
        vecs = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(**inputs)       # (B, 512)
            feats = feats / feats.norm(dim=-1, keepdim=True)      # cosine-ready
            vecs.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(vecs)

    print("✅ Image embedder ready")
    return embed_images

# ── Step 4: Build Documents ────────────────────────────────────────────────────

def build_documents(restaurants, recipes, image_paths):
    """Build article and image documents for vector indexing."""

    # Article documents from restaurant data
    article_docs = []
    for i, r in enumerate(restaurants):
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style', '')}\n"
            f"Location: {r.get('location', '')}"
        )
        article_docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": f"rest_{i}",
                    "cuisine": r.get("food_style"),
                    "location": r.get("location"),
                    "source": "restaurant",
                },
            )
        )
    print(f"✅ Article docs: {len(article_docs)}")

    # Image documents from recipe images
    image_docs = []
    for i, (p, rec) in enumerate(zip(image_paths, recipes)):
        image_docs.append(
            Document(
                page_content=rec.get("name", f"recipe image {i}"),
                metadata={
                    "doc_id": f"img_{i}",
                    "image_path": p,
                    "source": "recipe_image",
                    "recipe_id": rec.get("id"),
                    "cuisine": rec.get("cuisine"),
                },
            )
        )
    print(f"✅ Image docs: {len(image_docs)}")

    return article_docs, image_docs

# ── Step 5: Build and Persist Vector Indexes ───────────────────────────────────

def build_vector_index(article_docs, image_docs, embed_texts, embed_images):
    """Embed documents and store in Chroma vector databases."""

    # Reset DB directory for clean rerun
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)

    # Article vector DB
    print("Building article vector DB...")
    A = embed_texts([d.page_content for d in article_docs])
    article_db = Chroma(
        collection_name="restaurant_articles",
        persist_directory=DB_DIR,
    )
    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=A.tolist(),
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )
    print("✅ Article DB ready")

    # Image vector DB
    print("Building image vector DB...")
    V = embed_images([d.metadata["image_path"] for d in image_docs])
    image_db = Chroma(
        collection_name="food_images",
        persist_directory=DB_DIR,
    )
    image_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in image_docs],
        embeddings=V.tolist(),
        documents=[d.page_content for d in image_docs],
        metadatas=[d.metadata for d in image_docs],
    )
    print("✅ Image DB ready")
    print("🎉 Multimodal Vector Index Construction COMPLETE")

    return article_db, image_db

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("Module 2 Lesson 1: Construct Multimodal Vector Index")
    print("=" * 50)

    image_paths          = download_images()
    restaurants, recipes = load_data()
    embed_texts          = init_text_embedder()
    embed_images         = init_image_embedder()
    article_docs, image_docs = build_documents(restaurants, recipes, image_paths)
    build_vector_index(article_docs, image_docs, embed_texts, embed_images)


if __name__ == "__main__":
    main()
