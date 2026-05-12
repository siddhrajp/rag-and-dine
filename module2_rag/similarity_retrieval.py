# module2_rag/similarity_retrieval.py
# Module 2 Lesson 2: Similarity Retrieval with Metadata Filtering
# Implements hybrid text and image retrieval over multimodal Chroma vector databases.

# ── Imports ────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# ── Constants ──────────────────────────────────────────────────────────────────

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

# ── Step 1: Validate Vector Databases ─────────────────────────────────────────

def validate_databases():
    """Verify Chroma persistence directory and collections exist."""
    if not os.path.isdir(DB_DIR):
        raise RuntimeError(
            f"Vector database not found: '{DB_DIR}'. "
            "Please run construct_multimodal_vector_index.py first."
        )

    article_db = Chroma(
        collection_name="restaurant_articles",
        persist_directory=DB_DIR,
    )
    image_db = Chroma(
        collection_name="food_images",
        persist_directory=DB_DIR,
    )

    n_articles = article_db._collection.count()
    n_images   = image_db._collection.count()

    if n_articles <= 0 or n_images <= 0:
        raise RuntimeError(
            "One or more collections are empty. "
            "Please rerun construct_multimodal_vector_index.py."
        )

    print(f"✅ Article vectors: {n_articles}")
    print(f"✅ Image vectors:   {n_images}")

    return article_db, image_db

# ── Step 2: Initialize Embedding Models ───────────────────────────────────────

def init_text_embedder():
    """Initialize sentence transformer for text embeddings (384-d)."""
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(texts, batch_size=64):
        return text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
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
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(vecs)

    print("✅ Image embedder ready")
    return embed_images

# ── Step 3: Retrieval Utilities ────────────────────────────────────────────────

def _unwrap(res: dict):
    """Unwrap Chroma's nested query output."""
    ids   = res.get("ids",       [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180):
    """Print retrieval results in a readable format."""
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta     = metas[i] if i < len(metas) else {}
        dist     = float(dists[i]) if i < len(dists) else None
        snippet  = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        cuisine  = meta.get("cuisine",  "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id   = meta.get("doc_id",   "N/A") if isinstance(meta, dict) else "N/A"
        source   = meta.get("source",   "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | distance={dist:.4f}")
        print(f"{snippet}")

# ── Step 4: Retrieval Functions ────────────────────────────────────────────────

def retrieve_articles(article_db, embed_texts, query: str, k: int = 5, where: dict = None):
    """Similarity retrieval over restaurant articles with optional metadata filter."""
    q_vec = embed_texts([query])[0]
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)


def retrieve_images_by_image(image_db, embed_images, query_image_path: str, k: int = 5, where: dict = None):
    """Image-to-image similarity retrieval using CLIP embeddings."""
    q_vec = embed_images([query_image_path])[0]
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

# ── Step 5: Retrieval Demos ────────────────────────────────────────────────────

def demo1_article_no_filter(article_db, embed_texts):
    """Demo 1: Pure similarity search over restaurant articles."""
    print("\n--- Demo 1: Article Similarity Search (No Filter) ---")
    q = "cozy restaurant with noodles and warm atmosphere"
    ids, docs, metas, dists = retrieve_articles(article_db, embed_texts, q, k=5)
    print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")
    print("✅ Demo 1 complete")


def demo2_article_with_filter(article_db, embed_texts):
    """Demo 2: Similarity search with metadata location filter."""
    print("\n--- Demo 2: Article Similarity Search + Metadata Filter ---")
    q = "handmade pasta and romantic dinner"
    where_filter = {"location": "Pasadena"}
    ids, docs, metas, dists = retrieve_articles(
        article_db, embed_texts, q, k=5, where=where_filter
    )
    if len(ids) == 0:
        print("⚠️ No results found with current filter.")
    else:
        print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search + metadata filter")
    print("✅ Demo 2 complete")


def demo3_image_similarity(image_db, embed_images, query_index: int = 0):
    """Demo 3: Image-to-image similarity search using CLIP."""
    print("\n--- Demo 3: Image Similarity Search (Image → Image) ---")
    meta_all = image_db._collection.get(include=["metadatas"])["metadatas"]

    if query_index >= len(meta_all):
        raise ValueError("query_index out of range.")

    query_img = meta_all[query_index]["image_path"]
    print(f"Query image: {query_img}")

    where_filter = {"source": "recipe_image"}
    ids, docs, metas, dists = retrieve_images_by_image(
        image_db, embed_images, query_img, k=5, where=where_filter
    )
    print_hits(ids, docs, metas, dists, title="Demo 3 — Image similarity search (image→image)")
    print("✅ Demo 3 complete")
    print("🎉 Similarity Retrieval with Metadata Filtering COMPLETE")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("Module 2 Lesson 2: Similarity Retrieval with Metadata Filtering")
    print("=" * 50)

    article_db, image_db = validate_databases()
    embed_texts          = init_text_embedder()
    embed_images         = init_image_embedder()

    demo1_article_no_filter(article_db, embed_texts)
    demo2_article_with_filter(article_db, embed_texts)
    demo3_image_similarity(image_db, embed_images, query_index=0)


if __name__ == "__main__":
    main()
