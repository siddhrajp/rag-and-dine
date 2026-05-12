# module2_rag/multimodal_fusion_ranking.py
# Module 2 Lesson 3: Multimodal Similarity Fusion and Retrieval Ranking
# Combines text and image retrieval results into a unified ranked list
# using score normalization and weighted late fusion.

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

    article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
    image_db   = Chroma(collection_name="food_images",         persist_directory=DB_DIR)

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


def init_clip_embedder():
    """Initialize CLIP model for image and query text embeddings (512-d)."""
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

    @torch.no_grad()
    def embed_query_clip_text(query: str):
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
        feats = clip_model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().numpy().astype(np.float32)

    print("✅ CLIP embedders ready")
    return embed_images, embed_query_clip_text

# ── Step 3: Utility Functions ──────────────────────────────────────────────────

def _unwrap(res: dict):
    """Unwrap Chroma's nested query output."""
    ids   = res.get("ids",       [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def _to_similarity(dists):
    """Convert distance (smaller=better) to similarity (larger=better)."""
    return 1.0 - np.array(dists, dtype=np.float32)


def _minmax(x):
    """Min-max normalize to [0, 1] with safe handling for constant arrays."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def print_fused(rows, title: str, max_chars: int = 90):
    """Print fused ranked results."""
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(snippet)

# ── Step 4: Retrieval Functions ────────────────────────────────────────────────

def retrieve_articles(article_db, embed_texts, query: str, k: int = 5, where: dict = None):
    """Text → article retrieval using MiniLM (384-d)."""
    q_vec = embed_texts([query])[0]
    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)


def retrieve_images_by_text(image_db, embed_query_clip_text, query: str, k: int = 5, where: dict = None):
    """Text → image retrieval using CLIP text encoder (512-d)."""
    q_vec = embed_query_clip_text(query)
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    return ids, docs, metas, _to_similarity(dists)

# ── Step 5: Multimodal Fusion ──────────────────────────────────────────────────

def fuse_rank(
    article_db, image_db, embed_texts, embed_query_clip_text,
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict = None,
    where_img: dict = None,
    top_n: int = 5
):
    """
    Weighted late fusion of text and image retrieval results.
    fused_score = w_text * norm(text_score) + w_img * norm(img_score)
    """
    # Retrieve per modality
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(
        article_db, embed_texts, query, k=k_text, where=where_text
    )
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(
        image_db, embed_query_clip_text, query, k=k_img, where=where_img
    )

    # Normalize within modality
    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    # Build fused candidate list
    rows = []
    for j in range(len(t_ids)):
        rows.append({
            "modality":   "article",
            "id":         t_metas[j].get("doc_id", t_ids[j]) if isinstance(t_metas[j], dict) else t_ids[j],
            "cuisine":    t_metas[j].get("cuisine",  "N/A")  if isinstance(t_metas[j], dict) else "N/A",
            "location":   t_metas[j].get("location", "N/A")  if isinstance(t_metas[j], dict) else "N/A",
            "source":     t_metas[j].get("source",   "N/A")  if isinstance(t_metas[j], dict) else "N/A",
            "text_score": float(t_norm[j]),
            "img_score":  0.0,
            "fused":      float(w_text * t_norm[j]),
            "snippet":    (t_docs[j] or "").replace("\n", " ").strip(),
        })

    for j in range(len(i_ids)):
        rows.append({
            "modality":   "image",
            "id":         i_metas[j].get("doc_id", i_ids[j]) if isinstance(i_metas[j], dict) else i_ids[j],
            "cuisine":    i_metas[j].get("cuisine",  "N/A")  if isinstance(i_metas[j], dict) else "N/A",
            "location":   i_metas[j].get("location", "N/A")  if isinstance(i_metas[j], dict) else "N/A",
            "source":     i_metas[j].get("source",   "N/A")  if isinstance(i_metas[j], dict) else "N/A",
            "text_score": 0.0,
            "img_score":  float(i_norm[j]),
            "fused":      float(w_img * i_norm[j]),
            "snippet":    (i_docs[j] or "").replace("\n", " ").strip(),
        })

    # Sort by fused score descending
    rows.sort(key=lambda r: r["fused"], reverse=True)
    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]

# ── Step 6: Demos ──────────────────────────────────────────────────────────────

def demo1_no_filters(article_db, image_db, embed_texts, embed_query_clip_text):
    """Demo 1: Balanced multimodal fusion, no metadata filters."""
    print("\n--- Demo 1: Multimodal Fusion (No Filters) ---")
    rows = fuse_rank(
        article_db, image_db, embed_texts, embed_query_clip_text,
        query="cozy noodles with warm atmosphere",
        k_text=5, k_img=5, w_text=0.6, w_img=0.4, top_n=5
    )
    print_fused(rows, title="Demo 1 — Multimodal fusion (no filters)")
    print("✅ Demo 1 complete")


def demo2_with_filters(article_db, image_db, embed_texts, embed_query_clip_text):
    """Demo 2: Fusion with metadata constraints."""
    print("\n--- Demo 2: Multimodal Fusion (Metadata Filters) ---")
    rows = fuse_rank(
        article_db, image_db, embed_texts, embed_query_clip_text,
        query="handmade pasta and romantic dinner",
        k_text=5, k_img=5, w_text=0.6, w_img=0.4,
        where_text={"location": "Pasadena"},
        where_img={"source": "recipe_image"},
        top_n=5
    )
    if len(rows) == 0:
        print("⚠️ No results found. Try relaxing filters.")
    else:
        print_fused(rows, title="Demo 2 — Multimodal fusion (metadata filters)")
    print("✅ Demo 2 complete")


def demo3_weight_tuning(article_db, image_db, embed_texts, embed_query_clip_text):
    """Demo 3: Compare text-heavy vs image-heavy fusion weights."""
    print("\n--- Demo 3: Weight Tuning ---")
    q = "fresh sushi and minimalist presentation"

    # Text-heavy
    rows_text = fuse_rank(
        article_db, image_db, embed_texts, embed_query_clip_text,
        query=q, k_text=5, k_img=5, w_text=0.8, w_img=0.2, top_n=5
    )
    print_fused(rows_text, title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)")

    # Image-heavy
    rows_img = fuse_rank(
        article_db, image_db, embed_texts, embed_query_clip_text,
        query=q, k_text=5, k_img=5, w_text=0.3, w_img=0.7, top_n=5
    )
    print_fused(rows_img, title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)")

    print("✅ Demo 3 complete")
    print("🎉 Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Module 2 Lesson 3: Multimodal Similarity Fusion & Ranking")
    print("=" * 55)

    article_db, image_db           = validate_databases()
    embed_texts                    = init_text_embedder()
    embed_images, embed_query_clip_text = init_clip_embedder()

    demo1_no_filters(article_db, image_db, embed_texts, embed_query_clip_text)
    demo2_with_filters(article_db, image_db, embed_texts, embed_query_clip_text)
    demo3_weight_tuning(article_db, image_db, embed_texts, embed_query_clip_text)


if __name__ == "__main__":
    main()
