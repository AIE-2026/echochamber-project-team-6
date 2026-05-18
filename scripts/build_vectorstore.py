# scripts/build_vectorstore.py
# Builds FAISS vector stores for cleaned agent bubble files.

from pathlib import Path
import os
import pickle

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# Move to the project root so paths work from any starting folder.
while not Path("data/bubbles").exists():
    os.chdir("..")


# Define input/output folders and embedding model.
BUBBLES_DIR = Path("data/bubbles")
VECTOR_DIR = Path("assets/vectorstores")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# Load multilingual embedding model.
model = SentenceTransformer(MODEL_NAME)


# Read each cleaned bubble file from data/bubbles.
for bubble_path in BUBBLES_DIR.glob("*.jsonl"):
    slug = bubble_path.stem
    print(f"Processing bubble: {slug}")

    df = pd.read_json(bubble_path, lines=True)
    texts = df["text"].fillna("").tolist()
    metadata = df.to_dict(orient="records")

    # Convert texts into vector embeddings.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    # Create FAISS index and add embeddings.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index and metadata for retrieval.
    out_dir = VECTOR_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))

    with open(out_dir / "index.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Print progress for verification.
    print("Saved in:", out_dir)
    print("Vectors in index:", index.ntotal)