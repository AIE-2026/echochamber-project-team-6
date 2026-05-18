# core/retriever.py
# ==================
# Loads a FAISS vector index for one agent and performs semantic search.
# This is the RAG (Retrieval-Augmented Generation) component.
#
# HOW RAG WORKS HERE:
#   When an agent responds, it first searches its own corpus for the
#   most similar comments to the stimulus. These are injected into the
#   prompt, grounding the agent's response in authentic community language.
#
# Students: you don't need to modify this file.
# To build the indexes, run: python scripts/build_vectorstore.py


# ==========================================================


# core/retriever.py
# =================
# Semantic retrieval over a FAISS vector store.
#
# This file does NOT use a generative LLM.
# It only does retrieval:
#
#   query text → query embedding → FAISS search → top‑k fragments
#
# In C6, these fragments will be inserted into an LLM prompt.

from pathlib import Path
import argparse
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------
# Config
# --------------------------------------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = Path("assets/vectorstores")


class Retriever:
    """
    Retriever for a single *agent* (bubble).

    Expected layout:
        assets/vectorstores/<agent_slug>/index.faiss   – FAISS index
        assets/vectorstores/<agent_slug>/index.pkl    – list of metadata dicts
    """

    def __init__(self, agent_slug: str):
        self.agent_slug = agent_slug
        self.path = VECTORSTORE_DIR / agent_slug

        # TODO1
        index_path = self.path / "index.faiss"
        metadata_path = self.path / "index.pkl"

        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # TODO2
        self.index = faiss.read_index(str(index_path))

        # TODO3
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # TODO4
        self.model = SentenceTransformer(MODEL_NAME)

        # TODO5
    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Return the *k* most similar fragments for ``query``.

        Each result dict contains the original metadata plus:
            - ``score``    (float, similarity)
            - ``position`` (int, index in the FAISS store)
        """
        
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            device="cpu",
        ).astype(np.float32)

        scores, positions = self.index.search(query_vec, k)   # both (1, k)

        results: list[dict] = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:                # FAISS uses -1 for empty slots
                continue
            # copy metadata to avoid mutating the original structure
            meta = self.metadata[pos].copy() if isinstance(self.metadata[pos], dict) \
                else dict(self.metadata[pos])
            meta["score"] = float(score)
            meta["position"] = int(pos)
            results.append(meta)

        return results

        # TODO6
    def format_for_prompt(self, chunks: list[dict]) -> str:
        """
        Convert a list of fragment dicts into a readable string.

        Example for a single fragment:
        [Fragment 1 | score=0.812]
        Lorem ipsum dolor sit amet...

        The string can be inserted verbatim into a later LLM prompt.
        """
        if not chunks:
            return "(Nu au fost găsite fragmente relevante.)"

        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            lines.append(f"[Fragment {i} | score={score:.3f}]\n{text}")

        return "\n\n".join(lines)

def main() -> None:
    """
    Simple command‑line test.

    Example:
        python -m core.retriever \\
            --agent anti_sistem \\
            --query "CCR a decis anularea alegerilor..." \\
            --k 5
    """
    parser = argparse.ArgumentParser(
        description="Test semantic retrieval for one agent bubble."
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent slug, e.g. anti_sistem",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Text used as semantic search query",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved fragments",
    )
    args = parser.parse_args()

    retriever = Retriever(args.agent)
    chunks = retriever.search(args.query, k=args.k)

    print("Agent:", args.agent)
    print("Interogare:", args.query)
    print("Vectori în index:", retriever.index.ntotal)
    print("Rezultate recuperate:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nRezultat {i}")
        print("Poziție:", chunk["position"])
        print("Scor:", round(chunk["score"], 3))

        if "agent" in chunk:
            print("Agent text:", chunk["agent"])
        if "source_channel" in chunk:
            print("Sursă:", chunk["source_channel"])
        if "video_title" in chunk:
            print("Video:", chunk["video_title"])

        print("Text:", chunk.get("text", "")[:500])


if __name__ == "__main__":
    main()