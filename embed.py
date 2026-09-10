"""Embed the cached documents and the five test queries with a local model."""

import json

import numpy as np

from data import ARTIFACTS, DOCUMENTS

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

DOC_VECTORS = ARTIFACTS / "doc_vectors.npy"
QUERIES = ARTIFACTS / "queries.json"
QUERY_VECTORS = ARTIFACTS / "query_vectors.npy"

# single source of truth for the query text (queries.md is just docs)
TEST_QUERIES = [
    "a question about a graphics card driver",
    "what does the Bible say about the resurrection",
    "for sale: used motorcycle in good condition",
    "government encryption policy and the clipper chip",
    "treatment options for chronic back pain",
]


def load_vectors():
    """Load what main() wrote, for the other scripts."""
    documents = json.loads(DOCUMENTS.read_text())
    queries = json.loads(QUERIES.read_text())
    return (
        documents,
        np.load(DOC_VECTORS),
        queries,
        np.load(QUERY_VECTORS),
    )


def build_model():
    """MiniLM with its final Normalize layer removed.

    Left in, every vector is unit length and cosine/dot/euclid rank identically,
    so Part 2 has nothing to compare. Raw mean-pooled vectors have varying norms.
    """
    # local import so importing load_vectors() doesn't pull in torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize

    model = SentenceTransformer(MODEL_NAME)
    modules = []
    for module in model:
        if not isinstance(module, Normalize):
            modules.append(module)
    return SentenceTransformer(modules=modules)


def main():
    documents = json.loads(DOCUMENTS.read_text())
    model = build_model()

    texts = []
    for d in documents:
        texts.append(d["text"])
    doc_vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    query_vectors = model.encode(
        TEST_QUERIES,
        normalize_embeddings=False,
    ).astype(np.float32)

    ARTIFACTS.mkdir(exist_ok=True)
    np.save(DOC_VECTORS, doc_vectors)
    np.save(QUERY_VECTORS, query_vectors)
    QUERIES.write_text(json.dumps(TEST_QUERIES, indent=2))

    norms = np.linalg.norm(doc_vectors, axis=1)
    print(f"\nembedded {doc_vectors.shape[0]} documents into {doc_vectors.shape[1]} dimensions")
    print(f"vector norms: min {norms.min():.3f}  mean {norms.mean():.3f}  max {norms.max():.3f}")
    print("that spread is what makes dot product rank differently from cosine")
    if norms.std() < 1e-3:
        raise SystemExit("vectors came out unit length; Part 2 needs varying magnitudes")


if __name__ == "__main__":
    main()
