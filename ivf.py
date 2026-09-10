"""Minimal IVF index: k-means clusters plus an nprobe cluster scan.

Qdrant only does HNSW for dense vectors, so the IVF side is hand-rolled here.
"""

import time

import numpy as np
from sklearn.cluster import KMeans

from embed import load_vectors

N_CLUSTERS = 48
SEED = 42


def unit(vectors):
    """Normalise each row to unit length (so dot product == cosine)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class IVFIndex:
    """Centroids plus the vector ids in each cluster."""

    def __init__(self, centroids, postings, vectors):
        self.centroids = centroids  # (n_clusters, dim), unit length
        self.postings = postings    # postings[c] = ids of the vectors in cluster c
        self.vectors = vectors      # (n_docs, dim), unit length

    def probe(self, query_vector, nprobe):
        """Ids of every vector in the nprobe nearest clusters.

        Ranks the 48 centroids, not the 6000 docs. Anything outside those
        clusters is missed -- that's where IVF trades recall for speed.
        """
        query = unit(np.asarray(query_vector)[None])[0]
        centroid_scores = self.centroids @ query
        probes = np.argsort(-centroid_scores)[:nprobe]
        chosen = []
        for c in probes:
            chosen.append(self.postings[c])
        return np.concatenate(chosen)

    def search_ivf(self, query_vector, nprobe, k=5):
        """Exact cosine search, restricted to the nprobe nearest clusters."""
        query = unit(np.asarray(query_vector)[None])[0]
        candidates = self.probe(query_vector, nprobe)

        scores = self.vectors[candidates] @ query
        # argsort is fine at this size; a real index would argpartition to k
        top = np.argsort(-scores)[:k]
        results = []
        for i in top:
            results.append((int(candidates[i]), float(scores[i])))
        return results


def build_ivf_index(vectors, n_clusters=N_CLUSTERS, seed=SEED):
    """Partition `vectors` into n_clusters buckets with k-means."""
    # normalise first so k-means (Euclidean) lines up with cosine scoring
    # -- i.e. spherical k-means
    normalised = unit(vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(normalised)

    # mean of unit vectors isn't unit length, so renormalise
    centroids = unit(kmeans.cluster_centers_)
    postings = []
    for c in range(n_clusters):
        mask = (labels == c)          # bool array, True where doc belongs to cluster c
        member_ids = np.flatnonzero(mask)  # the positions where mask is True
        postings.append(member_ids)

    return IVFIndex(centroids, postings, normalised)


def main():
    documents, doc_vectors, queries, query_vectors = load_vectors()

    start = time.perf_counter()
    index = build_ivf_index(doc_vectors)
    build_seconds = time.perf_counter() - start

    sizes = []
    for p in index.postings:
        sizes.append(len(p))
    print(f"built {N_CLUSTERS} clusters over {len(documents)} vectors in {build_seconds:.1f}s")
    print(f"cluster sizes: min {min(sizes)}  median {int(np.median(sizes))}  max {max(sizes)}")

    for query, query_vector in zip(queries, query_vectors):
        print(f"\n{query!r}")
        for nprobe in (1, 8):
            scanned = len(index.probe(query_vector, nprobe))
            print(f"  nprobe={nprobe:<2} scanned {scanned:5d} of {len(documents)} vectors")
            for doc_id, score in index.search_ivf(query_vector, nprobe=nprobe, k=5):
                print(f"    {score:.4f}  {documents[doc_id]['category']}")


if __name__ == "__main__":
    main()
