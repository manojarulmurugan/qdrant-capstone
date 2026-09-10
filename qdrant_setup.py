"""Create the Qdrant collections and upload the embedded documents into them."""

import time

from qdrant_client import QdrantClient, models

from embed import VECTOR_SIZE, load_vectors

QDRANT_URL = "http://localhost:6333"

# Part 2: one collection per distance metric, same vectors and payload.
METRIC_COLLECTIONS = {
    "news_cosine": models.Distance.COSINE,
    "news_euclid": models.Distance.EUCLID,
    "news_dot": models.Distance.DOT,
}

# Qdrant's query planner only walks the HNSW graph once a segment holds more
# than full_scan_threshold KB of vectors; below that it brute forces the segment
# because at that size a scan is genuinely faster. The default is 10000 KB and
# these 6000 384-dimensional float32 vectors come to 9000 KB, so at the default
# the graph is built and then never used, and every HNSW config scores an
# identical 100% against exact search. 10 KB is the lowest Qdrant accepts and is
# far below 9000 KB, so the graph becomes load bearing for every query.
FULL_SCAN_THRESHOLD = 10

# Part 3: same data and metric as news_cosine, only the HNSW params differ
# (defaults are m=16, ef_construct=100).
DEFAULT_HNSW = models.HnswConfigDiff(full_scan_threshold=FULL_SCAN_THRESHOLD)
WEAK_COLLECTION = "news_cosine_weak"
WEAK_HNSW = models.HnswConfigDiff(
    m=4, ef_construct=8, full_scan_threshold=FULL_SCAN_THRESHOLD
)


def connect(url=QDRANT_URL):
    return QdrantClient(url=url)


def create_collection(client, name, distance, hnsw_config=None):
    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=distance),
        hnsw_config=hnsw_config,
        optimizers_config=models.OptimizersConfigDiff(
            # force an HNSW build: our ~9 MB of vectors is under the 20 MB
            # default threshold, so otherwise Qdrant just serves brute force
            indexing_threshold=1,
            # one segment = one graph, so only m/ef_construct differ
            default_segment_number=1,
        ),
    )


def upload(client, name, documents, doc_vectors):
    ids = []
    payloads = []
    for d in documents:
        ids.append(d["id"])
        payloads.append({"category": d["category"], "text": d["text"]})
    client.upload_collection(
        collection_name=name,
        ids=ids,
        vectors=doc_vectors,
        payload=payloads,
        batch_size=256,
    )


def wait_until_indexed(client, name, timeout=300):
    """Block until indexing finishes, so timings aren't taken mid-build."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count:
            return info
        time.sleep(0.5)
    raise TimeoutError(f"{name} was still indexing after {timeout}s")


def main():
    documents, doc_vectors, _, _ = load_vectors()
    client = connect()

    plan = []
    for name, distance in METRIC_COLLECTIONS.items():
        plan.append((name, distance, DEFAULT_HNSW))
    plan.append((WEAK_COLLECTION, models.Distance.COSINE, WEAK_HNSW))

    for name, distance, hnsw_config in plan:
        create_collection(client, name, distance, hnsw_config)
        upload(client, name, documents, doc_vectors)
        info = wait_until_indexed(client, name)
        label = "m=4 ef_construct=8" if hnsw_config is WEAK_HNSW else "m=16 ef_construct=100"
        print(f"{name:20} {distance.value:8} {info.indexed_vectors_count:5d} indexed  ({label})")


if __name__ == "__main__":
    main()
