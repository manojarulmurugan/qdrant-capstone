"""Run all five queries through every search method and record scores and latency."""

import argparse
import json
import statistics
import time
from pathlib import Path

from qdrant_client import models

from embed import load_vectors
from ivf import build_ivf_index
from qdrant_setup import METRIC_COLLECTIONS, WEAK_COLLECTION, connect

RESULTS = Path(__file__).resolve().parent / "results"

TOP_K = 5
EF_VALUES = (16, 64, 128)
NPROBE_VALUES = (1, 8)
REPEATS = 5

EXACT = "exact"
HNSW_COLLECTIONS = {"news_cosine": "hnsw-default", WEAK_COLLECTION: "hnsw-weak"}


def snippet(text, width=110):
    """Collapse text to one line for the JSON and console output."""
    return " ".join(text.split())[:width]


def timed(call, repeats=REPEATS):
    """(result, median ms) over `repeats` runs.

    First call is discarded -- it warms the connection and disk cache. Median
    so a stray GC pause doesn't move the number.
    """
    call()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        durations.append((time.perf_counter() - start) * 1000)
    return result, statistics.median(durations)


def qdrant_search(client, collection, query_vector, search_params=None, k=TOP_K):
    response = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=k,
        search_params=search_params,
        # category only -- full text over HTTP would dominate the latency
        with_payload=["category"],
    )
    hits = []
    for point in response.points:
        hits.append((point.id, point.score))
    return hits


def overlap(hits, exact_hits):
    """Fraction of the exact top-k that this method also found."""
    found = set()
    for doc_id, _ in hits:
        found.add(doc_id)
    truth = set()
    for doc_id, _ in exact_hits:
        truth.add(doc_id)
    return len(found & truth) / len(truth)


def describe(hits, documents):
    rows = []
    for rank, (doc_id, score) in enumerate(hits, start=1):
        rows.append(
            {
                "rank": rank,
                "id": doc_id,
                "score": round(score, 5),
                "category": documents[doc_id]["category"],
                "snippet": snippet(documents[doc_id]["text"]),
            }
        )
    return rows


def run_distance_metrics(client, documents, queries, query_vectors, verbose=False):
    """Part 2: the same query and vectors against cosine, euclid and dot."""
    records = []
    for index, (query, query_vector) in enumerate(zip(queries, query_vectors)):
        for collection, distance in METRIC_COLLECTIONS.items():
            hits, ms = timed(lambda: qdrant_search(client, collection, query_vector))
            records.append(
                {
                    "query_index": index,
                    "query": query,
                    "collection": collection,
                    "metric": distance.value,
                    "latency_ms": round(ms, 3),
                    "results": describe(hits, documents),
                }
            )
            if verbose:
                report(f"{distance.value:<8} ({collection})", hits, documents, ms)
    return records


def run_hnsw(client, documents, queries, query_vectors, verbose=False):
    """Part 3: exact ground truth, then both HNSW builds across several ef values."""
    records = []
    for index, (query, query_vector) in enumerate(zip(queries, query_vectors)):
        exact_hits, exact_ms = timed(
            lambda: qdrant_search(
                client, "news_cosine", query_vector, models.SearchParams(exact=True)
            )
        )
        records.append(
            {
                "query_index": index,
                "query": query,
                "config": EXACT,
                "collection": "news_cosine",
                "ef": None,
                "overlap": 1.0,
                "latency_ms": round(exact_ms, 3),
                "results": describe(exact_hits, documents),
            }
        )
        if verbose:
            report("exact (brute force)", exact_hits, documents, exact_ms)

        for collection, label in HNSW_COLLECTIONS.items():
            for ef in EF_VALUES:
                hits, ms = timed(
                    lambda: qdrant_search(
                        client, collection, query_vector, models.SearchParams(hnsw_ef=ef)
                    )
                )
                records.append(
                    {
                        "query_index": index,
                        "query": query,
                        "config": f"{label}-ef{ef}",
                        "collection": collection,
                        "ef": ef,
                        "overlap": overlap(hits, exact_hits),
                        "latency_ms": round(ms, 3),
                        "results": describe(hits, documents),
                    }
                )
                if verbose:
                    report(
                        f"{label} ef={ef}", hits, documents, ms, overlap(hits, exact_hits)
                    )
    return records


def run_ivf(index, documents, queries, query_vectors, exact_by_query, verbose=False):
    """Part 4: the hand-rolled IVF index at a low and a high nprobe."""
    records = []
    for qi, (query, query_vector) in enumerate(zip(queries, query_vectors)):
        for nprobe in NPROBE_VALUES:
            hits, ms = timed(
                lambda: index.search_ivf(query_vector, nprobe=nprobe, k=TOP_K)
            )
            scanned = len(index.probe(query_vector, nprobe))
            records.append(
                {
                    "query_index": qi,
                    "query": query,
                    "config": f"ivf-nprobe{nprobe}",
                    "nprobe": nprobe,
                    "vectors_scanned": scanned,
                    "overlap": overlap(hits, exact_by_query[qi]),
                    "latency_ms": round(ms, 3),
                    "results": describe(hits, documents),
                }
            )
            if verbose:
                report(
                    f"ivf nprobe={nprobe} (scanned {scanned})",
                    hits,
                    documents,
                    ms,
                    overlap(hits, exact_by_query[qi]),
                )
    return records


def report(label, hits, documents, ms, ov=None):
    """Print one method's top-5 for the live demo."""
    suffix = "" if ov is None else f"  overlap {ov:.0%}"
    print(f"\n  {label}   {ms:.2f} ms{suffix}")
    for rank, (doc_id, score) in enumerate(hits, start=1):
        category = documents[doc_id]["category"]
        print(f"    {rank}. [{score:8.4f}] {category:26} {snippet(documents[doc_id]['text'], 70)}")


def summarise(hnsw_records, ivf_records):
    """Part 5: average overlap and latency per configuration, across all queries."""
    buckets = {}
    for record in hnsw_records + ivf_records:
        buckets.setdefault(record["config"], []).append(record)

    rows = []
    for config, records in buckets.items():
        overlaps = []
        latencies = []
        for r in records:
            overlaps.append(r["overlap"])
            latencies.append(r["latency_ms"])
        rows.append(
            {
                "config": config,
                "mean_overlap": round(statistics.mean(overlaps), 3),
                "mean_latency_ms": round(statistics.mean(latencies), 3),
                "queries": len(records),
            }
        )
    rows.sort(key=lambda r: (-r["mean_overlap"], r["mean_latency_ms"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        type=int,
        metavar="N",
        help="run query N (0-4) through every method, printing as it goes",
    )
    args = parser.parse_args()

    documents, doc_vectors, queries, query_vectors = load_vectors()
    client = connect()
    print(f"building IVF index over {len(documents)} vectors...")
    index = build_ivf_index(doc_vectors)

    if args.demo is not None:
        queries = [queries[args.demo]]
        query_vectors = query_vectors[args.demo : args.demo + 1]
        print(f"\n=== {queries[0]!r} ===")

    verbose = args.demo is not None

    metric_records = run_distance_metrics(
        client, documents, queries, query_vectors, verbose
    )
    hnsw_records = run_hnsw(client, documents, queries, query_vectors, verbose)

    exact_by_query = {}
    for r in hnsw_records:
        if r["config"] != EXACT:
            continue
        exact_hits = []
        for hit in r["results"]:
            exact_hits.append((hit["id"], hit["score"]))
        exact_by_query[r["query_index"]] = exact_hits
    ivf_records = run_ivf(
        index, documents, queries, query_vectors, exact_by_query, verbose
    )

    if args.demo is not None:
        print()
        return

    RESULTS.mkdir(exist_ok=True)
    summary = summarise(hnsw_records, ivf_records)
    for name, payload in [
        ("distance_metrics.json", metric_records),
        ("hnsw.json", hnsw_records),
        ("ivf.json", ivf_records),
        ("summary.json", summary),
    ]:
        (RESULTS / name).write_text(json.dumps(payload, indent=2))
        print(f"wrote results/{name}")

    print(f"\n{'config':22} {'mean overlap':>13} {'mean latency':>14}")
    for row in summary:
        print(f"{row['config']:22} {row['mean_overlap']:12.0%} {row['mean_latency_ms']:12.2f} ms")


if __name__ == "__main__":
    main()
