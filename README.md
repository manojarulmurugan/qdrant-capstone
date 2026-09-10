# Qdrant search algorithm comparison

Semantic search over 6,000 posts from the 20 Newsgroups corpus, run through three distance metrics, two HNSW configurations at three search-time `ef` values, and a hand-written IVF index at two `nprobe` values, so the accuracy and latency tradeoffs can be measured rather than assumed.

Findings are in [comparison.md](comparison.md); the raw numbers are in `results/`.

## Start Qdrant

```bash
docker compose up -d
```

That publishes the REST API and dashboard on 6333 and gRPC on 6334, and mounts `./qdrant_storage` so collections survive a restart. Confirm it is up at <http://localhost:6333/dashboard>.

The equivalent single command, without Compose:

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.19.1
```

## Set up Python

Python 3.11 specifically: this was built on an Intel Mac, where the last PyTorch release with x86_64 wheels is 2.2.2, which supports up to 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python data.py          # sample 6,000 posts -> artifacts/documents.json
python embed.py         # embed docs and queries -> artifacts/*.npy
python qdrant_setup.py  # create and fill the four collections
python compare.py       # run every method, write results/*.json
```

`data.py` and `embed.py` only need to run once; their output is cached in `artifacts/`, which is gitignored because it is reproducible from the scripts.

For the live demo, run every method against a single query and print as it goes:

```bash
python compare.py --demo 0
```

`ivf.py` is also runnable on its own, which prints the cluster size distribution and the effect of `nprobe` without touching Qdrant:

```bash
python ivf.py
```

## Layout

| File | Role |
|------|------|
| `data.py` | samples and caches the corpus |
| `embed.py` | embeds documents and the five queries with `all-MiniLM-L6-v2` |
| `qdrant_setup.py` | creates the cosine, euclid, dot and under-tuned collections |
| `ivf.py` | k-means IVF index with `build_ivf_index` and `search_ivf` |
| `compare.py` | runs every method, writes `results/` |
| `queries.md` | the five test queries and the categories they target |
| `comparison.md` | the written comparison |

## Verifying Qdrant is up

```
$ curl -s http://localhost:6333/
{"title":"qdrant - vector search engine","version":"1.19.1", ...}

$ curl -s -o /dev/null -w "%{http_code}\n" http://localhost:6333/dashboard
200
```

The dashboard at <http://localhost:6333/dashboard> was confirmed loading and listing all four collections, each GREEN with 6,000 points and 384-dimensional vectors: `news_cosine` (Cosine), `news_cosine_weak` (Cosine), `news_dot` (Dot) and `news_euclid` (Euclid).

After `python qdrant_setup.py`, the same can be checked from the command line:

```bash
curl -s http://localhost:6333/collections | python3 -m json.tool
```
