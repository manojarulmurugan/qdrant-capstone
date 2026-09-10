# Comparison and findings

## How the numbers were produced

6,000 posts sampled from 20 Newsgroups with headers, footers and quoted reply
text removed, and anything under 200 characters dropped. Every post was embedded
with `all-MiniLM-L6-v2` into 384 dimensions. Qdrant 1.19.1 runs in Docker with
four collections, all holding identical vectors and payload.

Latency is the median of five runs after one warm-up call, measured client side
with `time.perf_counter`. Overlap is the size of the intersection between a
method's top-5 and exact search's top-5, divided by 5.

The embedding model is loaded without its final `Normalize` module. Published as
`Transformer -> Pooling -> Normalize`, it returns unit-length vectors whatever
`encode(normalize_embeddings=...)` is set to. On unit vectors dot product is
exactly cosine, and Euclidean distance is a decreasing function of cosine, so all
three metrics in Part 2 would rank identically. Dropping that layer gives raw
mean-pooled vectors with norms from 1.014 to 4.075, mean 2.097.

## Two Qdrant defaults that had to be changed

Both of these silently produce results that look fine and mean nothing at this
dataset size. 6,000 x 384 float32 vectors come to 9,000 KB.

| Setting | Default | Effect at 9,000 KB | Set to |
|---|---|---|---|
| `optimizers_config.indexing_threshold` | 20,000 KB | no HNSW graph is ever built, every search is brute force | 1 |
| `hnsw_config.full_scan_threshold` | 10,000 KB | graph is built but the query planner never walks it | 10 |

The second one was found the hard way. With the graph built and
`indexed_vectors_count` reporting 6,000, every HNSW configuration still returned
100% overlap against exact search, including the deliberately crippled one at
`ef=4`. A recall check over 50 held-out queries also came back at exactly 100.0%,
which is not a plausible result for an approximate index. The cause was that
Qdrant only walks the graph once a segment exceeds `full_scan_threshold`, and
below that a scan is genuinely faster, so it scans. The numbers below were taken
after lowering it to 10 KB, the minimum Qdrant accepts.

`default_segment_number` is pinned to 1 so the vectors are not spread across one
indexed segment per CPU core, which would give each collection several
independent HNSW graphs and make the results depend on the machine. Qdrant still
reports `segments_count` of 2, because it keeps a small appendable segment for
incoming writes alongside the indexed one, and `indexed_vectors_count` of 6,000
confirms every vector made it into the graph. With that pinned, `m` and
`ef_construct` are the only difference between the two cosine collections.

## Part 2: distance metrics

Same vectors, same payload, same index. Only the metric differs. Top-5 overlap
between each pair of metrics:

| Query | cos vs dot | cos vs euclid | dot vs euclid | distinct top-1s |
|---|---|---|---|---|
| 0. graphics card driver | 1/5 | 4/5 | 2/5 | 1 |
| 1. Bible and resurrection | 3/5 | 5/5 | 3/5 | 3 |
| 2. used motorcycle for sale | 2/5 | 4/5 | 1/5 | 2 |
| 3. clipper chip | **0/5** | 3/5 | 2/5 | 3 |
| 4. chronic back pain | 2/5 | 4/5 | 1/5 | 2 |

### The case where ranking changes: query 3

"government encryption policy and the clipper chip". Cosine and dot product
return **no documents in common at all**, yet every hit under both metrics is
from `sci.crypt`, so both are finding relevant material. They disagree about
which relevant material ranks highest.

| id | norm | cosine | dot | euclid | chars | position |
|---|---|---|---|---|---|---|
| 1603 | 1.789 | 0.7612 | 9.1264 | 5.4660 | 5,786 | cosine #1 |
| 557 | 1.862 | 0.7505 | 9.3667 | 5.4464 | 5,937 | cosine #2 |
| 3538 | 3.459 | 0.5197 | **12.0502** | 5.7267 | 204 | dot #1 |
| 2058 | 2.265 | 0.6771 | 10.2797 | **5.4315** | 401 | euclid #1 |
| 2782 | 3.033 | 0.4742 | 9.6410 | 5.9032 | 201 | dot #5 |

Document 3538 wins on dot product with a score of 12.05 while ranking nowhere
near the top on cosine, where its 0.5197 is well behind 1603's 0.7612. The reason
is entirely magnitude. Its norm is 3.459 against 1603's 1.789.

The direction of that effect is the opposite of what I first expected. The long
document has the *shorter* vector. Mean pooling averages the token embeddings, so
a short post about one thing has few token vectors all pointing the same way and
they reinforce each other, while a 5,786 character post ranges over many
subtopics whose token vectors partially cancel and pull the mean back toward the
origin. Vector norm here is a proxy for topical concentration, and length dilutes
it. Cosine top-5 averages norm 1.820 at 1,267 to 5,937 characters; dot top-5
averages 2.703 at 201 to 746 characters.

So on this query:

- **Cosine** measures angle alone and is blind to magnitude. It ranks first a
  long, detailed document that explains how the Clipper Chip works.
- **Dot product** is `|a||b|cos` and rewards magnitude. It ranks first a
  204-character question asking where to buy encryption chips, which is far more
  concentrated but far less informative.
- **Euclid** lands between the two, and provably so. Expanding
  `||v-q||^2 = ||v||^2 + ||q||^2 - 2(v.q)` and noting `||q||^2` is constant for a
  fixed query, minimising Euclidean distance means minimising
  `||v||^2 - 2(v.q)`. It chases a high dot product like dot does, but pays a
  penalty for large `||v||`. Document 2058 scores -15.43 on that expression
  against 3538's -12.14, which is why Euclid puts 2058 first.

## Part 3: HNSW against exact search

`news_cosine` uses Qdrant defaults, `m=16` and `ef_construct=100`.
`news_cosine_weak` holds identical vectors with `m=4` and `ef_construct=8`.

| Config | q0 | q1 | q2 | q3 | q4 | Mean overlap | Mean latency |
|---|---|---|---|---|---|---|---|
| exact (`exact: true`) | 100% | 100% | 100% | 100% | 100% | 100% | 13.17 ms |
| default, ef=16 | 100% | 100% | 80% | 100% | 100% | 96% | 10.58 ms |
| default, ef=64 | 100% | 100% | 100% | 100% | 100% | 100% | 9.31 ms |
| default, ef=128 | 100% | 100% | 100% | 100% | 100% | 100% | 10.81 ms |
| weak, ef=16 | 80% | 80% | 80% | 100% | 20% | 72% | 9.24 ms |
| weak, ef=64 | 80% | 80% | 100% | 100% | 20% | 76% | 9.04 ms |
| weak, ef=128 | 100% | 80% | 100% | 100% | 20% | 80% | 8.75 ms |

The default build reaches exact agreement by `ef=64` and stays there. The weak
build climbs from 72% to 80% across the same range and never closes the gap.

Query 4 is the clearest single result in the project. Under the weak build it
sits at 20% overlap at `ef=16`, at `ef=64` and at `ef=128`, completely flat.
Search-time `ef` controls how many candidates the greedy walk keeps while
traversing the graph. It cannot add edges that were never built. With `m=4` and
`ef_construct=8`, the region of the graph holding those documents has no path
into it from where the search starts, so widening the search changes nothing.

## Part 4: the IVF index

48 k-means clusters over the same 6,000 vectors, built in about 3.6 seconds.
Cluster sizes run from 61 to 252, median 120.

| nprobe | q0 | q1 | q2 | q3 | q4 | Mean overlap | Vectors scanned | Mean latency |
|---|---|---|---|---|---|---|---|---|
| 1 | 100% | 60% | 80% | 80% | 100% | 84% | 69 to 252 | 0.15 ms |
| 8 | 100% | 100% | 100% | 100% | 100% | 100% | 950 to 1,311 | 0.43 ms |

Vectors are L2-normalised before k-means. k-means minimises squared Euclidean
distance while the search scores by cosine, and on unit vectors the two produce
the same ordering, so normalising first keeps the partitioning consistent with
the scoring. This is spherical k-means. Centroids are renormalised after fitting
because the mean of a set of unit vectors is not itself unit length.

## Part 5: everything together

| Method | Mean overlap vs exact | Mean latency | Work done per query |
|---|---|---|---|
| exact brute force | 100% | 13.17 ms | all 6,000 vectors |
| HNSW default, ef=16 | 96% | 10.58 ms | graph walk, 16 candidates |
| HNSW default, ef=64 | 100% | 9.31 ms | graph walk, 64 candidates |
| HNSW default, ef=128 | 100% | 10.81 ms | graph walk, 128 candidates |
| HNSW weak, ef=16 | 72% | 9.24 ms | graph walk, 16 candidates |
| HNSW weak, ef=64 | 76% | 9.04 ms | graph walk, 64 candidates |
| HNSW weak, ef=128 | 80% | 8.75 ms | graph walk, 128 candidates |
| IVF, nprobe=1 | 84% | 0.15 ms | about 150 vectors |
| IVF, nprobe=8 | 100% | 0.43 ms | about 1,100 vectors |

**Latency here is not a fair comparison and should not be read as one.** Every
Qdrant number includes an HTTP round trip to localhost, which at this dataset
size costs more than the search itself. That is why exact search over 6,000
vectors "costs" 13 ms and why the HNSW rows barely separate from each other. The
IVF numbers are in-process NumPy with no serialisation at all. The overlap column
is directly comparable; the latency column is only comparable within a row group.

### Which method closed the gap, and why

Both did, and both for the same reason, but only one of them could.

IVF went from 84% to 100% as `nprobe` moved from 1 to 8, scanning roughly 150
vectors instead of 1,100. At `nprobe=1` only the single nearest cluster is
searched and everything else is invisible no matter how similar it really is.
Query 1 shows this at 60%: two of the true top-5 sit in neighbouring clusters,
and since documents near a cluster boundary can be closer to the query than
documents at the centre of the chosen cluster, they are lost. Raising `nprobe`
widens the shortlist, boundary cases come back, and at `nprobe=8` nothing in the
true top-5 is missed. Push `nprobe` to 48 and IVF becomes brute force.

The default HNSW collection behaved the same way, 96% at `ef=16` and 100% from
`ef=64`. The weak collection did not. It improved only from 72% to 80% and
plateaued, because `ef` and `nprobe` are search-time dials while `m` and
`ef_construct` are build-time ones. `nprobe` and `ef` decide how much of an
existing structure to explore. `m` decides what structure exists. Spending more
search effort on a graph whose edges were never built returns almost nothing,
which is the whole reason the weak collection stays stuck.

That is the speed and accuracy tradeoff in its usual form: every method starts
cheap and approximate, and every method converges on exact search as you let it
examine more candidates, right up to the point where it is examining everything
and has become brute force. What the weak collection adds is that the tradeoff
only applies within the quality of index you actually built.

### Metric choice and index choice are separate decisions

Query 3 makes the point. Cosine and dot returned top-5 lists with zero documents
in common, from identical vectors, identical payload and identical HNSW
configuration. Nothing about the index changed, only the metric.

The mirror image is in Part 3. `news_cosine` and `news_cosine_weak` hold the same
vectors under the same cosine metric and disagree on 28% of results at `ef=16`,
because only the index changed.

The metric decides what "nearest" means. The index decides how hard you look for
it. A well-tuned HNSW graph over the wrong metric will return the wrong documents
quickly, and no amount of `ef` will help, because it is faithfully approximating
an answer to the wrong question.

### When would a real system want IVF instead?

Qdrant ships only HNSW, and at this scale that is clearly the right call. Two
things from Part 4 point at where IVF wins.

The first is build cost and memory. IVF stores 48 centroids plus a list of ids,
and building it took about 3.6 seconds. HNSW stores `m` edges per node forever
and has to be rebuilt to change them. At billion scale that graph does not fit in
memory, whereas IVF postings live on disk and only the centroids need to be
resident.

The second is control. `nprobe` is a per-query dial with a directly predictable
cost, since scanning 8 clusters costs about eight times scanning 1. That makes
IVF straightforward to tune against a latency budget, and it means one index can
serve both cheap and expensive queries. Changing HNSW's real quality means
changing `m`, which means reindexing.

IVF also updates more cheaply. Adding a vector means appending to one posting
list, against HNSW's graph surgery. The catch is that centroids drift as the data
changes, so IVF needs periodic re-clustering that HNSW does not.
