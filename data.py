
import json
from collections import Counter
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DOCUMENTS = ARTIFACTS / "documents.json"

SAMPLE_SIZE = 6000
MIN_CHARS = 200
SEED = 42


def load_documents(sample_size=SAMPLE_SIZE, seed=SEED):
    bundle = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=seed,
    )

    documents = []
    for text, label in zip(bundle.data, bundle.target):
        text = text.strip()
        if len(text) < MIN_CHARS:
            continue
        documents.append(
            {
                "id": len(documents),
                "text": text,
                "category": bundle.target_names[label],
            }
        )
        if len(documents) == sample_size:
            break

    return documents


def main():
    documents = load_documents()
    ARTIFACTS.mkdir(exist_ok=True)
    DOCUMENTS.write_text(json.dumps(documents))

    categories = []
    for d in documents:
        categories.append(d["category"])
    counts = Counter(categories)
    print(f"kept {len(documents)} documents across {len(counts)} categories")
    for category, n in counts.most_common():
        print(f"  {n:5d}  {category}")
    print(f"wrote {DOCUMENTS}")


if __name__ == "__main__":
    main()
