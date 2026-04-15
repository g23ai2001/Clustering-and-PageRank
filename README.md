# Clustering and PageRank

> **Course Assignment** | MLBD


---

## Overview

This repository contains a complete implementation of three classical algorithms in data mining and distributed computing, executed in a **PySpark** environment inside a Jupyter Notebook.

| Part | Topic | Algorithm / Technique |
|------|-------|----------------------|
| 1 | Clustering | Farthest-First Traversal (k-center) + k-means++ |
| 2 | Web Search | Inverted Index with TF-IDF scoring |
| 3 | PageRank | Iterative PageRank on PySpark RDDs |

---

## Repository Structure

```
.
├── Clustering_and_PageRank_Solution.ipynb   # Main notebook (all 3 parts)
├── datasets/
│   ├── Q1-UCI Spam clustering/
│   │   └── spambase.data                   # 4601 × 58 spam feature vectors
│   ├── Q2-webSearch/
│   │   ├── webpages/                        # HTML/text webpage files
│   │   ├── actions.txt                      # Search engine actions
│   │   └── answers.txt                      # Expected query answers
│   └── Q3-PageRank/
│       ├── small.txt                        # Small graph (53 nodes)
│       └── whole.txt                        # Full graph (1000 nodes)
└── README.md
```

---

## Part 1 — Clustering

### Algorithms

**`readVectorsSeq(filename)`**  
Reads a comma-separated feature file and returns a Python list of `DenseVector` objects.

**`kcenter(P, k)` — Farthest-First Traversal**  
- Picks initial centre deterministically (first point)  
- Maintains `min_dist[i]` = minimum squared distance from point `i` to any chosen centre  
- Greedily picks the farthest point as the next centre  
- Time complexity: **O(|P| × k)**

**`kmeansPP(P, k)` — k-means++ Seeding**  
- Picks first centre uniformly at random  
- Subsequent centres sampled with probability proportional to D² (squared distance to nearest existing centre)  
- Time complexity: **O(|P| × k)**

**`kmeansObj(P, C)`**  
Returns the average squared distance from each point to its nearest centre (the k-means objective divided by |P|).

### Experiments (Dataset: UCI Spambase — 4601 points × 58 dimensions)

| Experiment | Description | Output |
|-----------|-------------|--------|
| 1 | `kcenter(P, k=5)` | Running time ~0.15 s |
| 2 | `kmeansPP(P, k=5)` → `kmeansObj` | Objective ≈ 123,958 |
| 3 | `kcenter(P, k1=20)` → `kmeansPP(X, k=5)` → `kmeansObj(P, C)` | Coreset objective ≈ 2,381,206 |

> **Coreset note:** The k1-centre coreset pre-selects diverse candidate points. A larger `k1` leads to a smaller kmeansObj for the final centres.

### References
- [Farthest-First Traversal — Wikiwand](http://www.wikiwand.com/en/Farthest-first_traversal)
- [k-Means++ Paper — Arthur & Vassilvitskii, SODA 2007](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)

---

## Part 2 — Web Search (Inverted Index)

### Classes Implemented

| Class | Responsibility |
|-------|---------------|
| `MySet` | Custom set (list-backed) with `addElement`, `union`, `intersection` |
| `Position` | Stores `(PageEntry, wordIndex)` — 1-indexed over ALL tokens |
| `WordEntry` | All `Position` entries for a single word; computes TF |
| `PageIndex` | Per-document index: word → `WordEntry` |
| `PageEntry` | Reads a webpage file, builds its `PageIndex` |
| `MyHashTable` | Polynomial rolling hash, separate chaining; maps word → `WordEntry` |
| `InvertedPageIndex` | Global inverted index backed by `MyHashTable` |
| `SearchEngine` | Dispatches `addPage` / query actions |

### Tokenisation Rules

- Convert to **lowercase**
- Replace `{}[]<>=(). ,;'"?#!-:` with a space
- **Stop words** (counted for positions, not stored):  
  `a, an, the, they, these, this, for, is, are, was, of, or, and, does, will, whose`
- Word positions are **1-indexed** over ALL tokens (including stop words)
- Plural normalisation: `stacks → stack`, `structures → structure`, `applications → application`

### Supported Actions

```
addPage <filename>
queryFindPagesWhichContainWord <word>
queryFindPositionsOfWordInAPage <word> <page>
```

### Results

All **11/11 tests passed** against `answers.txt`.

---

## Part 3 — PageRank on PySpark RDD

### Algorithm

$$r^{(0)} = \frac{1}{n} \mathbf{1}$$

$$r^{(i)}_j = \frac{1-\beta}{n} + \beta \sum_{i \to j} \frac{r^{(i-1)}_i}{\deg(i)}$$

**Parameters:** β = 0.8, 40 iterations, r₀ = 1/n for all nodes  
**Duplicate edges** between the same pair are deduplicated (treated as one edge).

### Implementation Details

1. Load edge list and deduplicate via `.distinct()`
2. Build adjacency list with outgoing weights `1 / out_degree`
3. Initialise ranks = `1/n` for all nodes
4. Each iteration: `adj.join(ranks)` → flatMap contributions → `reduceByKey` sum → apply teleport
5. `leftOuterJoin` ensures nodes with zero in-links still receive the teleport contribution
6. DAG lineage is checkpointed every 10 iterations via `.collect()` + `sc.parallelize()`

### Results

**small.txt (53 nodes validation):**

| Rank | Node | Score |
|------|------|-------|
| 1 | 53 | 0.035731 |
| 2 | 14 | 0.034171 |
| 3 | 40 | 0.033630 |
| 4 | 1 | 0.030006 |
| 5 | 27 | 0.029720 |

> Top score ≈ **0.036** ✅ (matches expected)

**whole.txt (1000 nodes — Top 5):**

| Rank | Node | Score |
|------|------|-------|
| 1 | 263 | 0.00202029 |
| 2 | 537 | 0.00194334 |
| 3 | 965 | 0.00192545 |
| 4 | 243 | 0.00185263 |
| 5 | 285 | 0.00182737 |

Mean rank = 0.001 = 1/1000 ✅ (scores correctly normalised)

---

## Setup & Running

### Prerequisites

```bash
# Python dependencies
pip install pyspark numpy findspark

# Verify Java is installed (required by Spark)
java -version
```

### Run the Notebook

```bash
jupyter notebook Clustering_and_PageRank_Solution.ipynb
```

Run all cells in order. The notebook initialises a local `SparkSession` automatically.

### Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PySpark | 3.5.8 |
| NumPy | ≥ 1.24 |
| Java | 11 or 17 (LTS) |

---

## Assumptions

- The `datasets/` folder is located in the **same directory** as the notebook.
- Plural normalisation is treated as **exhaustive** — only the three pairs specified (`stacks/stack`, `structures/structure`, `applications/application`) are handled.
- Stop words list is treated as **exhaustive** per assignment specification.
- For PageRank, multiple directed edges between the same pair are collapsed to **one edge** before computing out-degrees.

---

## License

This code is submitted as coursework. All rights reserved by the author.
