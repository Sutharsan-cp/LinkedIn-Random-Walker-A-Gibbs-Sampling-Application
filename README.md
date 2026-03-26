# LinkedIn Random Walker: A Gibbs Sampling Application

> **Homophily-Based Label Propagation via Gibbs Sampling on a Graph Dataset**

An AI project that models a LinkedIn-like social network as a graph and uses **Gibbs Sampling (MCMC)** to infer unknown node labels (professional communities)The Gibbs Sampler correctly guessed the profession of **207 out of 240 hidden people**, just from looking at who they're connected to!

---

## Demo Results

| Metric | Value |
|---|---|
| Graph | 300 nodes · 3,363 edges |
| Homophily Index | 0.78 |
| **Gibbs Sampler Accuracy** | **86.25%** |
| Label Propagation Accuracy | 79.58% |
| Macro F1 (Gibbs) | 0.86 |

---

## Concept

```
LinkedIn Graph  →  Hide 80% labels  →  Gibbs Sampling  →  Predict labels
```

1. **Graph** — 300 users split into 5 professional communities (nodes = users, edges = connections)
2. **Homophily** — intra-community edge probability (30%) >> inter-community (2%), mimicking real LinkedIn
3. **Semi-supervised setup** — only 20% of labels are revealed; the rest are inferred
4. **Gibbs Sampler** — iteratively samples each unknown node's label conditioned on its neighbors:
   `P(label_i | neighbors) ∝ count(neighbors with label_i) + smoothing`
5. **Random Walker** — enriches neighborhood statistics via graph traversal before sampling
6. **Label Propagation** — simpler baseline using row-normalised adjacency matrix

---

## Project Structure

```
AI_PACKAGE/
├── main.py                          # Entry point — runs full 9-step pipeline
├── config.py                        # All tunable parameters
├── requirements.txt
├── graph/
│   ├── graph_builder.py             # Stochastic Block Model graph (homophilic)
│   └── dataset.py                   # Label masking + homophily index
├── sampling/
│   ├── random_walker.py             # Random walk + visit-frequency statistics
│   └── gibbs_sampler.py             # Core Gibbs MCMC sampler
├── propagation/
│   └── label_propagation.py         # LP baseline (α·A_norm·Y + (1-α)·Y0)
├── utils/
│   ├── metrics.py                   # Accuracy, F1, confusion matrix, convergence
│   └── visualizer.py                # 8 dark-themed matplotlib/seaborn plots
├── outputs/                         # Auto-generated PNG visualizations
└── EXPLANATION.md                   # Plain-English project explanation
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `networkx`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `tqdm`, `pandas`, `seaborn`

---

## Usage

```bash
python main.py
```

The pipeline runs all 9 steps and saves 8 plots to `outputs/`:

| Output File | Description |
|---|---|
| `graph_true.png` | True community labels |
| `graph_gibbs.png` | Gibbs-predicted labels |
| `graph_lp.png` | Label Propagation labels |
| `convergence.png` | Label-change rate per Gibbs iteration |
| `accuracy_comparison.png` | Gibbs vs LP accuracy bar chart |
| `degree_distribution.png` | Node degree histogram |
| `confusion_matrix.png` | Per-class confusion heatmap |
| `posterior_distributions.png` | Posterior label distributions per node |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `NUM_NODES` | 300 | Number of users in the graph |
| `NUM_COMMUNITIES` | 5 | Number of professional communities |
| `P_IN` | 0.30 | Intra-community edge probability |
| `P_OUT` | 0.02 | Inter-community edge probability |
| `KNOWN_LABEL_FRACTION` | 0.20 | Fraction of revealed labels |
| `GIBBS_ITERATIONS` | 200 | Gibbs sampling iterations |
| `BURN_IN` | 50 | Burn-in period (discarded samples) |
| `LP_ALPHA` | 0.85 | Label propagation damping factor |

---

## Algorithm — Gibbs Sampling

1. **Bootstrapping**: Initialise unknown nodes by sampling from a distribution based *only* on observed neighbors.
2. **Burn-in Phase**: Run $B$ iterations where labels are updated but samples are not yet collected.
3. **Sampling Phase**: Run $S$ iterations, updating labels and incrementing frequency counts for each node.
4. **Final Assignment**: Assign the label with the highest frequency (mode) after the sampling phase.

---

## Per-Class Results (Gibbs Sampler)

| Community | Precision | Recall | F1 |
|---|---|---|---|
| Software Engineer | 0.88 | 0.96 | 0.92 |
| Data Scientist | 0.81 | 0.88 | 0.84 |
| Product Manager | 0.94 | 0.96 | 0.95 |
| UX Designer | 1.00 | 0.83 | 0.91 |
| Business Analyst | 0.94 | 0.92 | 0.93 |

---

## Course

**Semester 6 — Artificial Intelligence**
