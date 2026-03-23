"""
graph/dataset.py
================
Manages the semi-supervised label setup:
  - Masks most node labels as "unknown"
  - Computes graph statistics and the homophily index

Homophily Index H = fraction of edges whose endpoints share the same label.
A perfectly homophilic graph has H = 1.0.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def mask_labels(
    G: nx.Graph,
    node_labels: Dict[int, int],
    known_fraction: float = config.KNOWN_LABEL_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """
    Randomly reveals a fraction of node labels as "observed" (known).
    The rest become "unknown" — targets for Gibbs sampling propagation.

    Returns
    -------
    observed_labels : dict  {node_id -> label}  (subset of node_labels)
    known_nodes     : list  node ids with revealed labels
    unknown_nodes   : list  node ids to be inferred
    """
    rng = np.random.default_rng(seed)
    all_nodes = list(node_labels.keys())

    # Stratified sampling: reveal `known_fraction` per community
    known_nodes = []
    num_communities = max(node_labels.values()) + 1
    for comm in range(num_communities):
        comm_nodes = [n for n, l in node_labels.items() if l == comm]
        k = max(1, int(len(comm_nodes) * known_fraction))
        selected = rng.choice(comm_nodes, size=k, replace=False).tolist()
        known_nodes.extend(selected)

    unknown_nodes = [n for n in all_nodes if n not in known_nodes]

    # Update graph attributes
    observed_labels = {}
    for node in known_nodes:
        G.nodes[node]['observed'] = True
        observed_labels[node] = node_labels[node]
    for node in unknown_nodes:
        G.nodes[node]['observed'] = False

    print(f"\n[Dataset] Label Setup:")
    print(f"  Known   nodes : {len(known_nodes)}  ({100*known_fraction:.0f}%)")
    print(f"  Unknown nodes : {len(unknown_nodes)} ({100*(1-known_fraction):.0f}%)")

    return observed_labels, known_nodes, unknown_nodes


def compute_homophily_index(G: nx.Graph, labels: Dict[int, int]) -> float:
    """
    Computes the edge homophily index:
      H = |{(u,v) ∈ E : label[u] == label[v]}| / |E|
    """
    same = sum(1 for u, v in G.edges() if labels.get(u) == labels.get(v))
    total = G.number_of_edges()
    return same / total if total > 0 else 0.0


def graph_statistics(G: nx.Graph, node_labels: Dict[int, int]) -> Dict:
    """Computes and prints summary statistics for the graph."""
    degrees = [d for _, d in G.degree()]
    h_index  = compute_homophily_index(G, node_labels)

    stats = {
        "num_nodes"       : G.number_of_nodes(),
        "num_edges"       : G.number_of_edges(),
        "avg_degree"      : np.mean(degrees),
        "max_degree"      : max(degrees),
        "min_degree"      : min(degrees),
        "density"         : nx.density(G),
        "homophily_index" : h_index,
        "num_communities" : len(set(node_labels.values())),
        "connected"       : nx.is_connected(G),
    }

    print("\n[Dataset] Graph Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    return stats
