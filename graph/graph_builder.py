"""
graph/graph_builder.py
======================
Builds a synthetic LinkedIn-like social network using the
Stochastic Block Model (SBM) — the canonical model for homophilic graphs.

Key idea:
  - Nodes are split into k professional communities.
  - P(edge | same community) = p_in   >> P(edge | diff community) = p_out
  - This enforces *homophily*: professionals tend to connect with peers.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_linkedin_graph(
    num_nodes: int = config.NUM_NODES,
    num_communities: int = config.NUM_COMMUNITIES,
    p_in: float = config.P_IN,
    p_out: float = config.P_OUT,
    community_labels: List[str] = config.COMMUNITY_LABELS,
    seed: int = config.RANDOM_SEED,
) -> Tuple[nx.Graph, dict]:
    """
    Build a Stochastic Block Model graph representing a LinkedIn network.

    Returns
    -------
    G : nx.Graph
        Graph with node attributes:
          - 'true_label'   : int   community index
          - 'label_name'   : str   human-readable profession
          - 'observed'     : bool  whether label is revealed (set later)
    node_labels : dict  {node_id -> true label index}
    """
    rng = np.random.default_rng(seed)

    # ── Assign nodes to communities uniformly ────────────────────────────────
    community_sizes = _split_nodes(num_nodes, num_communities, rng)

    # ── Build SBM probability matrix ─────────────────────────────────────────
    # p_matrix[i][j] = probability of edge between community i and j
    p_matrix = []
    for i in range(num_communities):
        row = []
        for j in range(num_communities):
            row.append(p_in if i == j else p_out)
        p_matrix.append(row)

    # ── Generate graph ────────────────────────────────────────────────────────
    np.random.seed(seed)  # networkx SBM uses numpy's global random state
    G = nx.stochastic_block_model(community_sizes, p_matrix, seed=seed)

    # ── Attach metadata ───────────────────────────────────────────────────────
    node_labels = {}
    node_id = 0
    for comm_idx, size in enumerate(community_sizes):
        label_name = community_labels[comm_idx % len(community_labels)]
        for _ in range(size):
            G.nodes[node_id]['true_label']  = comm_idx
            G.nodes[node_id]['label_name']  = label_name
            G.nodes[node_id]['observed']    = False   # will be set by dataset.py
            node_labels[node_id] = comm_idx
            node_id += 1

    print(f"[GraphBuilder] Created LinkedIn graph:")
    print(f"  Nodes      : {G.number_of_nodes()}")
    print(f"  Edges      : {G.number_of_edges()}")
    print(f"  Communities: {num_communities} {community_labels[:num_communities]}")
    print(f"  p_in={p_in}, p_out={p_out}")

    return G, node_labels


def _split_nodes(num_nodes: int, num_communities: int, rng) -> List[int]:
    """Splits num_nodes as evenly as possible into num_communities groups."""
    base = num_nodes // num_communities
    remainder = num_nodes % num_communities
    sizes = [base + (1 if i < remainder else 0) for i in range(num_communities)]
    return sizes


def get_community_partition(G: nx.Graph) -> dict:
    """Returns {community_idx -> [node_ids]} mapping from true labels."""
    partition = {}
    for node in G.nodes():
        comm = G.nodes[node]['true_label']
        partition.setdefault(comm, []).append(node)
    return partition
