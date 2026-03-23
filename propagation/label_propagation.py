"""
propagation/label_propagation.py
=================================
Homophily-based Label Propagation (LP) — a classic semi-supervised method.

Algorithm
---------
Each node's label distribution is iteratively updated as a weighted
average of its neighbors' distributions, scaled by a damping factor α.

Y(t+1) = α · A_norm · Y(t) + (1-α) · Y_0

where:
  A_norm  = row-normalised adjacency matrix
  Y_0     = initial label matrix (one-hot for known nodes, uniform for unknown)
  α       = damping factor (how much signal is propagated vs. clamped)

Homophily weighting:
  In a highly homophilic graph, label propagation is very effective because
  in-community edges dominate the neighbourhood signal.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LabelPropagation:
    """
    Homophily-based Label Propagation baseline.

    Parameters
    ----------
    G               : NetworkX graph
    observed_labels : dict {node -> label}
    unknown_nodes   : list of nodes to infer
    num_labels      : number of classes
    alpha           : damping / propagation strength (0 < α < 1)
    iterations      : max iterations
    """

    def __init__(
        self,
        G: nx.Graph,
        observed_labels: Dict[int, int],
        unknown_nodes: List[int],
        num_labels: int  = config.NUM_COMMUNITIES,
        alpha: float     = config.LP_ALPHA,
        iterations: int  = config.LP_ITERATIONS,
    ):
        self.G               = G
        self.observed_labels = observed_labels
        self.unknown_nodes   = set(unknown_nodes)
        self.num_labels      = num_labels
        self.alpha           = alpha
        self.iterations      = iterations

        self.nodes           = sorted(G.nodes())
        self.n               = len(self.nodes)
        self.node_to_idx     = {node: i for i, node in enumerate(self.nodes)}

        # Build row-normalised adjacency matrix
        self.A_norm = self._build_normalised_adjacency()

        # Build initial label matrix Y0
        self.Y0, self.Y = self._init_label_matrix()

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> Dict[int, int]:
        """
        Runs label propagation and returns predicted labels for unknown nodes.
        """
        print(f"\n[LabelPropagation] Running {self.iterations} iterations "
              f"(α={self.alpha}) …")

        for _ in tqdm(range(self.iterations), desc="Label Propagation"):
            Y_new = self.alpha * (self.A_norm @ self.Y) + (1 - self.alpha) * self.Y0
            # Clamp observed nodes back to one-hot (known labels don't change)
            for node, label in self.observed_labels.items():
                idx = self.node_to_idx[node]
                Y_new[idx] = self.Y0[idx]
            delta = np.max(np.abs(Y_new - self.Y))
            self.Y = Y_new
            if delta < 1e-6:
                print(f"[LabelPropagation] Converged early.")
                break

        # Extract hard labels for unknown nodes
        predicted = {}
        for node in self.unknown_nodes:
            idx = self.node_to_idx[node]
            predicted[node] = int(np.argmax(self.Y[idx]))

        print("[LabelPropagation] Done.")
        return predicted

    def get_soft_labels(self, node: int) -> np.ndarray:
        """Returns the soft label probability distribution for `node`."""
        idx = self.node_to_idx[node]
        return self.Y[idx].copy()

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _build_normalised_adjacency(self) -> np.ndarray:
        """Builds D^(-1) · A, the row-normalised adjacency matrix."""
        A = nx.to_numpy_array(self.G, nodelist=self.nodes)
        degree = A.sum(axis=1, keepdims=True)
        degree[degree == 0] = 1   # avoid division by zero for isolated nodes
        return A / degree

    def _init_label_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialises Y0: one-hot for observed nodes, uniform for unknown.
        Returns (Y0, Y) where Y is the working copy.
        """
        Y0 = np.full((self.n, self.num_labels), 1.0 / self.num_labels)
        for node, label in self.observed_labels.items():
            idx = self.node_to_idx[node]
            Y0[idx] = 0.0
            Y0[idx, label] = 1.0
        return Y0, Y0.copy()
