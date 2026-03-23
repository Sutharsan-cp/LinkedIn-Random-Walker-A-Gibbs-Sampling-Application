"""
sampling/random_walker.py
=========================
Implements random walks on the LinkedIn graph.

Purpose in this project:
  - Explore node neighborhoods via stochastic traversal
  - Compute visit-frequency statistics that enrich Gibbs conditionals
  - Demonstrate the 'Random Walker' aspect of the project title

Two walk strategies:
  1. Simple Random Walk   – uniform transition over neighbors
  2. Biased Random Walk   – preference toward same-label neighbors (homophilic)
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RandomWalker:
    """
    Performs random walks on a graph and accumulates neighborhood statistics
    used to inform Gibbs Sampling conditionals.
    """

    def __init__(
        self,
        G: nx.Graph,
        walk_length: int = config.WALK_LENGTH,
        num_walks: int   = config.NUM_WALKS,
        seed: int        = config.RANDOM_SEED,
    ):
        self.G           = G
        self.walk_length = walk_length
        self.num_walks   = num_walks
        self.rng         = np.random.default_rng(seed)

        # visit_counts[node][neighbor] = #times neighbor visited from node
        self.visit_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # label_visit_counts[node][label] = #times a node of <label> was visited from node
        self.label_visit_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # ── Public API ──────────────────────────────────────────────────────────

    def run_walks(self, observed_labels: Dict[int, int]) -> None:
        """
        Runs `num_walks` walks of length `walk_length` from every node.
        Accumulates visit counts for use by the Gibbs sampler.
        """
        nodes = list(self.G.nodes())
        total = len(nodes) * self.num_walks
        print(f"\n[RandomWalker] Running {self.num_walks} walks × {len(nodes)} nodes"
              f" (length={self.walk_length}) …")

        done = 0
        for start_node in nodes:
            for _ in range(self.num_walks):
                walk = self._simple_walk(start_node)
                # Record which labels were encountered (skip start node itself)
                for visited in walk[1:]:
                    self.visit_counts[start_node][visited] += 1
                    if visited in observed_labels:
                        lbl = observed_labels[visited]
                        self.label_visit_counts[start_node][lbl] += 1
            done += self.num_walks

        print(f"[RandomWalker] Done. Total walks: {done}")

    def get_label_visit_probs(
        self,
        node: int,
        num_labels: int,
        smoothing: float = config.SMOOTHING,
    ) -> np.ndarray:
        """
        Returns a probability distribution over labels for `node`
        based on which label-bearing nodes were encountered during walks.
        Uses Laplace smoothing to avoid zero probabilities.
        """
        counts = np.array(
            [self.label_visit_counts[node].get(l, 0) for l in range(num_labels)],
            dtype=float,
        )
        counts += smoothing          # Laplace smoothing
        return counts / counts.sum()

    # ── Walk Implementations ─────────────────────────────────────────────────

    def _simple_walk(self, start: int) -> List[int]:
        """Uniform random walk: at each step, move to a uniformly random neighbor."""
        path = [start]
        current = start
        for _ in range(self.walk_length - 1):
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break
            current = int(self.rng.choice(neighbors))
            path.append(current)
        return path

    def sample_walk_sequence(self, start: int) -> List[int]:
        """Public helper: returns a single walk sequence starting from `start`."""
        return self._simple_walk(start)

    def most_visited_neighbors(self, node: int, top_k: int = 5) -> List[Tuple[int, int]]:
        """Returns top-k most visited neighbors from `node` during walks."""
        counts = self.visit_counts[node]
        return sorted(counts.items(), key=lambda x: -x[1])[:top_k]
