"""
sampling/gibbs_sampler.py
=========================
Core Gibbs Sampling (MCMC) engine for label inference on the LinkedIn graph.

Algorithm
---------
Given:
  - Graph G = (V, E)
  - Observed labels: Y_obs  (subset of nodes)
  - Unknown:         Y_unk  (remaining nodes to infer)

Gibbs Sampling iterates:
  For t in 1..T:
    For each unknown node i (in random order):
      Sample  label_i ~ P(label_i | labels of neighbors of i)

Conditional distribution (homophily prior):
  P(l | N(i)) ∝ count(neighbors with label l) + smoothing

After burn-in, the mode (most frequent sample) is taken as final label.

Walk-enhanced variant:
  Combines immediate neighbor counts with random-walk visit statistics
  for a richer neighborhood signal.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sampling.random_walker import RandomWalker


class GibbsSampler:
    """
    Gibbs Sampler for label propagation on a homophilic graph.

    Parameters
    ----------
    G                : NetworkX graph with 'true_label' attributes
    observed_labels  : dict {node -> label} for seed/known nodes
    unknown_nodes    : list of nodes to infer
    num_labels       : number of distinct community labels
    walker           : optional pre-run RandomWalker for walk statistics
    """

    def __init__(
        self,
        G: nx.Graph,
        observed_labels: Dict[int, int],
        unknown_nodes: List[int],
        num_labels: int    = config.NUM_COMMUNITIES,
        iterations: int    = config.GIBBS_ITERATIONS,
        burn_in: int       = config.BURN_IN,
        smoothing: float   = config.SMOOTHING,
        walker: Optional[RandomWalker] = None,
        seed: int          = config.RANDOM_SEED,
    ):
        self.G               = G
        self.observed_labels = dict(observed_labels)
        self.unknown_nodes   = unknown_nodes
        self.num_labels      = num_labels
        self.iterations      = iterations
        self.burn_in         = burn_in
        self.smoothing       = smoothing
        self.walker          = walker
        self.rng             = np.random.default_rng(seed)

        # Current label state (observed + unknown initialised randomly)
        self.current_labels: Dict[int, int] = dict(observed_labels)
        for node in unknown_nodes:
            self.current_labels[node] = int(self.rng.integers(0, num_labels))

        # Posterior sample counts: sample_counts[node][label] = frequency
        self.sample_counts: Dict[int, np.ndarray] = {
            node: np.zeros(num_labels, dtype=int) for node in unknown_nodes
        }

        # Convergence tracking
        self.convergence_curve: List[float] = []   # label-change rate per iter

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> Dict[int, int]:
        """
        Runs the Gibbs Sampling loop.

        Returns
        -------
        predicted_labels : dict {unknown_node -> inferred_label}
        """
        print(f"\n[GibbsSampler] Starting: {self.iterations} iterations "
              f"(burn-in={self.burn_in}) on {len(self.unknown_nodes)} unknown nodes …")

        for t in tqdm(range(1, self.iterations + 1), desc="Gibbs Sampling"):
            changes = 0
            # Randomise sweep order each iteration
            nodes_order = list(self.unknown_nodes)
            self.rng.shuffle(nodes_order)

            for node in nodes_order:
                old_label = self.current_labels[node]
                new_label = self._sample_conditional(node)
                self.current_labels[node] = new_label
                if new_label != old_label:
                    changes += 1

                # Accumulate posterior samples after burn-in
                if t > self.burn_in:
                    self.sample_counts[node][new_label] += 1

            change_rate = changes / max(len(self.unknown_nodes), 1)
            self.convergence_curve.append(change_rate)

        predicted = {
            node: int(np.argmax(self.sample_counts[node]))
            for node in self.unknown_nodes
        }
        print(f"[GibbsSampler] Done. Final label-change rate: {self.convergence_curve[-1]:.4f}")
        return predicted

    def get_posterior_distribution(self, node: int) -> np.ndarray:
        """Returns normalised posterior label distribution for a node."""
        counts = self.sample_counts[node].astype(float)
        total = counts.sum()
        return counts / total if total > 0 else np.ones(self.num_labels) / self.num_labels

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _sample_conditional(self, node: int) -> int:
        """
        Samples label_i ~ P(label_i | labels of neighbors).

        Combines:
          (a) Immediate neighbor label counts (primary signal)
          (b) Walk-based label visit counts   (secondary signal, if walker available)
        """
        probs = self._neighbor_label_probs(node)

        if self.walker is not None:
            walk_probs = self.walker.get_label_visit_probs(
                node, self.num_labels, self.smoothing
            )
            # Geometric mean combination
            probs = np.sqrt(probs * walk_probs)
            probs /= probs.sum()

        return int(self.rng.choice(self.num_labels, p=probs))

    def _neighbor_label_probs(self, node: int) -> np.ndarray:
        """
        Computes conditional P(label | neighbors) from immediate neighbor labels.
        Uses Laplace smoothing.
        """
        counts = np.full(self.num_labels, self.smoothing)
        for neighbor in self.G.neighbors(node):
            lbl = self.current_labels.get(neighbor)
            if lbl is not None:
                counts[lbl] += 1.0
        return counts / counts.sum()
