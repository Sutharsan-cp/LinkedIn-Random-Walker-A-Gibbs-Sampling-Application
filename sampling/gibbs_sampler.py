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
        ground_truth: Optional[Dict[int, int]] = None,
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
        self.ground_truth    = ground_truth
        self.rng             = np.random.default_rng(seed)

        # Current label state (Bootstrapped initially)
        self.current_labels: Dict[int, int] = dict(observed_labels)
        for node in unknown_nodes:
            self.current_labels[node] = self._bootstrap_node(node)

        # Posterior sample counts (Phase 4)
        self.sample_counts: Dict[int, np.ndarray] = {
            node: np.zeros(num_labels, dtype=int) for node in unknown_nodes
        }

        # Diagnostics (Phase 5/6 extension)
        self.diagnostics = {
            "iteration": [],
            "log_likelihood": [],
            "accuracy": [],
            "mean_entropy": [],
            "change_rate": [],
            "map_stability": [],
        }
        self._prev_map_labels = {}

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

        # 2. Burn-in Phase (Phase 3)
        for t in tqdm(range(1, self.burn_in + 1), desc="Burn-in"):
            changes = 0
            nodes_order = list(self.unknown_nodes)
            self.rng.shuffle(nodes_order)
            for node in nodes_order:
                old_label = self.current_labels[node]
                new_label = self._sample_conditional(node)
                self.current_labels[node] = new_label
                if new_label != old_label:
                    changes += 1
            
            self._record_diagnostics(t, changes)

        # 3. Initialize Counts (Phase 4)
        for node in self.unknown_nodes:
            self.sample_counts[node].fill(0)
            self._prev_map_labels[node] = self.current_labels[node]

        # 4. Sampling Phase (Phase 5)
        sampling_iterations = max(1, self.iterations - self.burn_in)
        for t_rel in tqdm(range(1, sampling_iterations + 1), desc="Sampling"):
            t = self.burn_in + t_rel
            changes = 0
            nodes_order = list(self.unknown_nodes)
            self.rng.shuffle(nodes_order)

            for node in nodes_order:
                old_label = self.current_labels[node]
                new_label = self._sample_conditional(node)
                self.current_labels[node] = new_label
                if new_label != old_label:
                    changes += 1

                self.sample_counts[node][new_label] += 1

            self._record_diagnostics(t, changes)

        # Extract predicted labels for return
        predicted = {
            node: int(np.argmax(self.sample_counts[node]))
            for node in self.unknown_nodes
        }
        # Final convergence summary (using compatibility with original API)
        self.convergence_curve = self.diagnostics["change_rate"] 
        print(f"[GibbsSampler] Done. Final Accuracy: {self.diagnostics['accuracy'][-1]:.4f}")
        return predicted

    def get_posterior_distribution(self, node: int) -> np.ndarray:
        """Returns normalised posterior label distribution for a node."""
        counts = self.sample_counts[node].astype(float)
        total = counts.sum()
        return counts / total if total > 0 else np.ones(self.num_labels) / self.num_labels

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _bootstrap_node(self, node: int) -> int:
        """
        Initial bootstrapping using only observed neighbors.
        If no observed neighbors, assign a random label.
        """
        counts = np.full(self.num_labels, self.smoothing)
        for neighbor in self.G.neighbors(node):
            if neighbor in self.observed_labels:
                lbl = self.observed_labels[neighbor]
                counts[lbl] += 1.0
        
        probs = counts / counts.sum()
        return int(self.rng.choice(self.num_labels, p=probs))

    def _sample_conditional(self, node: int) -> int:
        """
        Samples label_i ~ P(label_i | labels of neighbors).
        f(a_i) implementation.
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
        Computes conditional P(label | neighbors). 
        This acts as f(a_i) in the user requirements.
        """
        counts = np.full(self.num_labels, self.smoothing)
        for neighbor in self.G.neighbors(node):
            lbl = self.current_labels.get(neighbor)
            if lbl is not None:
                counts[lbl] += 1.0
        return counts / counts.sum()

    # ── Diagnostic Helpers ───────────────────────────────────────────────────

    def _record_diagnostics(self, iteration: int, changes: int):
        """Records MCMC metrics. Note: label-change rate is unreliable due to label switching."""
        self.diagnostics["iteration"].append(iteration)
        self.diagnostics["change_rate"].append(changes / len(self.unknown_nodes))
        
        # 1. Log-Pseudo-Likelihood: sum_i log P(y_i | neighbors)
        log_pl = 0.0
        for node in self.unknown_nodes:
            probs = self._neighbor_label_probs(node)
            log_pl += np.log(max(probs[self.current_labels[node]], 1e-10))
        self.diagnostics["log_likelihood"].append(log_pl)

        # 2. Accuracy Tracking (with knowledge of ground truth)
        if self.ground_truth:
            correct = sum(1 for n in self.unknown_nodes if self.current_labels[n] == self.ground_truth[n])
            self.diagnostics["accuracy"].append(correct / len(self.unknown_nodes))
        else:
            self.diagnostics["accuracy"].append(0.0)

        # 3. Mean Posterior Entropy
        # For iterations after burn-in, we use sample_counts to estimate posterior
        if iteration > self.burn_in:
            entropies = []
            for node in self.unknown_nodes:
                p = self.get_posterior_distribution(node)
                # Shanon entropy: -sum p log p
                ent = -np.sum(p * np.log2(p + 1e-10))
                entropies.append(ent)
            self.diagnostics["mean_entropy"].append(np.mean(entropies))
        else:
            # During burn-in, entropy is not well-defined from samples
            self.diagnostics["mean_entropy"].append(np.log2(self.num_labels))

        # 4. MAP Stability: Fraction of nodes whose MAP label hasn't changed
        stable = 0
        for node in self.unknown_nodes:
            if iteration > self.burn_in:
                new_map = int(np.argmax(self.sample_counts[node]))
            else:
                new_map = self.current_labels[node]
            
            if new_map == self._prev_map_labels.get(node):
                stable += 1
            self._prev_map_labels[node] = new_map
        self.diagnostics["map_stability"].append(stable / len(self.unknown_nodes))
