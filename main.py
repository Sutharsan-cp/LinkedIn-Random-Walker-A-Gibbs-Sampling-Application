"""
main.py
=======
LinkedIn Random Walker: Gibbs Sampling Application
===================================================
Full pipeline entry point.

Pipeline:
  1. Build LinkedIn-like SBM graph
  2. Compute graph statistics & homophily index
  3. Mask labels (semi-supervised setup)
  4. Run Random Walker (builds neighbourhood statistics)
  5. Run Gibbs Sampler (MCMC label inference)
  6. Run Label Propagation (comparison baseline)
  7. Evaluate both methods
  8. Generate & save all visualizations
  9. Print summary report
"""

import os
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from graph.graph_builder import build_linkedin_graph
from graph.dataset       import mask_labels, graph_statistics, compute_homophily_index
from sampling.random_walker   import RandomWalker
from sampling.gibbs_sampler   import GibbsSampler
from propagation.label_propagation import LabelPropagation
from utils.metrics    import (
    compute_accuracy,
    compute_classification_report,
    compute_confusion_matrix,
    convergence_stats,
)
from utils.visualizer import (
    plot_graph,
    plot_convergence,
    plot_accuracy_comparison,
    plot_degree_distribution,
    plot_confusion_matrix,
    plot_posterior_distributions,
)


SEPARATOR = "=" * 65


def main():
    t0 = time.time()
    print(SEPARATOR)
    print("  LinkedIn Random Walker: Gibbs Sampling Application")
    print("  Homophily-Based Label Propagation via Gibbs Sampling")
    print(SEPARATOR)

    # ── 1. Build Graph ────────────────────────────────────────────────────────
    print("\n[1/9] Building LinkedIn graph …")
    G, node_labels = build_linkedin_graph()

    # ── 2. Graph Statistics ───────────────────────────────────────────────────
    print("\n[2/9] Computing graph statistics …")
    stats = graph_statistics(G, node_labels)

    # ── 3. Semi-Supervised Label Setup ────────────────────────────────────────
    print("\n[3/9] Setting up semi-supervised labels …")
    observed_labels, known_nodes, unknown_nodes = mask_labels(G, node_labels)

    # ── 4. Random Walker ──────────────────────────────────────────────────────
    print("\n[4/9] Running Random Walker …")
    walker = RandomWalker(G)
    walker.run_walks(observed_labels)

    # ── 5. Gibbs Sampling ─────────────────────────────────────────────────────
    print("\n[5/9] Running Gibbs Sampler …")
    sampler = GibbsSampler(
        G=G,
        observed_labels=observed_labels,
        unknown_nodes=unknown_nodes,
        walker=walker,
    )
    gibbs_predictions = sampler.run()

    # ── 6. Label Propagation (Baseline) ──────────────────────────────────────
    print("\n[6/9] Running Label Propagation (baseline) …")
    lp = LabelPropagation(G, observed_labels, unknown_nodes)
    lp_predictions = lp.run()

    # ── 7. Evaluation ─────────────────────────────────────────────────────────
    print("\n[7/9] Evaluating …")
    gibbs_acc = compute_accuracy(gibbs_predictions, node_labels, unknown_nodes)
    lp_acc    = compute_accuracy(lp_predictions,    node_labels, unknown_nodes)

    gibbs_report = compute_classification_report(
        gibbs_predictions, node_labels, unknown_nodes
    )
    cm = compute_confusion_matrix(gibbs_predictions, node_labels, unknown_nodes)
    conv_stats = convergence_stats(sampler.convergence_curve)

    # ── 8. Visualizations ────────────────────────────────────────────────────
    print(f"\n[8/9] Saving visualizations to '{config.OUTPUT_DIR}/' …")

    # All plots share the same spring-layout position for consistency
    pos = plot_graph(G, node_labels, "True Community Labels", "graph_true.png")
    plot_graph(G, {**observed_labels, **gibbs_predictions},
               "Gibbs Sampler Predicted Labels", "graph_gibbs.png", pos=pos)
    plot_graph(G, {**observed_labels, **lp_predictions},
               "Label Propagation Predicted Labels", "graph_lp.png", pos=pos)
    plot_convergence(sampler.convergence_curve)
    plot_accuracy_comparison(gibbs_acc, lp_acc)
    plot_degree_distribution(G)
    plot_confusion_matrix(cm)

    # Posterior distribution for a sample of unknown nodes
    sample_nodes = unknown_nodes[:6]
    plot_posterior_distributions(sampler, sample_nodes)

    # ── 9. Summary Report ────────────────────────────────────────────────────
    print(f"\n[9/9] Summary Report")
    print(SEPARATOR)
    print(f"  Graph:  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Homophily Index       : {stats['homophily_index']:.4f}")
    print(f"  Known Nodes           : {len(known_nodes)}")
    print(f"  Unknown Nodes         : {len(unknown_nodes)}")
    print(SEPARATOR)
    print(f"  Gibbs Sampler Accuracy: {gibbs_acc*100:.2f}%")
    print(f"  Label Propagation Acc : {lp_acc*100:.2f}%")
    print(SEPARATOR)
    print(f"  Convergence (Gibbs):")
    print(f"    Initial change rate : {conv_stats['initial_change_rate']:.4f}")
    print(f"    Final change rate   : {conv_stats['final_change_rate']:.4f}")
    print(f"    Converged at iter   : {conv_stats['converged_at_iter']}")
    print(SEPARATOR)
    print("\nPer-class Report (Gibbs Sampler):\n")
    print(gibbs_report)
    print(SEPARATOR)
    print(f"  Outputs saved to: {os.path.abspath(config.OUTPUT_DIR)}/")
    print(f"  Total runtime   : {time.time()-t0:.1f}s")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
