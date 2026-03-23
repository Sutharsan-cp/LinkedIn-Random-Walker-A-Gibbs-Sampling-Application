"""
utils/visualizer.py
===================
All visualization functions for the LinkedIn Random Walker project.

Plots generated:
  1. graph_true.png           – Graph colored by true labels
  2. graph_gibbs.png          – Graph colored by Gibbs-predicted labels
  3. graph_lp.png             – Graph colored by LP-predicted labels
  4. convergence.png          – Gibbs label-change rate over iterations
  5. accuracy_comparison.png  – Bar chart: Gibbs vs LP accuracy
  6. degree_distribution.png  – Node degree histogram
  7. confusion_matrix.png     – Gibbs confusion matrix heatmap
"""

import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Professional colour palette (one per community)
PALETTE = [
    "#4C72B0",  # Blue       – Software Engineer
    "#DD8452",  # Orange     – Data Scientist
    "#55A868",  # Green      – Product Manager
    "#C44E52",  # Red        – UX Designer
    "#8172B3",  # Purple     – Business Analyst
]

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _get_node_colors(G: nx.Graph, labels: Dict[int, int], alpha_unknown: bool = False) -> List:
    """Maps label indices to colours."""
    colors = []
    for node in G.nodes():
        lbl = labels.get(node, -1)
        if lbl == -1:
            colors.append("#CCCCCC")
        else:
            colors.append(PALETTE[lbl % len(PALETTE)])
    return colors


def _label_legend(label_names: List[str]) -> List[mpatches.Patch]:
    return [
        mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=label_names[i])
        for i in range(len(label_names))
    ]


def plot_graph(
    G: nx.Graph,
    labels: Dict[int, int],
    title: str,
    filename: str,
    label_names: List[str] = config.COMMUNITY_LABELS,
    pos=None,
) -> object:
    """Draws the graph with nodes coloured by label. Returns the layout position."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0D0D0D")
    fig.patch.set_facecolor("#0D0D0D")

    if pos is None:
        pos = nx.spring_layout(G, seed=config.RANDOM_SEED, k=0.6)

    colors = _get_node_colors(G, labels)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08, edge_color="white", width=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=55,
                           linewidths=0.5, edgecolors="white")

    ax.legend(handles=_label_legend(label_names), loc="upper left",
              framealpha=0.3, facecolor="#1A1A2E", labelcolor="white",
              fontsize=9, title="Community", title_fontsize=9)
    ax.set_title(title, color="white", fontsize=14, pad=12, fontweight="bold")
    ax.axis("off")

    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return pos


def plot_convergence(curve: List[float], burn_in: int = config.BURN_IN) -> None:
    """Plots Gibbs label-change rate over iterations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")

    x = list(range(1, len(curve) + 1))
    ax.plot(x, curve, color="#4C72B0", lw=1.8, label="Label-change rate")
    ax.axvline(burn_in, color="#DD8452", linestyle="--", lw=1.2, label=f"Burn-in ({burn_in})")
    ax.axhline(0.05, color="#55A868", linestyle=":", lw=1.2, label="5% threshold")
    ax.fill_between(x, curve, alpha=0.15, color="#4C72B0")

    ax.set_xlabel("Iteration", color="white")
    ax.set_ylabel("Label-change rate", color="white")
    ax.set_title("Gibbs Sampler Convergence", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(framealpha=0.3, facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    path = os.path.join(config.OUTPUT_DIR, "convergence.png")
    plt.tight_layout()
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_comparison(
    gibbs_acc: float,
    lp_acc: float,
    label_names: Optional[List[str]] = None,
) -> None:
    """Bar chart comparing Gibbs vs Label Propagation accuracy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")

    methods = ["Gibbs Sampling\n(MCMC)", "Label Propagation\n(Baseline)"]
    accs    = [gibbs_acc * 100, lp_acc * 100]
    colors  = ["#4C72B0", "#DD8452"]

    bars = ax.bar(methods, accs, color=colors, width=0.45, edgecolor="#333", linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", color="white", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.set_title("Method Comparison: Gibbs vs Label Propagation", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.axhline(100, color="#555", linestyle="--", lw=0.8)

    path = os.path.join(config.OUTPUT_DIR, "accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_degree_distribution(G: nx.Graph) -> None:
    """Histogram of node degree distribution."""
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")

    ax.hist(degrees, bins=30, color="#4C72B0", edgecolor="#1A1A2E", alpha=0.85)
    ax.axvline(np.mean(degrees), color="#DD8452", linestyle="--",
               lw=1.5, label=f"Mean degree = {np.mean(degrees):.1f}")
    ax.set_xlabel("Degree", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Degree Distribution of LinkedIn Graph", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(framealpha=0.3, facecolor="#1A1A2E", labelcolor="white", fontsize=9)

    path = os.path.join(config.OUTPUT_DIR, "degree_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str] = config.COMMUNITY_LABELS,
) -> None:
    """Seaborn heatmap of confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0D1117")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[l[:10] for l in label_names],
        yticklabels=[l[:10] for l in label_names],
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Gibbs Sampler – Confusion Matrix", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True", color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_facecolor("#0D1117")

    path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_posterior_distributions(
    sampler,
    nodes_sample: List[int],
    label_names: List[str] = config.COMMUNITY_LABELS,
) -> None:
    """Bar charts of posterior label distributions for a few nodes."""
    n = min(len(nodes_sample), 6)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor("#0D1117")
    if n == 1:
        axes = [axes]

    for ax, node in zip(axes, nodes_sample[:n]):
        probs = sampler.get_posterior_distribution(node)
        names = [l[:8] for l in label_names]
        colors = PALETTE[:len(names)]
        ax.bar(names, probs, color=colors, edgecolor="#1A1A2E")
        ax.set_title(f"Node {node}", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_ylabel("P(label)", color="white")
        ax.set_facecolor("#0D1117")
        ax.spines[:].set_color("#333")
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)

    fig.suptitle("Posterior Label Distributions (Gibbs)", color="white",
                 fontsize=12, fontweight="bold")
    path = os.path.join(config.OUTPUT_DIR, "posterior_distributions.png")
    plt.tight_layout()
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
