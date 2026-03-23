"""
utils/metrics.py
================
Evaluation metrics for the LinkedIn Random Walker project.

Metrics:
  - accuracy               : classification accuracy on unknown nodes
  - per_class_report       : precision / recall / F1 via sklearn
  - homophily_index        : fraction of same-label edges (imported from dataset)
  - convergence_curve      : label-change rate over Gibbs iterations
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_accuracy(
    predicted: Dict[int, int],
    true_labels: Dict[int, int],
    nodes: List[int],
) -> float:
    """
    Computes classification accuracy over `nodes`.

    Parameters
    ----------
    predicted   : {node -> predicted_label}
    true_labels : {node -> true_label}
    nodes       : subset of nodes to evaluate (unknown nodes)

    Returns
    -------
    accuracy : float in [0, 1]
    """
    y_true = [true_labels[n] for n in nodes if n in true_labels]
    y_pred = [predicted.get(n, -1) for n in nodes if n in true_labels]
    return accuracy_score(y_true, y_pred)


def compute_classification_report(
    predicted: Dict[int, int],
    true_labels: Dict[int, int],
    nodes: List[int],
    label_names: List[str] = config.COMMUNITY_LABELS,
) -> str:
    """
    Returns sklearn classification report string.
    """
    y_true = [true_labels[n] for n in nodes if n in true_labels]
    y_pred = [predicted.get(n, -1) for n in nodes if n in true_labels]
    labels = list(range(len(label_names)))
    return classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=label_names[:len(labels)],
        zero_division=0,
    )


def compute_confusion_matrix(
    predicted: Dict[int, int],
    true_labels: Dict[int, int],
    nodes: List[int],
    num_labels: int = config.NUM_COMMUNITIES,
) -> np.ndarray:
    """Returns confusion matrix (num_labels × num_labels)."""
    y_true = [true_labels[n] for n in nodes if n in true_labels]
    y_pred = [predicted.get(n, -1) for n in nodes if n in true_labels]
    labels = list(range(num_labels))
    return confusion_matrix(y_true, y_pred, labels=labels)


def convergence_stats(curve: List[float]) -> Dict:
    """Summarises Gibbs convergence from the label-change rate curve."""
    arr = np.array(curve)
    # Iteration where change rate first drops below 5%
    converged_at = next(
        (i + 1 for i, v in enumerate(arr) if v < 0.05),
        len(arr)
    )
    return {
        "initial_change_rate"  : float(arr[0]),
        "final_change_rate"    : float(arr[-1]),
        "converged_at_iter"    : converged_at,
        "mean_change_rate"     : float(arr.mean()),
    }
