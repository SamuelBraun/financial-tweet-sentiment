"""
Evaluation module for financial tweet sentiment classification.

Provides metrics computation, confusion matrix plotting, cross-validation
helpers, and results table formatting.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import FIGURES_DIR, RANDOM_STATE

CLASS_NAMES = ["Bearish", "Bullish", "Neutral"]


# ===========================================================================
# Metrics computation
# ===========================================================================

def compute_metrics(y_true, y_pred, class_names: List[str] = CLASS_NAMES) -> dict:
    """Compute comprehensive classification metrics.

    Returns:
        Dictionary with accuracy, macro/per-class precision/recall/F1.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, name in enumerate(class_names):
        metrics[f"{name}_precision"] = per_class_p[i]
        metrics[f"{name}_recall"] = per_class_r[i]
        metrics[f"{name}_f1"] = per_class_f1[i]

    return metrics


def print_classification_report(y_true, y_pred,
                                class_names: List[str] = CLASS_NAMES,
                                model_name: str = "Model"):
    """Print a formatted sklearn classification report."""
    print(f"\n{'='*60}")
    print(f"Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")


# ===========================================================================
# Confusion matrix visualization
# ===========================================================================

def plot_confusion_matrix(y_true, y_pred,
                          class_names: List[str] = CLASS_NAMES,
                          model_name: str = "Model",
                          save_path: Optional[str] = None,
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (8, 6)):
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        normalize: If True, show percentages; otherwise raw counts.
        save_path: If given, save figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    plt.show()
    plt.close(fig)


# ===========================================================================
# Cross-validation helper
# ===========================================================================

def cross_validate_model(model, X, y, n_folds: int = 5,
                         class_names: List[str] = CLASS_NAMES,
                         model_name: str = "Model") -> Dict[str, float]:
    """Run stratified k-fold CV and return mean ± std of macro-F1.

    Also returns per-fold scores for detailed analysis.

    Args:
        model: Unfitted sklearn-compatible classifier.
        X: Feature matrix (dense or sparse).
        y: Labels array.
        n_folds: Number of folds.
        model_name: For display purposes.

    Returns:
        Dict with 'mean_macro_f1', 'std_macro_f1', 'fold_scores',
        and 'cv_predictions'.
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_STATE)
    fold_scores = []
    all_preds = np.zeros_like(y)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        clf = clone(model)
        clf.fit(X_train_fold, y_train_fold)
        preds = clf.predict(X_val_fold)

        fold_f1 = f1_score(y_val_fold, preds, average="macro", zero_division=0)
        fold_scores.append(fold_f1)
        all_preds[val_idx] = preds

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)

    print(f"{model_name}: Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f} "
          f"({n_folds}-fold CV)")

    return {
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "fold_scores": fold_scores,
        "cv_predictions": all_preds,
    }


# ===========================================================================
# Results table
# ===========================================================================

def build_results_table(results: List[Dict]) -> pd.DataFrame:
    """Build a summary DataFrame from a list of experiment results.

    Each result dict should have keys:
        'feature', 'model', 'macro_f1', 'std' (optional),
        'accuracy', 'preprocessing'.

    Returns:
        Formatted DataFrame suitable for the report.
    """
    df = pd.DataFrame(results)

    # Format F1 ± std if std is available
    if "std" in df.columns:
        df["Macro F1"] = df.apply(
            lambda r: f"{r['macro_f1']:.4f} ± {r['std']:.4f}"
            if pd.notna(r.get("std")) else f"{r['macro_f1']:.4f}",
            axis=1,
        )
    else:
        df["Macro F1"] = df["macro_f1"].apply(lambda x: f"{x:.4f}")

    if "accuracy" in df.columns:
        df["Accuracy"] = df["accuracy"].apply(lambda x: f"{x:.4f}")

    display_cols = ["preprocessing", "feature", "model", "Macro F1"]
    if "Accuracy" in df.columns:
        display_cols.append("Accuracy")

    return df[display_cols].rename(columns={
        "preprocessing": "Preprocessing",
        "feature": "Features",
        "model": "Classifier",
    })


def build_results_pivot(results: List[Dict]) -> pd.DataFrame:
    """Build a pivot table: rows=features, cols=classifiers, cells=macro-F1.

    This is the centerpiece table for the Evaluation section of the report.
    """
    df = pd.DataFrame(results)

    if "std" in df.columns:
        df["cell"] = df.apply(
            lambda r: f"{r['macro_f1']:.3f}±{r['std']:.3f}"
            if pd.notna(r.get("std")) else f"{r['macro_f1']:.3f}",
            axis=1,
        )
    else:
        df["cell"] = df["macro_f1"].apply(lambda x: f"{x:.3f}")

    # Create row label combining preprocessing + features
    df["row_label"] = df["preprocessing"] + " + " + df["feature"]

    pivot = df.pivot_table(
        index="row_label", columns="model", values="cell",
        aggfunc="first",
    )
    return pivot
