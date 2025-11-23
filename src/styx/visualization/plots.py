"""Plotting utilities for visualizing training progress and metrics."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
):
    """Plot training and validation loss/metrics.

    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    metrics = []
    if "train_loss" in history:
        metrics.append("loss")

    if "train_metrics" in history:
        metrics.extend(history["train_metrics"].keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric == "loss":
            train_values = history["train_loss"]
            val_values = history.get("val_loss", [])
        else:
            train_values = history["train_metrics"][metric]
            val_values = history.get("val_metrics", {}).get(metric, [])

        epochs = range(1, len(train_values) + 1)
        ax.plot(epochs, train_values, label=f"Train {metric}", marker="o")

        if val_values:
            ax.plot(epochs, val_values, label=f"Val {metric}", marker="s")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_gradient_norms(
    gradient_norms: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Plot gradient norms over training steps.

    Args:
        gradient_norms: List of gradient norm values
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    steps = range(len(gradient_norms))
    plt.plot(steps, gradient_norms, alpha=0.7)
    plt.xlabel("Training Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_learning_rate_schedule(
    learning_rates: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Plot learning rate schedule over training.

    Args:
        learning_rates: List of learning rate values
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    steps = range(len(learning_rates))
    plt.plot(steps, learning_rates, linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def compare_optimizers(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "loss",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Compare multiple optimizers by plotting their training curves.

    Args:
        histories: Dict mapping optimizer names to their training histories
        metric: Metric to compare (default: "loss")
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    for name, history in histories.items():
        if metric == "loss":
            values = history["train_loss"]
        else:
            values = history["train_metrics"].get(metric, [])

        if values:
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=name, marker="o", markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Optimizer Comparison - {metric.capitalize()}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
