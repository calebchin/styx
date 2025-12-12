"""Plotting utilities for visualizing training progress and metrics."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def plot_escape_events(
    loss_history: List[float],
    alme_stats: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """Plot loss curve with escape events marked.

    Args:
        loss_history: Training or validation loss history
        alme_stats: List of ALME statistics dictionaries (one per epoch)
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot loss with escape markers
    epochs = range(1, len(loss_history) + 1)
    ax1.plot(epochs, loss_history, "b-", label="Loss", linewidth=2)

    # Mark escape events
    escape_epochs = []
    for epoch_idx, stats in enumerate(alme_stats):
        if epoch_idx > 0:
            prev_escapes = alme_stats[epoch_idx - 1]["escape_count"]
            curr_escapes = stats["escape_count"]
            if curr_escapes > prev_escapes:
                escape_epochs.append(epoch_idx + 1)

    if escape_epochs:
        escape_losses = [loss_history[e - 1] for e in escape_epochs]
        ax1.scatter(
            escape_epochs,
            escape_losses,
            color="red",
            s=100,
            marker="*",
            label="Escape Events",
            zorder=5,
        )

    ax1.set_ylabel("Loss")
    ax1.set_title("Loss History with Escape Events")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot gradient norms
    grad_norms = [stats["current_grad_norm"] for stats in alme_stats]
    ax2.plot(epochs, grad_norms, "g-", linewidth=2)

    if escape_epochs:
        escape_norms = [grad_norms[e - 1] for e in escape_epochs]
        ax2.scatter(escape_epochs, escape_norms, color="red", s=100, marker="*", zorder=5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Norms with Escape Events")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved escape events plot to {save_path}")
    else:
        plt.show()


def plot_population_diversity(
    alme_stats: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Plot population diversity metrics during escape events.

    Args:
        alme_stats: List of ALME statistics dictionaries
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Extract escape distances
    escape_epochs = []
    escape_distances = []
    cumulative_escapes = []

    for epoch_idx, stats in enumerate(alme_stats):
        cumulative_escapes.append(stats["escape_count"])
        if epoch_idx > 0:
            prev_escapes = alme_stats[epoch_idx - 1]["escape_count"]
            curr_escapes = stats["escape_count"]
            if curr_escapes > prev_escapes:
                escape_epochs.append(epoch_idx + 1)
                # Get the most recent escape distance
                if stats["escape_distances"]:
                    escape_distances.append(stats["escape_distances"][-1])

    # Plot escape distances
    if escape_distances:
        ax1.bar(range(1, len(escape_distances) + 1), escape_distances, color="steelblue")
        ax1.set_ylabel("Escape Distance")
        ax1.set_title("Distance Traveled During Each Escape")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_xlabel("Escape Event #")
    else:
        ax1.text(0.5, 0.5, "No escape events", ha="center", va="center")
        ax1.set_title("Distance Traveled During Each Escape")

    # Plot cumulative escapes
    epochs = range(1, len(cumulative_escapes) + 1)
    ax2.plot(epochs, cumulative_escapes, "o-", linewidth=2, markersize=4, color="darkred")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cumulative Escapes")
    ax2.set_title("Cumulative Escape Events Over Training")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved population diversity plot to {save_path}")
    else:
        plt.show()


def plot_optimizer_comparison_detailed(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10),
):
    """Create comprehensive comparison plot for multiple optimizers.

    Args:
        results: Dictionary mapping optimizer names to result dictionaries
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    for name, result in results.items():
        epochs = range(1, len(result["train_loss_history"]) + 1)
        ax1.plot(epochs, result["train_loss_history"], label=name, marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation loss
    ax2 = fig.add_subplot(gs[0, 1])
    for name, result in results.items():
        if "val_loss_history" in result:
            epochs = range(1, len(result["val_loss_history"]) + 1)
            ax2.plot(epochs, result["val_loss_history"], label=name, marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    for name, result in results.items():
        if "train_accuracy_history" in result:
            epochs = range(1, len(result["train_accuracy_history"]) + 1)
            ax3.plot(epochs, result["train_accuracy_history"], label=name, marker="o", markersize=3)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Training Accuracy")
    ax3.set_title("Training Accuracy Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Validation accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    for name, result in results.items():
        if "val_accuracy_history" in result:
            epochs = range(1, len(result["val_accuracy_history"]) + 1)
            ax4.plot(epochs, result["val_accuracy_history"], label=name, marker="s", markersize=3)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Validation Accuracy")
    ax4.set_title("Validation Accuracy Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Final metrics comparison (bar chart)
    ax5 = fig.add_subplot(gs[2, :])
    optimizer_names = list(results.keys())
    final_val_losses = [results[name]["final_val_loss"] for name in optimizer_names]
    final_val_accs = [
        results[name].get("final_val_accuracy", 0) * 100 for name in optimizer_names
    ]

    x = np.arange(len(optimizer_names))
    width = 0.35

    ax5_twin = ax5.twinx()
    bars1 = ax5.bar(x - width / 2, final_val_losses, width, label="Val Loss", color="steelblue")
    bars2 = ax5_twin.bar(
        x + width / 2, final_val_accs, width, label="Val Accuracy (%)", color="coral"
    )

    ax5.set_xlabel("Optimizer")
    ax5.set_ylabel("Validation Loss", color="steelblue")
    ax5_twin.set_ylabel("Validation Accuracy (%)", color="coral")
    ax5.set_title("Final Validation Metrics Comparison")
    ax5.set_xticks(x)
    ax5.set_xticklabels(optimizer_names)
    ax5.tick_params(axis="y", labelcolor="steelblue")
    ax5_twin.tick_params(axis="y", labelcolor="coral")
    ax5.grid(True, alpha=0.3, axis="y")

    # Add legends
    ax5.legend(loc="upper left")
    ax5_twin.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved detailed comparison plot to {save_path}")
    else:
        plt.show()


def plot_landscape_comparison(
    results: Dict[str, Dict[str, Dict]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """Compare optimizer performance across different landscape types.

    Args:
        results: Nested dict: {landscape_type: {optimizer_name: results}}
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    landscape_types = list(results.keys())
    n_landscapes = len(landscape_types)

    fig, axes = plt.subplots(2, n_landscapes, figsize=figsize)
    if n_landscapes == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, landscape in enumerate(landscape_types):
        landscape_results = results[landscape]

        # Plot loss history
        ax_loss = axes[0, col_idx]
        for opt_name, opt_results in landscape_results.items():
            if "loss_history" in opt_results:
                # Synthetic problem
                iters = range(len(opt_results["loss_history"]))
                ax_loss.plot(
                    iters,
                    opt_results["loss_history"],
                    label=opt_name,
                    alpha=0.8,
                )
            elif "val_loss_history" in opt_results:
                # Neural network
                epochs = range(1, len(opt_results["val_loss_history"]) + 1)
                ax_loss.plot(
                    epochs,
                    opt_results["val_loss_history"],
                    label=opt_name,
                    marker="o",
                    markersize=3,
                )

        ax_loss.set_xlabel("Iteration/Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{landscape}\nLoss Comparison")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale("log")

        # Plot final performance
        ax_final = axes[1, col_idx]
        opt_names = list(landscape_results.keys())
        final_losses = []

        for opt_name in opt_names:
            opt_results = landscape_results[opt_name]
            if "final_loss" in opt_results:
                final_losses.append(opt_results["final_loss"])
            elif "final_val_loss" in opt_results:
                final_losses.append(opt_results["final_val_loss"])
            else:
                final_losses.append(0)

        colors = ["steelblue" if "ALME" not in name else "coral" for name in opt_names]
        ax_final.bar(opt_names, final_losses, color=colors)
        ax_final.set_ylabel("Final Loss")
        ax_final.set_title(f"{landscape}\nFinal Performance")
        ax_final.grid(True, alpha=0.3, axis="y")
        ax_final.set_yscale("log")

        # Rotate x labels if needed
        if len(opt_names) > 2:
            ax_final.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved landscape comparison plot to {save_path}")
    else:
        plt.show()
