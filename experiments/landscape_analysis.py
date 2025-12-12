"""Landscape analysis experiments: testing ALME on smooth vs jagged loss surfaces.

This script creates synthetic optimization problems and real neural network tasks
with varying degrees of loss surface smoothness to evaluate ALME's performance.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from styx.datasets.loaders import get_mnist_loaders
from styx.experiments.trainer import Trainer
from styx.models.simple_nets import MLP
from styx.optimizers import ALME


# ============================================================================
# Synthetic Optimization Problems
# ============================================================================


class SyntheticOptimizationProblem(nn.Module):
    """Base class for synthetic optimization problems."""

    def __init__(self, dim: int):
        super().__init__()
        self.params = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy forward pass (not used for synthetic problems)."""
        return x

    def loss(self) -> torch.Tensor:
        """Compute loss based on current parameters."""
        raise NotImplementedError


class Rosenbrock(SyntheticOptimizationProblem):
    """Rosenbrock function: smooth, single global minimum.

    f(x, y) = (a - x)^2 + b(y - x^2)^2
    Generalized to n dimensions.
    """

    def __init__(self, dim: int = 2, a: float = 1.0, b: float = 100.0):
        super().__init__(dim)
        self.a = a
        self.b = b

    def loss(self) -> torch.Tensor:
        x = self.params
        sum_term = 0.0
        for i in range(len(x) - 1):
            sum_term += (self.a - x[i]) ** 2 + self.b * (x[i + 1] - x[i] ** 2) ** 2
        return sum_term


class Rastrigin(SyntheticOptimizationProblem):
    """Rastrigin function: highly multimodal, many local minima.

    f(x) = An + sum(x_i^2 - A*cos(2*pi*x_i))
    """

    def __init__(self, dim: int = 2, A: float = 10.0):
        super().__init__(dim)
        self.A = A

    def loss(self) -> torch.Tensor:
        x = self.params
        n = len(x)
        sum_term = torch.sum(x**2 - self.A * torch.cos(2 * np.pi * x))
        return self.A * n + sum_term


class Ackley(SyntheticOptimizationProblem):
    """Ackley function: many local minima with one global minimum.

    Characterized by a nearly flat outer region and a large hole at the center.
    """

    def __init__(self, dim: int = 2, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi):
        super().__init__(dim)
        self.a = a
        self.b = b
        self.c = c

    def loss(self) -> torch.Tensor:
        x = self.params
        n = len(x)
        sum1 = torch.sum(x**2)
        sum2 = torch.sum(torch.cos(self.c * x))
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / n))
        term2 = -torch.exp(sum2 / n)
        return term1 + term2 + self.a + np.e


def optimize_synthetic(
    problem: SyntheticOptimizationProblem,
    optimizer_type: str,
    optimizer_kwargs: Dict,
    max_iters: int = 1000,
) -> Dict:
    """Optimize a synthetic problem and track results.

    Args:
        problem: Synthetic optimization problem
        optimizer_type: Type of optimizer ('SGD', 'Adam', 'AdamW', 'ALME')
        optimizer_kwargs: Keyword arguments for optimizer
        max_iters: Maximum iterations

    Returns:
        Dictionary of results
    """
    # Create optimizer
    if optimizer_type == "SGD":
        optimizer = optim.SGD([problem.params], **optimizer_kwargs)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam([problem.params], **optimizer_kwargs)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW([problem.params], **optimizer_kwargs)
    elif optimizer_type == "ALME":
        optimizer = ALME([problem.params], **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Track results
    loss_history = []
    param_history = []
    is_alme = isinstance(optimizer, ALME)
    alme_stats = []

    # Optimization loop
    for iteration in range(max_iters):
        optimizer.zero_grad()
        loss = problem.loss()
        loss.backward()
        optimizer.step()

        # Track
        loss_history.append(loss.item())
        param_history.append(problem.params.detach().clone().cpu().numpy())

        # ALME-specific tracking
        if is_alme:
            optimizer.update_loss(loss.item())
            # For synthetic problems, we can't use check_and_escape with dataloader
            # So we manually check stagnation and sample if needed
            if optimizer._detect_stagnation():
                # Simple escape: just sample and evaluate
                candidates = optimizer._sample_candidates()
                best_loss = float("inf")
                best_params = None

                for candidate in candidates:
                    optimizer._set_param_vector(candidate)
                    candidate_loss = problem.loss().item()
                    if candidate_loss < best_loss:
                        best_loss = candidate_loss
                        best_params = candidate

                if best_params is not None:
                    optimizer._set_param_vector(best_params)
                    optimizer.state["global"]["escape_count"] += 1

            stats = optimizer.get_stats()
            alme_stats.append(stats)

        # Early stopping if converged
        if loss.item() < 1e-8:
            break

    results = {
        "optimizer": optimizer_type,
        "problem": problem.__class__.__name__,
        "final_loss": loss_history[-1],
        "min_loss": min(loss_history),
        "loss_history": loss_history,
        "param_history": [p.tolist() for p in param_history],  # Convert numpy arrays to lists
        "iterations": len(loss_history),
    }

    if is_alme:
        results["alme_stats"] = alme_stats
        results["final_alme_stats"] = optimizer.get_stats()

    return results


# ============================================================================
# Neural Network Landscape Analysis
# ============================================================================


def create_smooth_architecture() -> nn.Module:
    """Create a shallow architecture (smoother loss surface)."""
    return MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10, dropout=0.0)


def create_jagged_architecture() -> nn.Module:
    """Create a deeper architecture (more jagged loss surface)."""
    return MLP(input_dim=784, hidden_dims=[256, 128, 64, 32], output_dim=10, dropout=0.2)


def run_nn_landscape_experiment(
    architecture: str,
    optimizer_type: str,
    optimizer_kwargs: Dict,
    epochs: int = 20,
    batch_size: int = 64,
    l2_reg: float = 0.0,
    device: str = "cpu",
) -> Dict:
    """Run neural network experiment with specified landscape properties.

    Args:
        architecture: 'smooth' or 'jagged'
        optimizer_type: Type of optimizer
        optimizer_kwargs: Optimizer hyperparameters
        epochs: Number of training epochs
        batch_size: Batch size
        l2_reg: L2 regularization strength (smooths landscape)
        device: Device to train on

    Returns:
        Dictionary of results
    """
    # Create model
    if architecture == "smooth":
        model = create_smooth_architecture()
    elif architecture == "jagged":
        model = create_jagged_architecture()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)

    # Load data
    train_loader, val_loader = get_mnist_loaders(batch_size=batch_size, normalize=True)

    # Create optimizer
    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    elif optimizer_type == "ALME":
        optimizer = ALME(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Apply L2 regularization if needed
    if l2_reg > 0:
        for param_group in optimizer.param_groups:
            param_group["weight_decay"] = l2_reg

    # Create trainer
    criterion = nn.CrossEntropyLoss()

    def accuracy(output, target):
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        metrics={"accuracy": accuracy},
    )

    # Training loop
    is_alme = isinstance(optimizer, ALME)
    alme_stats = []

    for epoch in range(epochs):
        train_results = trainer.train_epoch(train_loader)
        trainer.history["train_loss"].append(train_results["loss"])
        trainer.history["train_metrics"]["accuracy"].append(train_results["accuracy"])

        if is_alme:
            optimizer.update_loss(train_results["loss"])

        val_results = trainer.validate(val_loader)
        trainer.history["val_loss"].append(val_results["loss"])
        trainer.history["val_metrics"]["accuracy"].append(val_results["accuracy"])

        if is_alme:
            optimizer.update_best_params(val_results["loss"])
            optimizer.check_and_escape(model, train_loader, val_loader, criterion, device)
            alme_stats.append(optimizer.get_stats())

    # Collect results
    results = {
        "optimizer": optimizer_type,
        "architecture": architecture,
        "l2_reg": l2_reg,
        "train_loss_history": trainer.history["train_loss"],
        "val_loss_history": trainer.history["val_loss"],
        "train_accuracy_history": trainer.history["train_metrics"]["accuracy"],
        "val_accuracy_history": trainer.history["val_metrics"]["accuracy"],
        "final_train_loss": trainer.history["train_loss"][-1],
        "final_val_loss": trainer.history["val_loss"][-1],
        "final_train_accuracy": trainer.history["train_metrics"]["accuracy"][-1],
        "final_val_accuracy": trainer.history["val_metrics"]["accuracy"][-1],
    }

    if is_alme:
        results["alme_stats"] = alme_stats
        results["final_alme_stats"] = optimizer.get_stats()

    return results


# ============================================================================
# Main Experiment Runner
# ============================================================================


def run_landscape_suite(save_dir: str = "experiments/results/landscape"):
    """Run complete landscape analysis suite."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ========================================
    # 1. Synthetic Problems
    # ========================================
    print("\n" + "=" * 60)
    print("SYNTHETIC OPTIMIZATION PROBLEMS")
    print("=" * 60)

    synthetic_problems = [
        ("Rosenbrock (smooth)", Rosenbrock(dim=10)),
        ("Rastrigin (jagged)", Rastrigin(dim=10)),
        ("Ackley (jagged)", Ackley(dim=10)),
    ]

    optimizers = [
        ("Adam", {"lr": 0.01}),
        ("ALME", {
            "lr": 0.01,
            "population_size": 10,
            "scale_distribution": {0.5: 2, 1.0: 3, 2.0: 3, 4.0: 1, 8.0: 1},
            "n_eval_steps": 3,
        }),
    ]

    for prob_name, problem in synthetic_problems:
        print(f"\n{prob_name}")
        all_results[prob_name] = {}

        for opt_name, opt_kwargs in optimizers:
            print(f"  Running {opt_name}...", end=" ")
            results = optimize_synthetic(problem, opt_name, opt_kwargs, max_iters=500)
            all_results[prob_name][opt_name] = results
            print(f"Final loss: {results['final_loss']:.6f}")

    # ========================================
    # 2. Neural Network Architectures
    # ========================================
    print("\n" + "=" * 60)
    print("NEURAL NETWORK LANDSCAPE ANALYSIS")
    print("=" * 60)

    nn_configs = [
        ("Smooth (shallow, no dropout, L2)", {
            "architecture": "smooth",
            "l2_reg": 0.01,
        }),
        ("Jagged (deep, dropout, no L2)", {
            "architecture": "jagged",
            "l2_reg": 0.0,
        }),
    ]

    nn_optimizers = [
        ("Adam", {"lr": 0.001}),
        ("ALME", {
            "lr": 0.001,
            "population_size": 10,
            "scale_distribution": {0.5: 2, 1.0: 3, 2.0: 3, 4.0: 1, 8.0: 1},
            "n_eval_steps": 3,
        }),
    ]

    for config_name, config in nn_configs:
        print(f"\n{config_name}")
        all_results[config_name] = {}

        for opt_name, opt_kwargs in nn_optimizers:
            print(f"  Running {opt_name}...")
            results = run_nn_landscape_experiment(
                optimizer_type=opt_name,
                optimizer_kwargs=opt_kwargs,
                epochs=15,
                **config,
            )
            all_results[config_name][opt_name] = results
            print(f"  Final val loss: {results['final_val_loss']:.4f}, "
                  f"Val accuracy: {results['final_val_accuracy']:.4f}")

    # Save all results
    results_file = save_path / "landscape_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to: {results_file}")
    print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run landscape analysis experiments")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/results/landscape",
        help="Directory to save results",
    )

    args = parser.parse_args()
    run_landscape_suite(args.save_dir)


if __name__ == "__main__":
    main()
