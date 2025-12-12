"""Benchmark script comparing ALME against baseline optimizers.

This script runs experiments with different optimizers (SGD, Adam, AdamW, ALME)
on MNIST and compares their performance. Results are saved to the results directory.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from styx.datasets.loaders import get_mnist_loaders
from styx.experiments.trainer import Trainer
from styx.models.simple_nets import MLP
from styx.optimizers import ALME


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_model(config: Dict) -> nn.Module:
    """Create model from configuration."""
    model_config = config["model"]
    if model_config["type"] == "MLP":
        return MLP(**model_config["params"])
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer from configuration."""
    opt_config = config["optimizer"]
    opt_type = opt_config["type"]
    params = opt_config["params"]

    if opt_type == "SGD":
        return optim.SGD(model.parameters(), **params)
    elif opt_type == "Adam":
        return optim.Adam(model.parameters(), **params)
    elif opt_type == "AdamW":
        return optim.AdamW(model.parameters(), **params)
    elif opt_type == "ALME":
        return ALME(model.parameters(), **params)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def accuracy_metric(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute classification accuracy."""
    pred = output.argmax(dim=1)
    correct = (pred == target).sum().item()
    return correct / target.size(0)


def run_experiment(config_path: str, save_dir: str):
    """Run a single experiment based on configuration.

    Args:
        config_path: Path to configuration JSON file
        save_dir: Directory to save results
    """
    # Load configuration
    config = load_config(config_path)
    print(f"\n{'='*60}")
    print(f"Running experiment: {config['experiment_name']}")
    print(f"{'='*60}\n")

    # Set device
    device = config["training"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Create model
    model = create_model(config)
    print(f"Model: {config['model']['type']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = create_optimizer(model, config)
    print(f"\nOptimizer: {config['optimizer']['type']}")
    print(f"Optimizer params: {config['optimizer']['params']}")

    # Load dataset
    dataset_config = config["dataset"]
    batch_size = config["training"]["batch_size"]

    if dataset_config["name"] == "mnist":
        train_loader, val_loader = get_mnist_loaders(
            batch_size=batch_size,
            normalize=dataset_config.get("normalize", True),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_config['name']}")

    # Create trainer
    criterion = nn.CrossEntropyLoss()
    metrics = {"accuracy": accuracy_metric}

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        metrics=metrics,
    )

    # Training loop
    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")

    # Track ALME-specific stats if using ALME
    is_alme = isinstance(optimizer, ALME)
    alme_stats = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_results = trainer.train_epoch(train_loader)
        trainer.history["train_loss"].append(train_results["loss"])
        trainer.history["train_metrics"]["accuracy"].append(train_results["accuracy"])

        print(f"Train Loss: {train_results['loss']:.4f}, "
              f"Train Accuracy: {train_results['accuracy']:.4f}")

        # Update loss history for ALME
        if is_alme:
            optimizer.update_loss(train_results["loss"])

        # Validate
        val_results = trainer.validate(val_loader)
        trainer.history["val_loss"].append(val_results["loss"])
        trainer.history["val_metrics"]["accuracy"].append(val_results["accuracy"])

        print(f"Val Loss: {val_results['loss']:.4f}, "
              f"Val Accuracy: {val_results['accuracy']:.4f}")

        # Update best parameters for ALME
        if is_alme:
            optimizer.update_best_params(val_results["loss"])

            # Check for local minima and attempt escape
            escaped = optimizer.check_and_escape(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            if escaped:
                print(f"  -> Escaped local minimum!")

            # Get ALME stats
            stats = optimizer.get_stats()
            alme_stats.append(stats)
            print(f"  -> Escape count: {stats['escape_count']}, "
                  f"Grad norm: {stats['current_grad_norm']:.6f}")

        # Track best validation loss
        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]

    # Final results
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if is_alme:
        final_stats = optimizer.get_stats()
        print(f"\nALME Statistics:")
        print(f"  Total escapes: {final_stats['escape_count']}")
        print(f"  Average escape distance: {final_stats['avg_escape_distance']:.6f}")
        print(f"  Best validation loss: {final_stats['best_val_loss']:.4f}")

    # Save results
    results = {
        "config": config,
        "final_train_loss": trainer.history["train_loss"][-1],
        "final_val_loss": trainer.history["val_loss"][-1],
        "best_val_loss": best_val_loss,
        "train_loss_history": trainer.history["train_loss"],
        "val_loss_history": trainer.history["val_loss"],
        "train_accuracy_history": trainer.history["train_metrics"]["accuracy"],
        "val_accuracy_history": trainer.history["val_metrics"]["accuracy"],
    }

    if is_alme:
        results["alme_stats"] = alme_stats
        results["final_alme_stats"] = optimizer.get_stats()

    # Save to file
    save_path = Path(save_dir) / f"{config['experiment_name']}_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {save_path}")


def run_all_baselines(configs_dir: str, save_dir: str):
    """Run all baseline experiments.

    Args:
        configs_dir: Directory containing configuration files
        save_dir: Directory to save results
    """
    configs_dir = Path(configs_dir)

    # Run baseline experiments
    baseline_configs = [
        "baseline_sgd.json",
        "baseline_adam.json",
        "baseline_adamw.json",
    ]

    for config_name in baseline_configs:
        config_path = configs_dir / config_name
        if config_path.exists():
            run_experiment(str(config_path), save_dir)
        else:
            print(f"Warning: Config not found: {config_path}")

    # Run ALME experiments
    alme_configs = [
        "alme_mnist_shallow.json",
        "alme_mnist_aggressive.json",
    ]

    for config_name in alme_configs:
        config_path = configs_dir / config_name
        if config_path.exists():
            run_experiment(str(config_path), save_dir)
        else:
            print(f"Warning: Config not found: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Run optimizer benchmark experiments")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to specific config file (runs single experiment)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all baseline and ALME experiments",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="experiments/configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    if args.config:
        # Run single experiment
        run_experiment(args.config, args.save_dir)
    elif args.all:
        # Run all experiments
        run_all_baselines(args.configs_dir, args.save_dir)
    else:
        print("Please specify either --config or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
