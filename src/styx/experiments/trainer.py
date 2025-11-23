"""Training loop implementation with metrics tracking."""

import json
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for running experiments and tracking metrics.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on ('cuda' or 'cpu')
        metrics: Optional dict of metric functions {name: callable}
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics or {}

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {name: [] for name in self.metrics.keys()},
            "val_metrics": {name: [] for name in self.metrics.keys()},
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics.keys()}
        num_batches = 0

        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Calculate metrics
            for name, metric_fn in self.metrics.items():
                metric_totals[name] += metric_fn(output, target)

        results = {"loss": total_loss / num_batches}
        for name in self.metrics.keys():
            results[name] = metric_totals[name] / num_batches

        return results

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics.keys()}
        num_batches = 0

        for data, target in tqdm(dataloader, desc="Validation"):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

            # Calculate metrics
            for name, metric_fn in self.metrics.items():
                metric_totals[name] += metric_fn(output, target)

        results = {"loss": total_loss / num_batches}
        for name in self.metrics.keys():
            results[name] = metric_totals[name] / num_batches

        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            Training history
        """
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")

            train_results = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_results["loss"])

            for name in self.metrics.keys():
                self.history["train_metrics"][name].append(train_results[name])

            if verbose:
                metrics_str = " - ".join([f"train_{k}: {v:.4f}" for k, v in train_results.items()])
                print(metrics_str)

            if val_loader is not None:
                val_results = self.validate(val_loader)
                self.history["val_loss"].append(val_results["loss"])

                for name in self.metrics.keys():
                    self.history["val_metrics"][name].append(val_results[name])

                if verbose:
                    metrics_str = " - ".join([f"val_{k}: {v:.4f}" for k, v in val_results.items()])
                    print(metrics_str)

        return self.history

    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            **kwargs: Additional items to save in checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            **kwargs,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)

    def save_history(self, path: str):
        """Save training history to JSON file.

        Args:
            path: Path to save history
        """
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
