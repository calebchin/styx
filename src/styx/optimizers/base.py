"""Base optimizer class for custom gradient descent implementations."""

from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    """Base class for custom optimizers.

    This serves as a template for implementing custom gradient descent variants.
    Extends PyTorch's Optimizer class to maintain compatibility with existing code.
    """

    def __init__(self, params, lr: float = 1e-3):
        """Initialize the optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Basic gradient descent step
                p.add_(p.grad, alpha=-group["lr"])

        return loss
