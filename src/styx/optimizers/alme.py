"""Adaptive Local Minima Escape (ALME) Optimizer.

This optimizer combines Adam-based gradient descent with a local minima escape
mechanism. When gradient stagnation or loss plateaus are detected, it samples
multiple candidate weight vectors, evaluates them via mini-optimization runs,
and continues from the best candidate.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.optim import Optimizer


class ALME(Optimizer):
    """Adaptive Local Minima Escape Optimizer.

    Combines Adam optimization with periodic local minima escape attempts.
    When stagnation is detected (via gradient norms or loss plateaus), the
    optimizer samples perturbed weight candidates, evaluates them with short
    optimization runs, and continues from the best candidate.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        grad_norm_window: Window size for tracking gradient norm history (default: 10)
        grad_norm_threshold: Threshold for gradient norm improvement to trigger escape (default: 1e-6)
        loss_window: Window size for tracking loss history (default: 10)
        loss_threshold: Threshold for loss improvement to trigger escape (default: 1e-5)
        stagnation_patience: Number of steps with stagnation before triggering escape (default: 5)
        use_loss_plateau: Whether to use loss plateau as secondary detection (default: True)
        population_size: Total number of candidates to sample during escape (default: 10)
        scale_distribution: Dict mapping scale multipliers to number of samples
            (default: {0.5: 2, 1: 3, 2: 3, 4: 1, 8: 1})
        n_eval_steps: Number of gradient steps per candidate evaluation (default: 3)
        eval_batch_size_factor: Factor to reduce batch size during evaluation (default: 1.0)
        eval_on: Whether to evaluate candidates on 'train' or 'val' loss (default: 'train')
        track_best: Whether to track best parameters seen by validation loss (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        grad_norm_window: int = 10,
        grad_norm_threshold: float = 1e-6,
        loss_window: int = 10,
        loss_threshold: float = 1e-5,
        stagnation_patience: int = 5,
        use_loss_plateau: bool = True,
        population_size: int = 10,
        scale_distribution: Optional[Dict[float, int]] = None,
        n_eval_steps: int = 3,
        eval_batch_size_factor: float = 1.0,
        eval_on: str = "train",
        track_best: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eval_on not in ["train", "val"]:
            raise ValueError(f"eval_on must be 'train' or 'val', got: {eval_on}")

        # Default scale distribution
        if scale_distribution is None:
            scale_distribution = {0.5: 2, 1.0: 3, 2.0: 3, 4.0: 1, 8.0: 1}

        # Validate scale distribution
        if sum(scale_distribution.values()) != population_size:
            raise ValueError(
                f"Scale distribution counts ({sum(scale_distribution.values())}) "
                f"must sum to population_size ({population_size})"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Hyperparameters for local minima detection
        self.grad_norm_window = grad_norm_window
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_window = loss_window
        self.loss_threshold = loss_threshold
        self.stagnation_patience = stagnation_patience
        self.use_loss_plateau = use_loss_plateau

        # Hyperparameters for candidate sampling and evaluation
        self.population_size = population_size
        self.scale_distribution = scale_distribution
        self.n_eval_steps = n_eval_steps
        self.eval_batch_size_factor = eval_batch_size_factor
        self.eval_on = eval_on
        self.track_best = track_best

        # Initialize tracking state
        self.state["global"] = {
            "step": 0,
            "grad_norm_history": [],
            "loss_history": [],
            "stagnation_count": 0,
            "escape_count": 0,
            "escape_distances": [],
            "best_val_loss": float("inf"),
            "best_params": None,
        }

    def _get_param_vector(self) -> torch.Tensor:
        """Flatten all parameters into a single 1D tensor."""
        return torch.cat([p.data.view(-1) for group in self.param_groups for p in group["params"]])

    def _set_param_vector(self, vec: torch.Tensor):
        """Set all parameters from a single 1D tensor."""
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                new_vals = vec[idx : idx + numel].view_as(p)
                p.data.copy_(new_vals)
                idx += numel

    def _get_grad_vector(self) -> torch.Tensor:
        """Flatten all gradients into a single 1D tensor."""
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).view(-1))
                else:
                    grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def _compute_grad_norm(self) -> float:
        """Compute L2 norm of gradients."""
        grad_vec = self._get_grad_vector()
        return torch.norm(grad_vec).item()

    def _detect_stagnation(self) -> bool:
        """Detect if optimizer is stuck in local minimum.

        Returns:
            True if stagnation detected, False otherwise
        """
        state = self.state["global"]

        # Need enough history
        if len(state["grad_norm_history"]) < self.grad_norm_window:
            return False

        # Primary: Check gradient norm stagnation
        recent_norms = state["grad_norm_history"][-self.grad_norm_window :]
        if len(recent_norms) >= 2:
            # Check if gradient norm improvement is below threshold
            norm_improvements = [
                abs(recent_norms[i] - recent_norms[i - 1]) for i in range(1, len(recent_norms))
            ]
            avg_improvement = sum(norm_improvements) / len(norm_improvements)

            grad_stagnant = avg_improvement < self.grad_norm_threshold
        else:
            grad_stagnant = False

        # Secondary: Check loss plateau (if enabled)
        loss_stagnant = False
        if self.use_loss_plateau and len(state["loss_history"]) >= self.loss_window:
            recent_losses = state["loss_history"][-self.loss_window :]
            if len(recent_losses) >= 2:
                loss_improvements = [
                    abs(recent_losses[i] - recent_losses[i - 1]) for i in range(1, len(recent_losses))
                ]
                avg_loss_improvement = sum(loss_improvements) / len(loss_improvements)
                loss_stagnant = avg_loss_improvement < self.loss_threshold

        # Trigger if both conditions met (or just gradient if loss plateau disabled)
        if self.use_loss_plateau:
            stagnant = grad_stagnant and loss_stagnant
        else:
            stagnant = grad_stagnant

        # Update stagnation counter
        if stagnant:
            state["stagnation_count"] += 1
        else:
            state["stagnation_count"] = 0

        # Trigger escape if stagnant for patience steps
        return state["stagnation_count"] >= self.stagnation_patience

    def _get_adam_step_sizes(self) -> torch.Tensor:
        """Compute per-parameter effective step sizes from Adam state.

        Returns:
            1D tensor of effective step sizes for each parameter
        """
        step_sizes = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p not in self.state:
                    # Not yet initialized, use lr as step size
                    step_sizes.append(torch.full_like(p, lr).view(-1))
                else:
                    state = self.state[p]
                    # Compute bias-corrected step size
                    # step = lr * sqrt(1 - beta2^t) / (1 - beta1^t) / (sqrt(v) + eps)
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                    effective_step = step_size / (exp_avg_sq.sqrt() + eps)

                    step_sizes.append(effective_step.view(-1))

        return torch.cat(step_sizes)

    def _sample_candidates(self) -> list[torch.Tensor]:
        """Sample perturbed weight candidates for local minima escape.

        Returns:
            List of candidate parameter vectors
        """
        current_params = self._get_param_vector()
        step_sizes = self._get_adam_step_sizes()

        candidates = []

        for scale, count in self.scale_distribution.items():
            for _ in range(count):
                # Sample Gaussian noise
                epsilon = torch.randn_like(current_params)

                # Scale by parameter magnitude and Adam step size
                perturbation = epsilon * current_params.abs() * step_sizes * scale

                # Create candidate
                candidate = current_params + perturbation
                candidates.append(candidate)

        return candidates

    def _evaluate_candidate(
        self,
        candidate: torch.Tensor,
        model,
        data_loader,
        criterion,
        device,
    ) -> Tuple[float, torch.Tensor]:
        """Evaluate a candidate by running mini-optimization.

        Args:
            candidate: Parameter vector to evaluate
            model: Model to optimize
            data_loader: DataLoader for evaluation
            criterion: Loss function
            device: Device to run on

        Returns:
            Tuple of (final_loss, final_params)
        """
        # Set candidate as current parameters
        self._set_param_vector(candidate.clone())

        # Run n_eval_steps gradient steps
        for step_idx in range(self.n_eval_steps):
            # Get a batch
            try:
                batch = next(iter(data_loader))
            except StopIteration:
                break

            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Manual Adam step on this candidate
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue

                        # Initialize state if needed
                        if p not in self.state:
                            self.state[p] = {
                                "step": 0,
                                "exp_avg": torch.zeros_like(p),
                                "exp_avg_sq": torch.zeros_like(p),
                            }

                        state = self.state[p]
                        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                        beta1, beta2 = group["betas"]

                        # Update biased first and second moment
                        exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                        # Compute step
                        denom = exp_avg_sq.sqrt().add_(group["eps"])
                        step_size = group["lr"]

                        # Update parameters
                        p.addcdiv_(exp_avg, denom, value=-step_size)

                        # Weight decay
                        if group["weight_decay"] != 0:
                            p.add_(p, alpha=-group["weight_decay"] * group["lr"])

        # Evaluate final loss
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        final_loss = total_loss / max(num_batches, 1)
        final_params = self._get_param_vector()

        return final_loss, final_params

    def _escape_local_minimum(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
    ):
        """Attempt to escape local minimum via candidate sampling.

        Args:
            model: Model being optimized
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            device: Device to run on
        """
        state = self.state["global"]
        current_params = self._get_param_vector().clone()

        # Sample candidates
        candidates = self._sample_candidates()

        # Determine which loader to use for evaluation
        eval_loader = train_loader if self.eval_on == "train" or val_loader is None else val_loader

        # Evaluate each candidate
        best_loss = float("inf")
        best_params = None

        for candidate in candidates:
            loss, final_params = self._evaluate_candidate(
                candidate, model, eval_loader, criterion, device
            )

            if loss < best_loss:
                best_loss = loss
                best_params = final_params.clone()

        # Set parameters to best candidate
        if best_params is not None:
            self._set_param_vector(best_params)

            # Track escape distance
            distance = torch.norm(best_params - current_params).item()
            state["escape_distances"].append(distance)
            state["escape_count"] += 1

        # Reset stagnation counter
        state["stagnation_count"] = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step (Adam).

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Standard Adam step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Initialize state if needed
                if p not in self.state:
                    self.state[p] = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    }

                state = self.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute step size
                step_size = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"] * group["lr"])

        # Update global step
        self.state["global"]["step"] += 1

        # Track gradient norm
        grad_norm = self._compute_grad_norm()
        self.state["global"]["grad_norm_history"].append(grad_norm)

        # Track loss if available
        if loss is not None:
            self.state["global"]["loss_history"].append(loss.item())

        return loss

    def update_loss(self, loss: float):
        """Manually update loss history (if step() doesn't use closure).

        Args:
            loss: Current training loss
        """
        self.state["global"]["loss_history"].append(loss)

    def check_and_escape(
        self,
        model,
        train_loader,
        val_loader=None,
        criterion=None,
        device="cpu",
    ) -> bool:
        """Check for stagnation and attempt escape if detected.

        Args:
            model: Model being optimized
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            device: Device to run on

        Returns:
            True if escape was triggered, False otherwise
        """
        if self._detect_stagnation():
            self._escape_local_minimum(model, train_loader, val_loader, criterion, device)
            return True
        return False

    def update_best_params(self, val_loss: float):
        """Update best parameters if validation loss improved.

        Args:
            val_loss: Current validation loss
        """
        if not self.track_best:
            return

        state = self.state["global"]
        if val_loss < state["best_val_loss"]:
            state["best_val_loss"] = val_loss
            state["best_params"] = self._get_param_vector().clone()

    def get_best_params(self) -> Optional[torch.Tensor]:
        """Get the best parameters seen during optimization.

        Returns:
            Best parameter vector, or None if not tracked
        """
        return self.state["global"]["best_params"]

    def load_best_params(self):
        """Load the best parameters into the model."""
        best_params = self.get_best_params()
        if best_params is not None:
            self._set_param_vector(best_params)

    def get_stats(self) -> Dict:
        """Get optimization statistics.

        Returns:
            Dictionary of statistics including escape count, distances, etc.
        """
        state = self.state["global"]
        return {
            "step": state["step"],
            "escape_count": state["escape_count"],
            "escape_distances": state["escape_distances"],
            "avg_escape_distance": (
                sum(state["escape_distances"]) / len(state["escape_distances"])
                if state["escape_distances"]
                else 0.0
            ),
            "best_val_loss": state["best_val_loss"],
            "current_grad_norm": (
                state["grad_norm_history"][-1] if state["grad_norm_history"] else 0.0
            ),
            "stagnation_count": state["stagnation_count"],
        }
