# Styx

Experimenting with gradient descent and optimization algorithms for deep learning.

This project implements and benchmarks various optimization algorithms, with a focus on escaping local minima through adaptive exploration strategies.

## Features

- **ALME Optimizer**: Adaptive Local Minima Escape optimizer combining Adam with population-based exploration
- **Baseline Optimizers**: Implementations and benchmarks for SGD, Adam, and AdamW
- **Landscape Analysis**: Tools for testing optimizers on smooth vs jagged loss surfaces
- **Comprehensive Visualization**: Plotting utilities for escape events, gradient norms, and optimizer comparisons
- **Flexible Training Infrastructure**: Modular trainer class with metric tracking and checkpointing

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/calebchin/styx.git
cd styx

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

Check out the notebooks for examples:
- [01_quickstart.ipynb](notebooks/01_quickstart.ipynb) - Introduction to the modules in Styx
- [02_alme_analysis.ipynb](notebooks/02_alme_analysis.ipynb) - ALME optimizer analysis and benchmarks

### Using the ALME Optimizer

```python
from styx.optimizers import ALME
from styx.models.simple_nets import MLP
import torch.nn as nn

# Create model
model = MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)

# Create ALME optimizer
optimizer = ALME(
    model.parameters(),
    lr=0.001,
    population_size=10,
    scale_distribution={0.5: 2, 1.0: 3, 2.0: 3, 4.0: 1, 8.0: 1},
    n_eval_steps=3,
    stagnation_patience=5,
)

# Training loop
for epoch in range(epochs):
    # Standard training
    train_loss = train_epoch(model, optimizer, train_loader, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Update ALME tracking
    optimizer.update_loss(train_loss)
    optimizer.update_best_params(val_loss)

    # Check for stagnation and attempt escape
    escaped = optimizer.check_and_escape(
        model, train_loader, val_loader, criterion, device
    )

    if escaped:
        print(f"Escaped local minimum at epoch {epoch}!")

# Load best parameters found during training
optimizer.load_best_params()
```

### Running Experiments

Run benchmark comparisons:
```bash
# Run all baseline and ALME experiments
python experiments/run_benchmark.py --all

# Run a specific configuration
python experiments/run_benchmark.py --config experiments/configs/alme_mnist_shallow.json

# Run landscape analysis
python experiments/landscape_analysis.py
```

## Implementing Custom Optimizers

```python
from styx.optimizers import BaseOptimizer
import torch

class MomentumSGD(BaseOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, lr=lr)
        self.defaults.update(defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p.data)

                velocity = state['velocity']
                velocity.mul_(group['momentum']).add_(p.grad)
                p.add_(velocity, alpha=-group['lr'])

        return None
```

## Running Tests

```bash
pytest
```

## ALME Algorithm Details

The Adaptive Local Minima Escape (ALME) optimizer combines gradient-based optimization with population-based exploration:

### Algorithm Overview

1. **Normal Gradient Descent**: Uses Adam optimizer for standard parameter updates
2. **Stagnation Detection**: Monitors gradient norms and loss improvements to detect local minima
   - Primary: Gradient norm stops decreasing
   - Secondary: Loss plateaus
3. **Candidate Sampling**: When stagnation detected, samples perturbed weight vectors
   - Perturbations: `w = ŵ + ε`, where `ε ~ N(0, σ²) * |ŵ|`
   - Multiple scales: {0.5, 1, 2, 4, 8} × Adam's per-parameter step sizes
4. **Candidate Evaluation**: Runs mini-optimization for each candidate
   - Takes k gradient steps with optional batch size reduction
   - Evaluates on training or validation loss
5. **Escape**: Continues from the best candidate's updated position

### Key Hyperparameters

- `population_size`: Number of candidates to sample (default: 10)
- `scale_distribution`: Dict mapping scale multipliers to sample counts
- `stagnation_patience`: Steps to wait before triggering escape (default: 5)
- `grad_norm_threshold`: Threshold for gradient norm improvement (default: 1e-6)
- `loss_threshold`: Threshold for loss improvement (default: 1e-5)
- `n_eval_steps`: Gradient steps per candidate evaluation (default: 3)

### Use Cases

ALME is particularly effective for:
- Non-convex optimization problems with many local minima
- Jagged or rough loss landscapes
- Problems where standard optimizers get stuck
- When you want to maintain the best parameters seen during training

## Project Structure

```
styx/
├── src/styx/
│   ├── datasets/          # Dataset loaders (MNIST, CIFAR-10)
│   ├── experiments/       # Training infrastructure
│   ├── models/           # Neural network architectures
│   ├── optimizers/       # Optimizer implementations
│   │   ├── alme.py      # ALME optimizer
│   │   └── base.py      # Base optimizer class
│   └── visualization/    # Plotting utilities
├── experiments/
│   ├── configs/         # Experiment configurations
│   ├── results/         # Experiment outputs
│   ├── run_benchmark.py # Benchmark script
│   └── landscape_analysis.py # Landscape testing
├── notebooks/           # Jupyter notebooks
├── tests/              # Unit tests
└── README.md
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

See [LICENSE](LICENSE) file for details.
