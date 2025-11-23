# Styx

A modular Python framework for experimenting with gradient descent techniques in deep learning.

## Features

- **Modular Architecture**: Clean separation of models, optimizers, datasets, and experiments
- **Custom Optimizers**: Easy-to-extend base optimizer class for implementing gradient descent variants
- **Neural Network Models**: Pre-built MLP and CNN architectures with configurable layers
- **Dataset Utilities**: Built-in loaders for MNIST and CIFAR-10 with preprocessing
- **Experiment Tracking**: Comprehensive training loop with metrics tracking and checkpointing
- **Visualization Tools**: Plot training curves, gradient norms, and compare optimizers
- **Reproducible Experiments**: JSON configuration files for versioning experiments

## Project Structure

```
styx/
├── src/styx/              # Main package
│   ├── optimizers/        # Custom gradient descent optimizers
│   ├── models/            # Neural network architectures
│   ├── datasets/          # Data loading utilities
│   ├── experiments/       # Training loops and experiment runners
│   └── visualization/     # Plotting and analysis tools
├── experiments/           # Experiment configurations and results
│   ├── configs/          # JSON configuration files
│   └── results/          # Training outputs and checkpoints
├── notebooks/            # Jupyter notebooks for exploration
└── tests/                # Unit tests

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/styx.git
cd styx

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using requirements.txt

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
import torch.nn as nn
from styx.models import MLP
from styx.datasets import get_mnist, get_dataloader
from styx.experiments import Trainer
from styx.visualization import plot_training_history

# Load dataset
train_dataset, test_dataset = get_mnist(download=True)
train_loader = get_dataloader(train_dataset, batch_size=64)
test_loader = get_dataloader(test_dataset, batch_size=64)

# Create model
model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    metrics={'accuracy': accuracy}
)

# Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=5
)

# Visualize
plot_training_history(history)
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

## Examples

Check out the [notebooks/](notebooks/) directory for detailed examples:
- [01_quickstart.ipynb](notebooks/01_quickstart.ipynb) - Basic usage and training

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

See [LICENSE](LICENSE) file for details.
