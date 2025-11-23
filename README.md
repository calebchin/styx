# Styx

Experimenting with gradient descent for deep learning.

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

Check out [01_quickstart.ipynb](notebooks/01_quickstart.ipynb) for a short example of how to use the modules in styx.  

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

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

See [LICENSE](LICENSE) file for details.
