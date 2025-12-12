# ALME Quick Start Guide

## Getting Started with ALME

This guide will help you run your first ALME experiments and analyze the results.

## Step 1: Verify Installation

Make sure all dependencies are installed:

```bash
# From the project root
pip install -e ".[dev]"
```

## Step 2: Run Unit Tests

Verify the implementation works correctly:

```bash
pytest tests/test_alme_optimizer.py -v
```

Expected output: `20 passed`

## Step 3: Run Your First Experiment

### Option A: Single Experiment

Run ALME on MNIST with default settings:

```bash
python experiments/run_benchmark.py --config experiments/configs/alme_mnist_shallow.json
```

This will:
- Train a 2-layer MLP on MNIST for 20 epochs
- Use ALME optimizer with default hyperparameters
- Save results to `experiments/results/alme_mnist_shallow_results.json`

### Option B: Baseline Comparison

Run all baseline optimizers plus ALME:

```bash
python experiments/run_benchmark.py --all
```

This will run 5 experiments:
1. SGD with momentum
2. Adam
3. AdamW
4. ALME (shallow configuration)
5. ALME (aggressive configuration)

**Time estimate:** ~5-10 minutes total on CPU

## Step 4: Analyze Results

### View Results Files

Check the JSON results:

```bash
cat experiments/results/alme_mnist_shallow_results.json | python -m json.tool | head -50
```

Key metrics to look for:
- `final_val_loss`: Final validation loss
- `best_val_loss`: Best validation loss achieved
- `escape_count`: Number of local minima escapes
- `avg_escape_distance`: Average distance traveled during escapes

### Jupyter Notebook Analysis

Open the analysis notebook:

```bash
jupyter notebook notebooks/02_alme_analysis.ipynb
```

**Note:** You'll need to have run the experiments first (Step 3) for the notebook to load results.

## Step 5: Landscape Analysis (Optional)

Test ALME on smooth vs jagged loss surfaces:

```bash
python experiments/landscape_analysis.py
```

This runs ALME and Adam on:
- **Synthetic problems**: Rosenbrock (smooth), Rastrigin (jagged), Ackley (jagged)
- **Neural networks**: Shallow MLP with L2 (smooth), Deep MLP with dropout (jagged)

Results saved to `experiments/results/landscape/landscape_analysis_results.json`

## Understanding the Results

### ALME Statistics

When you run ALME experiments, look for these key indicators:

```python
{
  "final_alme_stats": {
    "escape_count": 3,              # Number of times ALME escaped
    "avg_escape_distance": 0.0042,  # How far it jumped each time
    "best_val_loss": 0.1234,        # Best loss seen during training
    "current_grad_norm": 0.0001,    # Current gradient magnitude
    "stagnation_count": 0           # Steps since last progress
  }
}
```

### Good Performance Indicators

- **escape_count > 0**: ALME detected and escaped local minima
- **best_val_loss < final_val_loss**: Escape mechanism found better solutions
- **avg_escape_distance > 0.001**: Meaningful exploration, not just noise

### Troubleshooting

**No escapes triggered:**
- Loss surface may be too smooth (expected on easy problems)
- Stagnation thresholds may be too strict
- Try the "aggressive" configuration

**Too many escapes:**
- Thresholds may be too lenient
- Increase `stagnation_patience` or tighten thresholds
- Normal on very jagged landscapes

**Worse than baselines:**
- Escape overhead may not be worth it on this problem
- Try reducing `population_size` or `n_eval_steps`
- ALME works best on non-convex problems with local minima

## Customizing ALME

### Create Your Own Config

Copy and modify an existing config:

```bash
cp experiments/configs/alme_mnist_shallow.json experiments/configs/my_experiment.json
```

Edit the hyperparameters:

```json
{
  "optimizer": {
    "type": "ALME",
    "params": {
      "lr": 0.001,                    // Adam learning rate
      "population_size": 15,          // Number of candidates to sample
      "scale_distribution": {         // Samples per scale multiplier
        "0.5": 3,
        "1.0": 5,
        "2.0": 5,
        "4.0": 2
      },
      "stagnation_patience": 3,       // Steps to wait before escape
      "grad_norm_threshold": 1e-5,    // Gradient norm improvement threshold
      "loss_threshold": 1e-4,         // Loss improvement threshold
      "n_eval_steps": 5,              // Gradient steps per candidate
      "eval_on": "train"              // Evaluate on "train" or "val"
    }
  }
}
```

Run your custom config:

```bash
python experiments/run_benchmark.py --config experiments/configs/my_experiment.json
```

### Use ALME in Your Code

```python
from styx.optimizers import ALME
import torch.nn as nn

# Your model
model = YourModel()

# Create ALME optimizer
optimizer = ALME(
    model.parameters(),
    lr=0.001,
    population_size=10,
    stagnation_patience=5,
)

# Training loop
for epoch in range(epochs):
    # Regular training
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()

    # End of epoch: check for stagnation
    train_loss = evaluate(model, train_loader)
    val_loss = evaluate(model, val_loader)

    optimizer.update_loss(train_loss)
    optimizer.update_best_params(val_loss)

    # Attempt escape if stagnant
    if optimizer.check_and_escape(model, train_loader, val_loader, criterion, device):
        print(f"Escaped at epoch {epoch}!")

    # Print stats
    stats = optimizer.get_stats()
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, "
          f"escapes={stats['escape_count']}, "
          f"grad_norm={stats['current_grad_norm']:.6f}")

# Load best parameters
optimizer.load_best_params()
```

## Visualization Examples

### Plot Escape Events

```python
from styx.visualization.plots import plot_escape_events
import json

# Load ALME results
results = json.load(open("experiments/results/alme_mnist_shallow_results.json"))

# Plot
plot_escape_events(
    results["val_loss_history"],
    results["alme_stats"],
    save_path="my_escapes.png"
)
```

### Compare Optimizers

```python
from styx.visualization.plots import plot_optimizer_comparison_detailed

results = {
    "SGD": json.load(open("experiments/results/baseline_sgd_results.json")),
    "Adam": json.load(open("experiments/results/baseline_adam_results.json")),
    "ALME": json.load(open("experiments/results/alme_mnist_shallow_results.json")),
}

plot_optimizer_comparison_detailed(
    results,
    save_path="optimizer_comparison.png"
)
```

## Next Steps

1. **Run all experiments** to establish baselines
2. **Analyze results** in the notebook
3. **Tune hyperparameters** for your specific problem
4. **Test on your own datasets** and models
5. **Experiment with scale distributions** to find optimal exploration

## Common Use Cases

### Research & Experimentation

ALME is great for:
- Exploring optimizer behavior
- Studying local minima in neural networks
- Benchmarking on non-convex problems

### Production Considerations

Before using ALME in production:
- Benchmark computational overhead
- Verify improvement over baselines
- Tune hyperparameters carefully
- Consider model size (works best on < 10M parameters)

## Getting Help

- Check [README.md](README.md) for algorithm details
- Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
- Look at test cases in `tests/test_alme_optimizer.py` for examples
- Open an issue on GitHub

Happy optimizing!
