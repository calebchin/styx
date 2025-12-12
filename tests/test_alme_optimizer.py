"""Unit tests for ALME optimizer."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from styx.optimizers import ALME


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_data():
    """Create simple synthetic data for testing."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)
    return loader


class TestALMEInitialization:
    """Test ALME optimizer initialization."""

    def test_initialization_default_params(self, simple_model):
        """Test initialization with default parameters."""
        optimizer = ALME(simple_model.parameters())
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["betas"] == (0.9, 0.999)
        assert optimizer.defaults["eps"] == 1e-8
        assert optimizer.defaults["weight_decay"] == 0

    def test_initialization_custom_params(self, simple_model):
        """Test initialization with custom parameters."""
        optimizer = ALME(
            simple_model.parameters(),
            lr=0.01,
            betas=(0.95, 0.99),
            weight_decay=0.01,
            population_size=20,
            scale_distribution={0.5: 4, 1.0: 6, 2.0: 6, 4.0: 2, 8.0: 2},
        )
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["betas"] == (0.95, 0.99)
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.population_size == 20

    def test_invalid_lr_raises_error(self, simple_model):
        """Test that invalid learning rate raises error."""
        with pytest.raises(ValueError):
            ALME(simple_model.parameters(), lr=-0.01)

    def test_invalid_scale_distribution_raises_error(self, simple_model):
        """Test that invalid scale distribution raises error."""
        with pytest.raises(ValueError):
            ALME(
                simple_model.parameters(),
                population_size=10,
                scale_distribution={0.5: 5, 1.0: 3},  # Sum is 8, not 10
            )

    def test_state_initialization(self, simple_model):
        """Test that optimizer state is properly initialized."""
        optimizer = ALME(simple_model.parameters())
        assert "global" in optimizer.state
        assert optimizer.state["global"]["step"] == 0
        assert optimizer.state["global"]["escape_count"] == 0
        assert len(optimizer.state["global"]["grad_norm_history"]) == 0


class TestALMEParameterVectorUtilities:
    """Test parameter vector manipulation utilities."""

    def test_get_param_vector(self, simple_model):
        """Test parameter vector extraction."""
        optimizer = ALME(simple_model.parameters())
        param_vec = optimizer._get_param_vector()

        # Should be a 1D tensor
        assert param_vec.dim() == 1

        # Should have correct total number of parameters
        total_params = sum(p.numel() for p in simple_model.parameters())
        assert param_vec.numel() == total_params

    def test_set_param_vector(self, simple_model):
        """Test parameter vector assignment."""
        optimizer = ALME(simple_model.parameters())

        # Get original parameters
        original_vec = optimizer._get_param_vector().clone()

        # Create new random parameters
        new_vec = torch.randn_like(original_vec)

        # Set new parameters
        optimizer._set_param_vector(new_vec)

        # Verify parameters changed
        updated_vec = optimizer._get_param_vector()
        assert torch.allclose(updated_vec, new_vec)
        assert not torch.allclose(updated_vec, original_vec)

    def test_get_grad_vector(self, simple_model, simple_data):
        """Test gradient vector extraction."""
        optimizer = ALME(simple_model.parameters())
        criterion = nn.MSELoss()

        # Perform a forward/backward pass
        batch = next(iter(simple_data))
        X, y = batch
        output = simple_model(X)
        loss = criterion(output, y)
        loss.backward()

        # Get gradient vector
        grad_vec = optimizer._get_grad_vector()

        # Should be a 1D tensor
        assert grad_vec.dim() == 1

        # Should have correct size
        total_params = sum(p.numel() for p in simple_model.parameters())
        assert grad_vec.numel() == total_params


class TestALMEOptimizationStep:
    """Test ALME optimization step."""

    def test_single_step(self, simple_model, simple_data):
        """Test single optimization step."""
        optimizer = ALME(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Get initial parameters
        initial_params = optimizer._get_param_vector().clone()

        # Perform optimization step
        batch = next(iter(simple_data))
        X, y = batch
        optimizer.zero_grad()
        output = simple_model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        updated_params = optimizer._get_param_vector()
        assert not torch.allclose(updated_params, initial_params)

        # Step counter should increment
        assert optimizer.state["global"]["step"] == 1

    def test_gradient_norm_tracking(self, simple_model, simple_data):
        """Test that gradient norms are tracked."""
        optimizer = ALME(simple_model.parameters())
        criterion = nn.MSELoss()

        # Perform a few steps
        for batch in list(simple_data)[:3]:
            X, y = batch
            optimizer.zero_grad()
            output = simple_model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Gradient norm history should be populated
        assert len(optimizer.state["global"]["grad_norm_history"]) == 3


class TestALMEStagnationDetection:
    """Test stagnation detection mechanism."""

    def test_no_stagnation_early_training(self, simple_model):
        """Test that stagnation is not detected in early training."""
        optimizer = ALME(simple_model.parameters(), grad_norm_window=10)

        # Add a few gradient norms (not enough for window)
        for i in range(5):
            optimizer.state["global"]["grad_norm_history"].append(1.0 - i * 0.1)

        # Should not detect stagnation
        assert not optimizer._detect_stagnation()

    def test_stagnation_detection_gradient_norm(self, simple_model):
        """Test stagnation detection via gradient norm."""
        optimizer = ALME(
            simple_model.parameters(),
            grad_norm_window=5,
            grad_norm_threshold=1e-6,
            stagnation_patience=3,
            use_loss_plateau=False,
        )

        # Add constant gradient norms (stagnant)
        for _ in range(10):
            optimizer.state["global"]["grad_norm_history"].append(1e-7)
            optimizer._detect_stagnation()

        # Should detect stagnation after patience steps
        assert optimizer.state["global"]["stagnation_count"] >= 3

    def test_loss_history_update(self, simple_model):
        """Test manual loss history update."""
        optimizer = ALME(simple_model.parameters())

        # Update loss
        optimizer.update_loss(0.5)
        assert len(optimizer.state["global"]["loss_history"]) == 1
        assert optimizer.state["global"]["loss_history"][0] == 0.5


class TestALMECandidateSampling:
    """Test candidate sampling mechanism."""

    def test_sample_candidates_count(self, simple_model):
        """Test that correct number of candidates is sampled."""
        optimizer = ALME(
            simple_model.parameters(),
            population_size=10,
            scale_distribution={0.5: 2, 1.0: 3, 2.0: 3, 4.0: 1, 8.0: 1},
        )

        # Initialize Adam state (needed for step sizes)
        for p in simple_model.parameters():
            optimizer.state[p] = {
                "step": 1,
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.ones_like(p),
            }

        candidates = optimizer._sample_candidates()
        assert len(candidates) == 10

    def test_candidates_are_different(self, simple_model):
        """Test that sampled candidates are different from each other."""
        optimizer = ALME(
            simple_model.parameters(),
            population_size=5,
            scale_distribution={1.0: 5},
        )

        # Initialize Adam state
        for p in simple_model.parameters():
            optimizer.state[p] = {
                "step": 1,
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.ones_like(p),
            }

        candidates = optimizer._sample_candidates()

        # All candidates should be different
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                assert not torch.allclose(candidates[i], candidates[j])


class TestALMEBestParamsTracking:
    """Test best parameters tracking."""

    def test_update_best_params(self, simple_model):
        """Test updating best parameters."""
        optimizer = ALME(simple_model.parameters(), track_best=True)

        # Update with initial validation loss
        optimizer.update_best_params(1.0)
        assert optimizer.state["global"]["best_val_loss"] == 1.0
        assert optimizer.state["global"]["best_params"] is not None

        # Update with better loss
        optimizer.update_best_params(0.5)
        assert optimizer.state["global"]["best_val_loss"] == 0.5

        # Update with worse loss (should not change)
        best_params_before = optimizer.state["global"]["best_params"].clone()
        optimizer.update_best_params(1.5)
        assert optimizer.state["global"]["best_val_loss"] == 0.5
        assert torch.allclose(optimizer.state["global"]["best_params"], best_params_before)

    def test_get_best_params(self, simple_model):
        """Test retrieving best parameters."""
        optimizer = ALME(simple_model.parameters(), track_best=True)

        # Initially None
        assert optimizer.get_best_params() is None

        # After update
        optimizer.update_best_params(1.0)
        best_params = optimizer.get_best_params()
        assert best_params is not None
        assert best_params.numel() == sum(p.numel() for p in simple_model.parameters())

    def test_load_best_params(self, simple_model):
        """Test loading best parameters."""
        optimizer = ALME(simple_model.parameters(), track_best=True)

        # Save initial state
        optimizer.update_best_params(1.0)
        best_params = optimizer.get_best_params().clone()

        # Modify current parameters
        new_params = torch.randn_like(best_params)
        optimizer._set_param_vector(new_params)

        # Load best parameters
        optimizer.load_best_params()
        current_params = optimizer._get_param_vector()
        assert torch.allclose(current_params, best_params)


class TestALMEStats:
    """Test statistics collection."""

    def test_get_stats(self, simple_model):
        """Test getting optimizer statistics."""
        optimizer = ALME(simple_model.parameters())

        stats = optimizer.get_stats()
        assert "step" in stats
        assert "escape_count" in stats
        assert "escape_distances" in stats
        assert "avg_escape_distance" in stats
        assert "best_val_loss" in stats
        assert "current_grad_norm" in stats
        assert "stagnation_count" in stats

    def test_stats_after_updates(self, simple_model):
        """Test stats after some updates."""
        optimizer = ALME(simple_model.parameters())

        # Perform some updates
        optimizer.state["global"]["step"] = 10
        optimizer.state["global"]["escape_count"] = 2
        optimizer.state["global"]["escape_distances"] = [0.5, 0.3]
        optimizer.state["global"]["grad_norm_history"] = [1.0, 0.5, 0.3]

        stats = optimizer.get_stats()
        assert stats["step"] == 10
        assert stats["escape_count"] == 2
        assert stats["avg_escape_distance"] == 0.4
        assert stats["current_grad_norm"] == 0.3
