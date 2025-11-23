"""Tests for model architectures."""

import torch

from styx.models import MLP, SimpleCNN


def test_mlp_forward():
    """Test MLP forward pass."""
    model = MLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)
    x = torch.randn(32, 784)
    output = model(x)
    assert output.shape == (32, 10)


def test_mlp_with_image_input():
    """Test MLP with image-shaped input (should flatten)."""
    model = MLP(input_dim=784, hidden_dims=[128], output_dim=10)
    x = torch.randn(32, 1, 28, 28)
    output = model(x)
    assert output.shape == (32, 10)


def test_simple_cnn_mnist():
    """Test SimpleCNN with MNIST-like input."""
    model = SimpleCNN(in_channels=1, num_classes=10, image_size=28)
    x = torch.randn(32, 1, 28, 28)
    output = model(x)
    assert output.shape == (32, 10)


def test_simple_cnn_cifar():
    """Test SimpleCNN with CIFAR-10-like input."""
    model = SimpleCNN(in_channels=3, num_classes=10, image_size=32)
    x = torch.randn(32, 3, 32, 32)
    output = model(x)
    assert output.shape == (32, 10)
