"""Data loading utilities for common datasets."""

from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_mnist(
    data_dir: str = "./data",
    download: bool = True,
    normalize: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Get MNIST train and test datasets.

    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        normalize: Whether to normalize to mean=0.5, std=0.5

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    return train_dataset, test_dataset


def get_cifar10(
    data_dir: str = "./data",
    download: bool = True,
    normalize: bool = True,
    augment_train: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Get CIFAR-10 train and test datasets.

    Args:
        data_dir: Directory to store/load data
        download: Whether to download if not present
        normalize: Whether to normalize with CIFAR-10 statistics
        augment_train: Whether to apply data augmentation to training set

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # CIFAR-10 normalization statistics
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform_list = []
    if augment_train:
        train_transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_transform_list.append(transforms.ToTensor())
    if normalize:
        train_transform_list.append(transforms.Normalize(mean, std))

    test_transform_list = [transforms.ToTensor()]
    if normalize:
        test_transform_list.append(transforms.Normalize(mean, std))

    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_dataset, test_dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader from a dataset.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
