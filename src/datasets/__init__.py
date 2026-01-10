# src/datasets/__init__.py

from .cifar10 import build_cifar10_dataloaders, CIFAR10Transforms


def build_dataloader(cfg):
    dataset_cfg = cfg["dataset"]

    return build_cifar10_dataloaders(
            data_root = dataset_cfg["data_root"],
            val_ratio = dataset_cfg["val_ratio"],
            batch_size = dataset_cfg["batch_size"],
            num_workers = dataset_cfg["num_workers"],
            )


__all__ = [
        "build_dataloader",
        "CIFAR10Transforms",
        ]
