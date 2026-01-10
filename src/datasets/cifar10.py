# src/datasets/cifar10.py

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T


class CIFAR10Transforms:
    @classmethod
    def train(cls):
        return T.Compose([
            T.Resize((32, 32)),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    @classmethod
    def test(cls):
        return T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

def build_cifar10_datasets(data_root, val_ratio):

    # 官方只提供了train数据集,所以需要split为train和val
    full_train_set = datasets.CIFAR10(
            root = data_root,
            train = True,
            download = False,
            transform = CIFAR10Transforms.train(),
            )

    total_size = len(full_train_set)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_set, val_set = random_split(
            full_train_set,
            [train_size, val_size],
            )

    test_set = datasets.CIFAR10(
            root = data_root,
            train = False,
            download = False,
            transform = CIFAR10Transforms.test(),
            )

    return train_set, val_set, test_set


def build_cifar10_dataloaders(data_root, val_ratio, batch_size, num_workers):

    train_set, val_set, test_set = build_cifar10_datasets(
            data_root = data_root,
            val_ratio = val_ratio,
            )

    train_loader = DataLoader(
            train_set,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = True,
            pin_memory = True,
            )

    val_loader = DataLoader(
            val_set,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = False,
            pin_memory = True,
            )

    test_loader = DataLoader(
            test_set,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = False,
            pin_memory = True,
            )

    return train_loader, val_loader, test_loader
