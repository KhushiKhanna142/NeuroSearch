# Data loading utilities
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loader(split, batch_size, num_workers=2):
    """
    Get CIFAR-10 DataLoader.
    
    Parameters
    ----------
    split : str
        'train' for training set, 'val' for validation/test set.
    batch_size : int
    num_workers : int
    """
    # Standard CIFAR-10 normalization values
    norm_transform = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm_transform,
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            norm_transform,
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
