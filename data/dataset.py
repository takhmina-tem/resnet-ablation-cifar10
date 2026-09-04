import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(augment=True):
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_dir, seed, batch_size=128, val_fraction=0.1, augment=True, num_workers=2):
    train_tf, eval_tf = get_transforms(augment)

    train_raw = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    train_raw_eval = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    labels = np.array(train_raw.targets)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=val_fraction, stratify=labels, random_state=seed)

    train_set = Subset(train_raw, train_idx)
    val_set = Subset(train_raw_eval, val_idx)
    # same images, no augmentation -- this is the fit measurement
    train_clean_set = Subset(train_raw_eval, train_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True, pin_memory=True)
    train_clean_loader = DataLoader(train_clean_set, batch_size=256, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, train_clean_loader, val_loader, test_loader
