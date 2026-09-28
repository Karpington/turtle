from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((192, 320)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15
        ),
        transforms.ToTensor(),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((192, 320)),
        transforms.ToTensor(),
    ])


def create_weighted_sampler(dataset):
    labels = dataset.data["label"].astype(int).tolist()

    class_counts = Counter(labels)

    class_weights = {
        label: 1.0 / count
        for label, count in class_counts.items()
    }

    sample_weights = [
        class_weights[label]
        for label in labels
    ]

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler