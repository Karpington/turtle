
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class TurtleDataset(Dataset):
    def __init__(self, csv_file, dataset_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self.dataset_root / row["path"]

        image = Image.open(image_path).convert("RGB")

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label