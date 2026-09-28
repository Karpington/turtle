from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import TurtleDataset
from src.model import TurtleCNN

import torch



transform = transforms.Compose([
    transforms.Resize((192, 320)),
    transforms.ToTensor(),
])


def check_csv_paths(csv_path, dataset_root):
    dataset = TurtleDataset(
        csv_file=csv_path,
        dataset_root=dataset_root,
        transform=None
    )

    missing_files = []

    for index in range(len(dataset.data)):
        row = dataset.data.iloc[index]
        image_path = Path(dataset_root) / row["path"]

        if not image_path.exists():
            missing_files.append(image_path)

    return missing_files


def create_loader(csv_path, dataset_root, batch_size=4, shuffle=False):
    dataset = TurtleDataset(
        csv_file=csv_path,
        dataset_root=dataset_root,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataset, loader


def main():
    dataset_root = "dataset"

    train_csv = "dataset/splits/train.csv"
    val_csv = "dataset/splits/val.csv"
    test_csv = "dataset/splits/test.csv"

    csv_files = [
        train_csv,
        val_csv,
        test_csv
    ]

    # Prüfen, ob alle Bilder existieren
    for csv_path in csv_files:
        missing_files = check_csv_paths(
            csv_path,
            dataset_root
        )

        if missing_files:
            print(f"\nFehlende Dateien in {csv_path}:")

            for image_path in missing_files:
                print(image_path)

            return

    print("Alle Bildpfade sind gültig.\n")

    train_dataset, train_loader = create_loader(
        csv_path=train_csv,
        dataset_root=dataset_root,
        batch_size=4,
        shuffle=True
    )

    val_dataset, val_loader = create_loader(
        csv_path=val_csv,
        dataset_root=dataset_root,
        batch_size=4,
        shuffle=False
    )

    test_dataset, test_loader = create_loader(
        csv_path=test_csv,
        dataset_root=dataset_root,
        batch_size=4,
        shuffle=False
    )

    print("Train Bilder:", len(train_dataset))
    print("Val Bilder:", len(val_dataset))
    print("Test Bilder:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("\nBatch Bilder:", images.shape)
    print("Batch Labels:", labels.shape)
    print("Labels:", labels)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TurtleCNN().to(device)

    images = images.to(device)

    outputs = model(images)

    print("\nCNN Input:", images.shape)
    print("CNN Output:", outputs.shape)
    print("Outputs:")
    print(outputs)


if __name__ == "__main__":
    main()