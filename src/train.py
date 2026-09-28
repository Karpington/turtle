import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.dataset import TurtleDataset
from src.model import TurtleCNN
from src.data_utils import (
    get_train_transform,
    get_eval_transform,
    create_weighted_sampler
)


# True:
#   Seltene Klassen werden beim Training häufiger gezogen.
#   Sinnvoll, solange du z.B. deutlich weniger echte 0er-Bilder hast.
#
# False:
#   Normales Training mit shuffle=True.
#   Das kannst du verwenden, sobald du genug echte Bilder beider Klassen hast.
USE_WEIGHTED_SAMPLER = True


def create_train_loader(csv_path, dataset_root, batch_size):
    train_dataset = TurtleDataset(
        csv_file=csv_path,
        dataset_root=dataset_root,
        transform=get_train_transform()
    )

    if USE_WEIGHTED_SAMPLER:
        sampler = create_weighted_sampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )

        print("WeightedRandomSampler aktiviert.")

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        print("Normales Shuffle aktiviert.")

    return train_loader


def create_val_loader(csv_path, dataset_root, batch_size):
    val_dataset = TurtleDataset(
        csv_file=csv_path,
        dataset_root=dataset_root,
        transform=get_eval_transform()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return val_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        labels = labels.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            labels = labels.unsqueeze(1)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    dataset_root = "dataset"

    train_csv = "dataset/splits/train.csv"
    val_csv = "dataset/splits/val.csv"

    batch_size = 4
    learning_rate = 0.001
    epochs = 30

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader = create_train_loader(
        train_csv,
        dataset_root,
        batch_size
    )

    val_loader = create_val_loader(
        val_csv,
        dataset_root,
        batch_size
    )

    model = TurtleCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "models/turtle_cnn_best.pth"
            )

            print("Bestes Modell gespeichert.")

    print("Training abgeschlossen.")


if __name__ == "__main__":
    main()