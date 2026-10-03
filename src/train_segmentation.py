from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.segmentation_dataset import TurtleSegmentationDataset
from src.segmentation_model import TurtleSegmentationModel


DATASET_ROOT = "dataset"
MODEL_PATH = "models/turtle_segmentation_best.pth"

IMAGE_SIZE = (288, 480)
BATCH_SIZE = 4
LEARNING_RATE = 0.0002
EPOCHS = 150
VAL_RATIO = 0.2
POS_WEIGHT = 40.0

EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 6
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 0.000001


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (probabilities * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probabilities.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight, device):
        super().__init__()

        weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss


def positive_dice_score(logits, targets, threshold=0.5, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    positive_images = targets.flatten(1).sum(dim=1) > 0

    if positive_images.sum() == 0:
        return None

    predictions = predictions[positive_images].flatten(1)
    targets = targets[positive_images].flatten(1)

    intersection = (predictions * targets).sum(dim=1)
    dice = (2 * intersection + smooth) / (
        predictions.sum(dim=1) + targets.sum(dim=1) + smooth
    )

    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    positive_images = targets.flatten(1).sum(dim=1) > 0

    if positive_images.sum() == 0:
        return None

    predictions = predictions[positive_images].flatten(1)
    targets = targets[positive_images].flatten(1)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    metric_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        dice = positive_dice_score(outputs.detach(), masks)
        iou = iou_score(outputs.detach(), masks)

        if dice is not None and iou is not None:
            total_dice += dice
            total_iou += iou
            metric_batches += 1

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / metric_batches if metric_batches > 0 else 0.0
    avg_iou = total_iou / metric_batches if metric_batches > 0 else 0.0

    return avg_loss, avg_dice, avg_iou


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    metric_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            dice = positive_dice_score(outputs, masks)
            iou = iou_score(outputs, masks)

            if dice is not None and iou is not None:
                total_dice += dice
                total_iou += iou
                metric_batches += 1

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / metric_batches if metric_batches > 0 else 0.0
    avg_iou = total_iou / metric_batches if metric_batches > 0 else 0.0

    return avg_loss, avg_dice, avg_iou


def main():
    Path("models").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = TurtleSegmentationDataset(
        image_dir=f"{DATASET_ROOT}/raw",
        mask_dir=f"{DATASET_ROOT}/masks",
        image_size=IMAGE_SIZE
    )

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)

    # Für den aktuellen Test reicht random_split.
    # Später sollten Train und Validation nach ganzen Tagen getrennt werden.
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Train Bilder:", len(train_dataset))
    print("Val Bilder:", len(val_dataset))
    print("Bildgröße:", IMAGE_SIZE)
    print("Positive Pixel Weight:", POS_WEIGHT)

    model = TurtleSegmentationModel().to(device)
    criterion = CombinedLoss(POS_WEIGHT, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE
    )

    best_val_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_dice, val_iou = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"LR: {current_lr:.7f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0

            torch.save(model.state_dict(), MODEL_PATH)

            print(f"Bestes Modell gespeichert. Val Dice: {best_val_dice:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"Keine Verbesserung: "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly Stopping: Val Dice hat sich nicht weiter verbessert.")
            break

    print("\nTraining abgeschlossen.")
    print(f"Bester Val Dice: {best_val_dice:.4f}")
    print(f"Bestes Modell: {MODEL_PATH}")


if __name__ == "__main__":
    main()