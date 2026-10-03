import torch
from torch.utils.data import DataLoader

from src.segmentation_dataset import TurtleSegmentationDataset
from src.segmentation_model import TurtleSegmentationModel


MODEL_PATH = "models/turtle_segmentation_best.pth"
IMAGE_SIZE = (288, 480)
BATCH_SIZE = 4
THRESHOLD = 0.5


def dice_score(prediction, target, smooth=1.0):
    prediction = prediction.flatten()
    target = target.flatten()

    intersection = (prediction * target).sum()

    dice = (2 * intersection + smooth) / (
        prediction.sum() + target.sum() + smooth
    )

    return dice.item()


def iou_score(prediction, target, smooth=1.0):
    prediction = prediction.flatten()
    target = target.flatten()

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = TurtleSegmentationDataset(
        image_dir="dataset/raw",
        mask_dir="dataset/masks",
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = TurtleSegmentationModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    positive_dice_scores = []
    positive_iou_scores = []

    positive_images = 0
    empty_images = 0

    empty_correct = 0
    empty_false_positive = 0

    detected_positive_images = 0
    missed_positive_images = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= THRESHOLD).float()

            for prediction, mask in zip(predictions, masks):
                target_has_turtle = mask.sum() > 0
                prediction_has_turtle = prediction.sum() > 0

                if target_has_turtle:
                    positive_images += 1

                    dice = dice_score(prediction, mask)
                    iou = iou_score(prediction, mask)

                    positive_dice_scores.append(dice)
                    positive_iou_scores.append(iou)

                    if prediction_has_turtle:
                        detected_positive_images += 1
                    else:
                        missed_positive_images += 1

                else:
                    empty_images += 1

                    if prediction_has_turtle:
                        empty_false_positive += 1
                    else:
                        empty_correct += 1

    avg_positive_dice = (
        sum(positive_dice_scores) / len(positive_dice_scores)
        if positive_dice_scores else 0.0
    )

    avg_positive_iou = (
        sum(positive_iou_scores) / len(positive_iou_scores)
        if positive_iou_scores else 0.0
    )

    detection_rate = (
        detected_positive_images / positive_images
        if positive_images > 0 else 0.0
    )

    empty_accuracy = (
        empty_correct / empty_images
        if empty_images > 0 else 0.0
    )

    print()
    print("=== Segmentierungs-Auswertung ===")
    print()
    print(f"Bilder mit Schildkröte: {positive_images}")
    print(f"Leere Bilder: {empty_images}")
    print()

    print(f"Positive Dice: {avg_positive_dice:.4f}")
    print(f"Positive IoU: {avg_positive_iou:.4f}")
    print()

    print(f"Schildkröte erkannt: {detected_positive_images}/{positive_images}")
    print(f"Schildkröte übersehen: {missed_positive_images}/{positive_images}")
    print(f"Erkennungsrate: {detection_rate * 100:.2f} %")
    print()

    print(f"Leere Bilder korrekt erkannt: {empty_correct}/{empty_images}")
    print(f"False Positives auf leeren Bildern: {empty_false_positive}/{empty_images}")
    print(f"Genauigkeit bei leeren Bildern: {empty_accuracy * 100:.2f} %")

    print()
    print("=== Bedeutung ===")
    print(
        "Positive Dice misst, wie gut die vorhergesagte Schildkrötenfläche "
        "mit der echten markierten Fläche übereinstimmt."
    )
    print(
        "Positive IoU misst die Überlappung zwischen Vorhersage und echter "
        "Schildkrötenmaske. Je näher an 1.0, desto besser."
    )
    print(
        "Die Erkennungsrate zeigt, auf wie vielen Bildern mit Schildkröte "
        "überhaupt Schildkrötenpixel vorhergesagt wurden."
    )
    print(
        "Die Genauigkeit bei leeren Bildern zeigt, wie oft das Modell korrekt "
        "keine Schildkröte erkannt hat."
    )

    print()
    print("Wichtig:")
    print(
        "Diese Werte sind nur dann eine echte Qualitätsmessung, wenn die Bilder "
        "nicht bereits im Training verwendet wurden."
    )


if __name__ == "__main__":
    main()