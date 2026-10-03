from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.segmentation_model import TurtleSegmentationModel


MODEL_PATH = "models/turtle_segmentation_best.pth"
IMAGE_PATH = "dataset/raw/2026-09-25/2026-09-25_11-48-02.jpg"

OUTPUT_DIR = Path("results/segmentation")
IMAGE_SIZE = (288, 480)
THRESHOLD = 0.5


def load_model(device):
    model = TurtleSegmentationModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def prepare_image(image_path, device):
    original_image = Image.open(image_path).convert("RGB")

    image = TF.resize(original_image, IMAGE_SIZE)
    image = TF.to_tensor(image)
    image = image.unsqueeze(0).to(device)

    return original_image, image


def predict_mask(model, image):
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities >= THRESHOLD).float()

    return prediction[0, 0].cpu().numpy(), probabilities[0, 0].cpu().numpy()


def create_overlay(original_image, mask):
    original = np.array(original_image).astype(np.float32)

    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image = mask_image.resize(original_image.size, Image.Resampling.NEAREST)
    mask = np.array(mask_image) > 0

    overlay = original.copy()
    overlay[mask, 0] = 255
    overlay[mask, 1] *= 0.4
    overlay[mask, 2] *= 0.4

    overlay = (0.7 * original + 0.3 * overlay).clip(0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_path = Path(IMAGE_PATH)

    if not image_path.exists():
        print(f"Bild nicht gefunden: {image_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(device)
    original_image, image = prepare_image(image_path, device)

    mask, probabilities = predict_mask(model, image)

    mask_output = Image.fromarray((mask * 255).astype(np.uint8))
    mask_output = mask_output.resize(original_image.size, Image.Resampling.NEAREST)

    overlay = create_overlay(original_image, mask)

    mask_path = OUTPUT_DIR / f"{image_path.stem}_mask.png"
    overlay_path = OUTPUT_DIR / f"{image_path.stem}_overlay.png"

    mask_output.save(mask_path)
    overlay.save(overlay_path)

    detected_pixels = int(mask.sum())
    max_probability = float(probabilities.max())

    print(f"Erkannte Schildkröten-Pixel: {detected_pixels}")
    print(f"Maximale Wahrscheinlichkeit: {max_probability * 100:.2f} %")

    if detected_pixels > 0:
        print("Ergebnis: Schildkröte erkannt")
    else:
        print("Ergebnis: Keine Schildkröte erkannt")

    print(f"Maske gespeichert: {mask_path}")
    print(f"Overlay gespeichert: {overlay_path}")


if __name__ == "__main__":
    main()