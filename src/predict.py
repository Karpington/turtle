from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import TurtleCNN


MODEL_PATH = "models/turtle_cnn_best.pth"

IMAGE_PATH = "dataset/raw/2026-09-26/2026-09-27_00-01-02.jpg"

THRESHOLD = 0.5


def load_model(model_path, device):
    model = TurtleCNN().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    return model


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((192, 320)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    # Aus [3, 192, 320] wird [1, 3, 192, 320]
    image = image.unsqueeze(0)

    return image


def predict(model, image, device):
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

        probability = torch.sigmoid(output)

    return probability.item()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    image_path = Path(IMAGE_PATH)

    if not image_path.exists():
        print(f"Bild wurde nicht gefunden: {image_path}")
        return

    model = load_model(
        MODEL_PATH,
        device
    )

    image = load_image(
        image_path
    )

    probability = predict(
        model,
        image,
        device
    )

    print(f"\nBild: {image_path}")
    print(
        f"Schildkröte-Wahrscheinlichkeit: "
        f"{probability * 100:.2f} %"
    )

    if probability >= THRESHOLD:
        print("Vorhersage: Schildkröte erkannt")
    else:
        print("Vorhersage: Keine Schildkröte erkannt")


if __name__ == "__main__":
    main()