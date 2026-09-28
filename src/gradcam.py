from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import TurtleCNN


MODEL_PATH = "models/turtle_cnn_best.pth"

IMAGE_PATH = "dataset/raw/2026-09-26/2026-09-26_00-05-02.jpg"

OUTPUT_PATH = "results/gradcam.jpg"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor):
        self.model.zero_grad()

        output = self.model(image_tensor)

        score = output[0, 0]

        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()


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


def load_image(image_path, device):
    original_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((192, 320)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(original_image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    return original_image, image_tensor


def create_overlay(original_image, cam):
    image = np.array(original_image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cam = cv2.resize(
        cam,
        (image.shape[1], image.shape[0])
    )

    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        cam,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    image_path = Path(IMAGE_PATH)

    if not image_path.exists():
        print(f"Bild nicht gefunden: {image_path}")
        return

    Path("results").mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model(
        MODEL_PATH,
        device
    )

    original_image, image_tensor = load_image(
        image_path,
        device
    )

    # Letzte Conv-Schicht unseres CNNs
    target_layer = model.features[9]

    gradcam = GradCAM(
        model,
        target_layer
    )

    cam = gradcam.generate(
        image_tensor
    )

    overlay = create_overlay(
        original_image,
        cam
    )

    cv2.imwrite(
        OUTPUT_PATH,
        overlay
    )

    probability = torch.sigmoid(
        model(image_tensor)
    ).item()

    print(
        f"Schildkröte-Wahrscheinlichkeit: "
        f"{probability * 100:.2f} %"
    )

    print(
        f"Grad-CAM gespeichert unter: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()