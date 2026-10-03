from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class TurtleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(192, 320)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.image_paths = sorted([
            path for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        relative_path = image_path.relative_to(self.image_dir)
        mask_path = self.mask_dir / relative_path.with_suffix(".png")

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Keine Maske für {image_path} gefunden. Erwartet: {mask_path}"
            )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask