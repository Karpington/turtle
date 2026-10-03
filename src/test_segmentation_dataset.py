from torch.utils.data import DataLoader

from src.segmentation_dataset import TurtleSegmentationDataset
from src.segmentation_model import TurtleSegmentationModel
import torch

def main():
    dataset = TurtleSegmentationDataset(
        image_dir="dataset/raw",
        mask_dir="dataset/masks",
        image_size=(192, 320)
    )

    print("Anzahl Bilder:", len(dataset))

    image, mask = dataset[0]

    print("Bild:", image.shape)
    print("Maske:", mask.shape)
    print("Maskenwerte erstes Bild:", mask.unique())

    positive_masks = 0
    empty_masks = 0

    for index in range(len(dataset)):
        _, mask = dataset[index]

        if mask.max() > 0:
            positive_masks += 1
        else:
            empty_masks += 1

    print("\nMasken mit Schildkröte:", positive_masks)
    print("Leere Masken:", empty_masks)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    images, masks = next(iter(loader))

    print("\nBatch Bilder:", images.shape)
    print("Batch Masken:", masks.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TurtleSegmentationModel().to(device)
    images = images.to(device)

    outputs = model(images)

    print("\nDevice:", device)
    print("Model Input:", images.shape)
    print("Model Output:", outputs.shape)


if __name__ == "__main__":
    main()