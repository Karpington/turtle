from pathlib import Path

from PIL import Image
import numpy as np


MASKS_DIR = Path("dataset/masks")


def main():
    mask_paths = sorted(MASKS_DIR.rglob("*.png"))

    positive_pixels = 0
    negative_pixels = 0
    positive_masks = 0
    empty_masks = 0

    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L"))

        positive = np.sum(mask > 127)
        negative = mask.size - positive

        positive_pixels += positive
        negative_pixels += negative

        if positive > 0:
            positive_masks += 1
        else:
            empty_masks += 1

    total_pixels = positive_pixels + negative_pixels

    print("Masken insgesamt:", len(mask_paths))
    print("Masken mit Schildkröte:", positive_masks)
    print("Leere Masken:", empty_masks)
    print()
    print("Schildkröten-Pixel:", positive_pixels)
    print("Hintergrund-Pixel:", negative_pixels)

    if total_pixels > 0:
        positive_percentage = positive_pixels / total_pixels * 100
        print(f"Schildkrötenanteil: {positive_percentage:.4f} %")

    if positive_pixels > 0:
        pos_weight = negative_pixels / positive_pixels
        print(f"Empfohlenes theoretisches POS_WEIGHT: {pos_weight:.2f}")


if __name__ == "__main__":
    main()