from pathlib import Path
import json
import shutil

from PIL import Image, ImageDraw


RAW_DIR = Path("dataset/raw")
ANNOTATIONS_DIR = Path("dataset/annotations")
MASKS_DIR = Path("dataset/masks")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TURTLE_LABELS = {"turtle", "schildkroete", "schildkröte"}


def create_mask_from_json(json_path, image_size):
    with open(json_path, "r", encoding="utf-8") as file:
        annotation = json.load(file)

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    found_turtle = False

    for shape in annotation.get("shapes", []):
        label = shape.get("label", "").strip().lower()

        if label not in TURTLE_LABELS:
            continue

        points = [(int(x), int(y)) for x, y in shape["points"]]

        if len(points) >= 3:
            draw.polygon(points, fill=255)
            found_turtle = True

    return mask, found_turtle


def main():
    if MASKS_DIR.exists():
        shutil.rmtree(MASKS_DIR)

    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        path for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        print("Keine Bilder gefunden.")
        return

    with_turtle = 0
    empty = 0
    annotations_found = 0

    for image_path in image_paths:
        relative_path = image_path.relative_to(RAW_DIR)

        annotation_path = ANNOTATIONS_DIR / relative_path.with_suffix(".json")
        mask_path = MASKS_DIR / relative_path.with_suffix(".png")

        mask_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            image_size = image.size

        if annotation_path.exists():
            annotations_found += 1
            mask, found_turtle = create_mask_from_json(annotation_path, image_size)

            if found_turtle:
                with_turtle += 1
            else:
                empty += 1
                print(f"JSON ohne gültiges Turtle-Polygon: {annotation_path}")
        else:
            mask = Image.new("L", image_size, 0)
            empty += 1

        mask.save(mask_path)

    print()
    print(f"Bilder: {len(image_paths)}")
    print(f"Gefundene Annotationen: {annotations_found}")
    print(f"Masken mit Schildkröte: {with_turtle}")
    print(f"Leere Masken: {empty}")
    print(f"Masken insgesamt: {with_turtle + empty}")


if __name__ == "__main__":
    main()