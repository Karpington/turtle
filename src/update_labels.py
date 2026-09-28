from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("dataset")
RAW_DIR = DATASET_ROOT / "raw"
LABELS_CSV = DATASET_ROOT / "labels" / "labels.csv"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def find_images():
    images = []

    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_path = path.relative_to(DATASET_ROOT)
            images.append(relative_path.as_posix())

    return images


def load_labels():
    if LABELS_CSV.exists():
        return pd.read_csv(
            LABELS_CSV,
            dtype={"path": str}
        )

    return pd.DataFrame(
        columns=["path", "label"]
    )


def main():
    LABELS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    images = find_images()
    labels_df = load_labels()

    existing_paths = set(
        labels_df["path"].astype(str)
    )

    new_rows = []

    for image_path in images:
        if image_path not in existing_paths:
            new_rows.append({
                "path": image_path,
                "label": ""
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows)

        labels_df = pd.concat(
            [labels_df, new_df],
            ignore_index=True
        )

    # Nach Pfad sortieren.
    # Bei Dateinamen wie 2026-09-27_15-30-00.jpg
    # ergibt das automatisch die richtige zeitliche Reihenfolge.
    labels_df = labels_df.sort_values(
        by="path"
    ).reset_index(drop=True)

    labels_df.to_csv(
        LABELS_CSV,
        index=False
    )

    print(f"Neue Bilder: {len(new_rows)}")
    print(f"Bilder insgesamt: {len(labels_df)}")


if __name__ == "__main__":
    main()