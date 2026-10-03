from pathlib import Path
import shutil


RAW_DIR = Path("dataset/raw")
ANNOTATIONS_DIR = Path("dataset/annotations")


def main():
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(RAW_DIR.rglob("*.json"))

    if not json_files:
        print("Keine JSON-Dateien in dataset/raw gefunden.")
        return

    moved = 0
    replaced = 0

    for json_path in json_files:
        relative_path = json_path.relative_to(RAW_DIR)
        target_path = ANNOTATIONS_DIR / relative_path

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            target_path.unlink()
            replaced += 1

        shutil.move(str(json_path), str(target_path))

        print(f"{relative_path} -> annotations/{relative_path}")
        moved += 1

    print()
    print(f"Verschoben: {moved}")
    print(f"Bestehende Annotationen ersetzt: {replaced}")


if __name__ == "__main__":
    main()