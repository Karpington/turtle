from src.test_dataset import main as test_dataset_main
from src.train import main as train_main
from src.evaluate import main as evaluate_main
from src.predict import main as predict_main
from src.gradcam import main as gradcam_main


def main():
    print("\n=== 1. Dataset testen ===")
    test_dataset_main()

    print("\n=== 2. Modell trainieren ===")
    train_main()

    print("\n=== 3. Modell auswerten ===")
    evaluate_main()

    print("\n=== 4. Einzelbild vorhersagen ===")
    predict_main()

    print("\n=== 5. Grad-CAM erstellen ===")
    gradcam_main()


if __name__ == "__main__":
    main()