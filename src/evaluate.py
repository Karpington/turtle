import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.dataset import TurtleDataset
from src.model import TurtleCNN


def create_dataloader(csv_path, dataset_root, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((192, 320)),
        transforms.ToTensor(),
    ])

    dataset = TurtleDataset(
        csv_file=csv_path,
        dataset_root=dataset_root,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return loader


def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= threshold).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(
                predictions.squeeze(1).cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )  

    return accuracy, precision, recall, f1, matrix


def main():
    dataset_root = "dataset"
    test_csv = "dataset/splits/test.csv"

    model_path = "models/turtle_cnn_best.pth"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    test_loader = create_dataloader(
        test_csv,
        dataset_root,
        batch_size=4
    )

    model = TurtleCNN().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    accuracy, precision, recall, f1, matrix = evaluate_model(
        model,
        test_loader,
        device
    )

    print("\nErgebnisse auf dem Testset:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()