"""Iris Flower Classification using K-Nearest Neighbors (KNN) and Scikit-learn."""

import sys
from pathlib import Path

import matplotlib

if "--no-plot" in sys.argv:
    matplotlib.use("Agg")  # no pop-up window; still saves the image

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).parent / "Iris.csv"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"


def load_data(path):
    """Load Iris.csv and do basic cleaning."""
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\nDownload Iris.csv from Kaggle and place it next to this script.")

    df = pd.read_csv(path)
    df = df.drop(columns=["Id"], errors="ignore")  # Id column is not a feature
    df = df.dropna().drop_duplicates()
    df[TARGET] = df[TARGET].str.replace("Iris-", "", regex=False).str.capitalize()
    return df


def train_model(df):
    """Split data, train KNN, and return model + test results."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Scaling puts all measurements on the same scale, which KNN needs
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate(model, X_test, y_test):
    """Print accuracy and show/save the confusion matrix."""
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
    plt.title("Iris KNN - Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to confusion_matrix.png")
    plt.show()


def ask_float(prompt):
    """Keep asking until the user enters a valid positive number."""
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("  Please enter a positive number (e.g. 5.1).")


def predict_loop(model):
    """Let the user enter measurements and get a predicted species."""
    print("\n--- Predict a flower species (measurements in cm) ---")
    while True:
        values = [ask_float(f"{name}: ") for name in
                  ["Sepal length", "Sepal width", "Petal length", "Petal width"]]
        sample = pd.DataFrame([values], columns=FEATURES)
        print(f"\n>>> Predicted species: Iris {model.predict(sample)[0]}\n")
        if input("Predict another? (y/n): ").strip().lower() != "y":
            print("Goodbye!")
            break


def main():
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} rows. Species: {', '.join(sorted(df[TARGET].unique()))}")
    model, X_test, y_test = train_model(df)
    evaluate(model, X_test, y_test)
    if "--no-predict" not in sys.argv:
        predict_loop(model)


if __name__ == "__main__":
    main()
