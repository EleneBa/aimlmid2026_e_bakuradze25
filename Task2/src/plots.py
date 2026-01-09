import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

DATA_PATH = "../data/e_bakuradze25_23456_csv.csv"

def find_label_column(df: pd.DataFrame) -> str:
    for c in ["is_spam", "class", "label", "spam", "target", "y"]:
        if c in df.columns:
            return c
    raise ValueError("Label column not found.")

def normalize_label(series: pd.Series):
    s = series.astype(str).str.strip().str.lower()
    mapping = {"spam": 1, "legitimate": 0, "ham": 0, "0": 0, "1": 1}
    if set(s.unique()).issubset(set(mapping.keys())):
        return s.map(mapping).astype(int)
    return series.astype(int)

def plot_class_distribution(y: pd.Series, out_path: str):
    counts = y.value_counts().sort_index()  # 0,1
    labels = ["Legitimate (0)", "Spam (1)"]

    plt.figure()
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Class Distribution: Legitimate vs Spam")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, out_path: str):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks([0, 1], ["Legitimate", "Spam"])
    plt.yticks([0, 1], ["Legitimate", "Spam"])

    # annotate
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()

def main():
    df = pd.read_csv(DATA_PATH)
    label_col = find_label_column(df)

    y = normalize_label(df[label_col])
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="raise")

    # Plot 1: class distribution
    plot_class_distribution(y, "../figures/class_distribution.png")

    # Train / eval for confusion matrix plot
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "../figures/confusion_matrix.png")

if __name__ == "__main__":
    main()
