import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

DATA_PATH = "../data/e_bakuradze25_23456_csv.csv"

def find_label_column(df: pd.DataFrame) -> str:
    candidates = ["is_spam", "class", "label", "spam", "target", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Label column not found. Columns: {list(df.columns)}")

def normalize_label(series: pd.Series):
    """
    Converts label to 0/1:
      spam -> 1, legitimate -> 0
    """
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "spam": 1,
        "legitimate": 0,
        "ham": 0,
        "0": 0,
        "1": 1
    }
    if set(s.unique()).issubset(set(mapping.keys())):
        return s.map(mapping).astype(int)
    # If already numeric but not in mapping
    try:
        return series.astype(int)
    except Exception:
        raise ValueError(f"Unrecognized label values: {series.unique()}")

def main():
    df = pd.read_csv(DATA_PATH)

    label_col = find_label_column(df)
    y = normalize_label(df[label_col])

    X = df.drop(columns=[label_col])

    # Ensure all features numeric
    X = X.apply(pd.to_numeric, errors="raise")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Logistic regression model pipeline (scaler + LR)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    # Coefficients
    lr = model.named_steps["lr"]
    coeffs = lr.coef_[0]
    intercept = lr.intercept_[0]

    print("=== Model coefficients ===")
    print(f"Intercept: {intercept}")
    print("Feature coefficients:")
    for name, coef in sorted(zip(X.columns, coeffs), key=lambda t: abs(t[1]), reverse=True):
        print(f"{name}: {coef}")

    # Evaluation
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Evaluation on 30% holdout ===")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.6f}")

if __name__ == "__main__":
    main()

