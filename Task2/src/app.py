import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import extract_features

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

def train_model():
    df = pd.read_csv(DATA_PATH)
    label_col = find_label_column(df)

    y = normalize_label(df[label_col])
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="raise")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    feature_names = list(X.columns)
    return model, feature_names

def main():
    model, feature_names = train_model()

    print("Model is trained. Paste an email text below. End with an empty line.")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    text = "\n".join(lines)

    feats = extract_features(text, feature_names)
    x_row = pd.DataFrame([feats], columns=feature_names)

    pred = model.predict(x_row)[0]
    proba = model.predict_proba(x_row)[0][1]  # probability of spam class (1)

    label = "SPAM" if pred == 1 else "LEGITIMATE"
    print(f"\nPrediction: {label}")
    print(f"Spam probability: {proba:.4f}")

if __name__ == "__main__":
    main()
