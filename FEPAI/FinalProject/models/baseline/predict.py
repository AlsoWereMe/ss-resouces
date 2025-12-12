import os
import joblib
import pandas as pd


def load_texts(path: str):
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist()


def infer_and_save(vectorizer, clf, x_path: str, out_path: str) -> None:
    X = load_texts(x_path)
    X_tfidf = vectorizer.transform(X)
    y_pred = clf.predict(X_tfidf)

    with open(out_path, "w") as f:
        for label in y_pred:
            f.write(str(label) + "\n")


def main():
    model_dir = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/results"
    test_short_path = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/data/processed/test_short.csv"
    test_long_path = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/data/processed/test_long.csv"

    output_dir = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Load vectorizer once
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf.joblib"))

    # Run inference for both short and long test sets
    splits = {
        "short": test_short_path,
        "long": test_long_path,
    }

    for name in ("lr", "svm", "mnb"):
        clf = joblib.load(os.path.join(model_dir, f"{name}.joblib"))

        for split_name, csv_path in splits.items():
            out_path = os.path.join(output_dir, f"{name}_{split_name}_pred.txt")
            infer_and_save(vectorizer, clf, csv_path, out_path)
            print(f"[{name}] {split_name} predictions saved to {out_path}")


if __name__ == "__main__":
    main()
