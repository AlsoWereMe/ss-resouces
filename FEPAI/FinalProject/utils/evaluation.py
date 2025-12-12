import os
import pandas as pd
from sklearn.metrics import f1_score


def load_true_labels(csv_path: str) -> list[int]:
    df = pd.read_csv(csv_path)
    return df["label"].astype(int).tolist()


def load_pred_labels(pred_path: str) -> list[int]:
    with open(pred_path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip() != ""]


def eval_f1(csv_path: str, pred_path: str) -> float:
    y_true = load_true_labels(csv_path)
    y_pred = load_pred_labels(pred_path)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred)} "
            f"(csv={csv_path}, pred={pred_path})"
        )

    return f1_score(y_true, y_pred)


def main():
    test_short_csv = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/data/processed/test_short.csv"
    test_long_csv = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/data/processed/test_long.csv"
    pred_dir = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/predictions"

    tasks = [
        ("lr", "short", test_short_csv, os.path.join(pred_dir, "lr_short_pred.txt")),
        ("lr", "long",  test_long_csv,  os.path.join(pred_dir, "lr_long_pred.txt")),
        ("svm", "short", test_short_csv, os.path.join(pred_dir, "svm_short_pred.txt")),
        ("svm", "long",  test_long_csv,  os.path.join(pred_dir, "svm_long_pred.txt")),
        ("mnb", "short", test_short_csv, os.path.join(pred_dir, "mnb_short_pred.txt")),
        ("mnb", "long",  test_long_csv,  os.path.join(pred_dir, "mnb_long_pred.txt")),
    ]

    # store results: {model: {short: f1, long: f1}}
    results = {}

    for model, split, csv_path, pred_path in tasks:
        f1 = eval_f1(csv_path, pred_path)
        print(f"{model} on test_{split}: F1 = {f1:.4f}")

        results.setdefault(model, {})[split] = f1

    print("\n=== Drop (F1_short - F1_long) ===")
    for model, scores in results.items():
        drop = scores["short"] - scores["long"]
        print(f"{model}: Drop = {drop:.4f}")


if __name__ == "__main__":
    main()
