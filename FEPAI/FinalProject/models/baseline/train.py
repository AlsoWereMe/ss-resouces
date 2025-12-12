import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


def load_train_data(path):
    df = pd.read_csv(path)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()
    return X, y


def main():
    train_path = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/data/processed/train.csv"
    model_dir = "/Users/alsowereme/workspace/ss-resouces/FEPAI/FinalProject/results"
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train = load_train_data(train_path)

    # TF-IDF feature extractor
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Models
    models = {
        "lr": LogisticRegression(max_iter=2000, solver="liblinear"),
        "svm": LinearSVC(),
        "mnb": MultinomialNB(alpha=1.0),
    }

    # Train and save
    for name, clf in models.items():
        clf.fit(X_train_tfidf, y_train)
        joblib.dump(clf, os.path.join(model_dir, f"{name}.joblib"))

    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf.joblib"))

    print("Training completed. Models saved.")


if __name__ == "__main__":
    main()
