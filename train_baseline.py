import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from config import CHALLENGE_DATA_PATH, LABELS, MODEL_DIR, OUTPUT_DIR
from data_utils import clean_text, load_project_data, make_splits


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_project_data(use_full_if_available=True)
    train_df, val_df, test_df = make_splits(df)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
        ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    pipe.fit(train_df["text"], train_df["label"])

    rows = []
    eval_splits = [("val", val_df), ("test", test_df)]
    if CHALLENGE_DATA_PATH.exists():
        challenge_df = pd.read_csv(CHALLENGE_DATA_PATH)
        challenge_df["text"] = challenge_df["text"].astype(str).map(clean_text)
        eval_splits.append(("challenge", challenge_df))

    for split_name, split_df in eval_splits:
        preds = pipe.predict(split_df["text"])
        rows.append({
            "model": "TFIDF_SGD_LogReg_baseline",
            "split": split_name,
            "accuracy": accuracy_score(split_df["label"], preds),
            "macro_f1": f1_score(split_df["label"], preds, average="macro"),
        })
        report = classification_report(split_df["label"], preds, labels=LABELS, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / f"baseline_{split_name}_classification_report.csv")
        cm = pd.DataFrame(confusion_matrix(split_df["label"], preds, labels=LABELS), index=LABELS, columns=LABELS)
        cm.to_csv(OUTPUT_DIR / f"baseline_{split_name}_confusion_matrix.csv")
        mistakes = split_df.copy()
        mistakes["pred"] = preds
        mistakes[mistakes["label"] != mistakes["pred"]].head(300).to_csv(OUTPUT_DIR / f"baseline_{split_name}_mistakes.csv", index=False)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "baseline_results_summary.csv", index=False)
    joblib.dump(pipe, MODEL_DIR / "tfidf_sgd_logreg_baseline.joblib")
    print(pd.DataFrame(rows))
    print(f"Saved model to {MODEL_DIR / 'tfidf_sgd_logreg_baseline.joblib'}")


if __name__ == "__main__":
    main()
