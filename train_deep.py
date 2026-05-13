import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, CHALLENGE_DATA_PATH, ID2LABEL, LABELS, MAX_LEN, MAX_VOCAB_SIZE, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE
from data_utils import build_vocab, clean_text, load_project_data, make_arrays, make_splits
from models_torch import BiLSTMClassifier, TextCNN


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(x, y, batch_size=BATCH_SIZE, shuffle=False):
    ds = TensorDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro, np.array(all_labels), np.array(all_preds)


def train_one_model(model_name: str, model, train_loader, val_loader, device, class_weights, epochs=2, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_macro = -1
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_acc, val_macro, _, _ = evaluate(model, val_loader, device)
        history.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro,
        })
        print(f"{model_name} epoch {epoch}/{epochs} - loss={np.mean(losses):.4f} val_macro_f1={val_macro:.4f}")
        if val_macro > best_macro:
            best_macro = val_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap for fast CPU demos; 0 means use all training data.")
    parser.add_argument("--models", type=str, default="TextCNN,BiLSTM", help="Comma-separated: TextCNN,BiLSTM")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_project_data(use_full_if_available=True)
    train_df, val_df, test_df = make_splits(df)
    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(max(1, int(args.max_train_samples * len(x) / len(train_df))), random_state=RANDOM_STATE)
        ).reset_index(drop=True)

    split_summary = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "rows": [len(train_df), len(val_df), len(test_df)],
        "unique_labels": [train_df["label"].nunique(), val_df["label"].nunique(), test_df["label"].nunique()],
    })
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    vocab = build_vocab(train_df["text"], max_vocab_size=MAX_VOCAB_SIZE, min_freq=1)
    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    x_train, y_train = make_arrays(train_df, vocab, MAX_LEN)
    x_val, y_val = make_arrays(val_df, vocab, MAX_LEN)
    x_test, y_test = make_arrays(test_df, vocab, MAX_LEN)

    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_val, y_val)
    test_loader = make_loader(x_test, y_test)

    counts = np.bincount(y_train, minlength=len(LABELS))
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    all_specs = {
        "TextCNN": TextCNN(vocab_size=len(vocab), embed_dim=64, num_classes=len(LABELS), num_filters=48),
        "BiLSTM": BiLSTMClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=48, num_classes=len(LABELS)),
    }
    specs = {name: model for name, model in all_specs.items() if name in selected}
    if not specs:
        raise ValueError(f"No valid models selected: {args.models}")

    all_rows = []
    all_history = []
    for model_name, model in specs.items():
        print(f"\nTraining {model_name} on {len(train_df):,} train rows, validating on {len(val_df):,} rows")
        model = model.to(device)
        trained, hist = train_one_model(model_name, model, train_loader, val_loader, device, class_weights, epochs=args.epochs)
        all_history.append(hist)
        eval_items = [("test", test_df, test_loader)]
        if CHALLENGE_DATA_PATH.exists():
            challenge_df = pd.read_csv(CHALLENGE_DATA_PATH)
            challenge_df["text"] = challenge_df["text"].astype(str).map(clean_text)
            challenge_df["label_id"] = challenge_df["label"].map({label: i for i, label in enumerate(LABELS)})
            x_chal, y_chal = make_arrays(challenge_df, vocab, MAX_LEN)
            eval_items.append(("challenge", challenge_df, make_loader(x_chal, y_chal)))

        for split_name, split_df, loader in eval_items:
            split_acc, split_macro, y_true, y_pred = evaluate(trained, loader, device)
            all_rows.append({"model": model_name, "split": split_name, "accuracy": split_acc, "macro_f1": split_macro})

            pred_labels = [ID2LABEL[int(i)] for i in y_pred]
            true_labels = [ID2LABEL[int(i)] for i in y_true]
            pd.DataFrame(classification_report(true_labels, pred_labels, labels=LABELS, output_dict=True, zero_division=0)).transpose().to_csv(
                OUTPUT_DIR / f"{model_name.lower()}_{split_name}_classification_report.csv"
            )
            pd.DataFrame(confusion_matrix(true_labels, pred_labels, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
                OUTPUT_DIR / f"{model_name.lower()}_{split_name}_confusion_matrix.csv"
            )
            mistakes = split_df.copy()
            mistakes["pred"] = pred_labels
            mistakes[mistakes["label"] != mistakes["pred"]].head(200).to_csv(OUTPUT_DIR / f"{model_name.lower()}_{split_name}_mistakes.csv", index=False)

        torch.save({"state_dict": trained.state_dict(), "vocab": vocab, "labels": LABELS, "max_len": MAX_LEN}, MODEL_DIR / f"{model_name.lower()}.pt")

    # Merge with any previous deep outputs so separate CPU runs can be combined.
    hist_df = pd.concat(all_history, ignore_index=True)
    hist_path = OUTPUT_DIR / "deep_training_history.csv"
    if hist_path.exists():
        old_hist = pd.read_csv(hist_path)
        hist_df = pd.concat([old_hist[~old_hist["model"].isin(hist_df["model"].unique())], hist_df], ignore_index=True)
    hist_df.to_csv(hist_path, index=False)

    rows_df = pd.DataFrame(all_rows)
    summary_path = OUTPUT_DIR / "deep_results_summary.csv"
    if summary_path.exists():
        old_rows = pd.read_csv(summary_path)
        rows_df = pd.concat([old_rows[~old_rows["model"].isin(rows_df["model"].unique())], rows_df], ignore_index=True)
    rows_df.to_csv(summary_path, index=False)
    print(rows_df)


if __name__ == "__main__":
    main()
