"""
Optional pretrained Transformer experiment.
Run after installing requirements and downloading the full dataset:
    python src/download_hf_dataset.py
    python src/train_transformer.py

For weak laptops, reduce MAX_SAMPLES or keep RUN_SMALL_DEBUG=True.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from config import ID2LABEL, LABEL2ID, LABELS, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE
from data_utils import load_project_data, make_splits

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 160
MAX_SAMPLES = None  # set to 3000 for faster CPU experiments


def main():
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_project_data(use_full_if_available=True)
    if MAX_SAMPLES:
        df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), MAX_SAMPLES // len(LABELS)), random_state=RANDOM_STATE))
    train_df, val_df, test_df = make_splits(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]]).rename_column("label_id", "labels").map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]]).rename_column("label_id", "labels").map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label_id"]]).rename_column("label_id", "labels").map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={int(k): v for k, v in ID2LABEL.items()},
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "distilbert_checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=RANDOM_STATE,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(test_ds)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "distilbert_test_metrics.csv", index=False)
    trainer.save_model(str(MODEL_DIR / "distilbert_campuscare"))
    tokenizer.save_pretrained(str(MODEL_DIR / "distilbert_campuscare"))
    print(metrics)


if __name__ == "__main__":
    main()
