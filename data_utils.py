import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    BIG_DATA_PATH,
    FULL_TEST_PATH,
    FULL_TRAIN_PATH,
    LABEL2ID,
    PUBLIC_LABEL_MAP,
    RANDOM_STATE,
    SAMPLE_DATA_PATH,
)


def clean_text(text: str) -> str:
    """Light cleaning that keeps useful punctuation while removing noisy artifacts."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _standardize_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Return the common columns needed by all models."""
    df = df.copy()
    if "status" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"status": "public_status"})
        df["label"] = df["public_status"].map(PUBLIC_LABEL_MAP)
    if "source" not in df.columns:
        df["source"] = source_name
    keep = [c for c in ["text", "label", "source", "message_id", "channel", "student_year", "severity_score", "is_synthetic"] if c in df.columns]
    return df[keep]


def load_project_data(use_full_if_available: bool = True) -> pd.DataFrame:
    """Load data in priority order: external full files, included big dataset, then small sample."""
    frames = []

    if use_full_if_available and FULL_TRAIN_PATH.exists():
        frames.append(_standardize_frame(pd.read_csv(FULL_TRAIN_PATH), "hf_train"))
    if use_full_if_available and FULL_TEST_PATH.exists():
        frames.append(_standardize_frame(pd.read_csv(FULL_TEST_PATH), "hf_test"))

    if not frames and BIG_DATA_PATH.exists():
        frames.append(_standardize_frame(pd.read_csv(BIG_DATA_PATH), "included_big_dataset"))

    if not frames:
        frames.append(_standardize_frame(pd.read_csv(SAMPLE_DATA_PATH), "included_seed_sample"))

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df = df[df["label"].isin(LABEL2ID.keys())]
    df["label_id"] = df["label"].map(LABEL2ID)
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().map(len)
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, test_size: float = 0.20, val_size: float = 0.10):
    """Stratified train/validation/test split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=RANDOM_STATE
    )
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["label"], random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+|<url>|<user>|[!?]", text.lower())


def build_vocab(texts: Iterable[str], max_vocab_size: int = 15000, min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.most_common(max_vocab_size - len(vocab)):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 96) -> List[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in simple_tokenize(text)]
    ids = ids[:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


def make_arrays(df: pd.DataFrame, vocab: Dict[str, int], max_len: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([encode_text(t, vocab, max_len) for t in df["text"]], dtype=np.int64)
    y = df["label_id"].to_numpy(dtype=np.int64)
    return x, y
