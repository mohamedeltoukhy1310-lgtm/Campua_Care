from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"

BIG_DATA_PATH = DATA_DIR / "campuscare_large_dataset.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_campuscare_feedback.csv"
FULL_TRAIN_PATH = DATA_DIR / "mental_heath_unbanlanced.csv"
FULL_TEST_PATH = DATA_DIR / "mental_health_combined_test.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_campuscare.csv"
CHALLENGE_DATA_PATH = DATA_DIR / "campuscare_challenge_test.csv"

LABELS = ["normal", "academic_stress", "emotional_distress", "urgent_human_review"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

PUBLIC_LABEL_MAP = {
    "Normal": "normal",
    "Anxiety": "academic_stress",
    "Depression": "emotional_distress",
    "Suicidal": "urgent_human_review",
}

RANDOM_STATE = 42
MAX_VOCAB_SIZE = 15000
MAX_LEN = 96
BATCH_SIZE = 128
