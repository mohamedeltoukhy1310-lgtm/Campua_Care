"""
Download the public dataset files used by the full project.
Run from the project root:
    python src/download_hf_dataset.py

The project will still run without these files using data/sample_campuscare_feedback.csv,
but the full report should be based on the downloaded dataset.
"""
from pathlib import Path

FILES = [
    "mental_heath_unbanlanced.csv",
    "mental_health_combined_test.csv",
    "mental_health_feature_engineered.csv",
]
DATASET = "ourafla/Mental-Health_Text-Classification_Dataset"


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    for name in FILES:
        path = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=name, local_dir=data_dir)
        print(f"Downloaded: {path}")


if __name__ == "__main__":
    main()
