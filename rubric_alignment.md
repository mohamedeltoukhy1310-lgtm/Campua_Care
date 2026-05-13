# Rubric Alignment — CampusCare AI Big Data

| Rubric item | How the project satisfies it |
|---|---|
| Problem Understanding & Task Framing | Defines a real university support-routing problem with clear labels, constraints, and human-in-the-loop safety framing. |
| Model Selection & Architectural Justification | Includes TF-IDF baseline, TextCNN, BiLSTM, and optional DistilBERT; explains why each model matches the task. |
| Implementation Quality & Technical Correctness | Contains structured Python modules for configuration, preprocessing, dataset generation, training, evaluation, plots, and app demo. |
| Use of Deep Learning Concepts | Covers embeddings, CNN filters over text, recurrent sequence modeling, class-weighted cross-entropy, AdamW, dropout, and optional Transformer attention. |
| Experimentation & Comparative Analysis | Compares baseline, TextCNN, and BiLSTM on standard test and hard challenge split; saves metrics, confusion matrices, and mistakes. |
| Innovation & Intellectual Contribution | Reframes support-message classification as CampusCare AI, creates a 60,000-row custom dataset, and adds an ambiguous challenge set. |
| Interpretation, Error Analysis & Reflection | Report explains why baseline beats deep models on challenge split and discusses negation, label overlap, ambiguity, and ethical risks. |
| Reproducibility, Notebook Quality & Communication | Provides reproducible scripts, notebook, experiment log, dataset card, README, outputs, plots, and saved models. |

## Extra notes for presentation

- Do not claim the dataset is clinical or real student data.
- Say clearly that it is a large custom synthetic dataset for course experimentation.
- Explain that the challenge set is more valuable than the normal test split for understanding failures.
- Each team member should own and explain a different part/model.
