# CampusCare AI Big Data Project Report

## 1. Problem Understanding and Task Framing

Universities receive large numbers of anonymous student messages through advising portals, feedback forms, mobile apps, course pages, and email. During exam periods, staff may need to quickly distinguish between routine questions, academic stress, emotional distress, and messages that require immediate human review.

**Goal:** build a deep learning text-classification system that routes each student message into one of four categories:

1. `normal`
2. `academic_stress`
3. `emotional_distress`
4. `urgent_human_review`

This is not a clinical diagnosis system. The model is framed as a **human-in-the-loop triage assistant**. A trained human must review high-risk predictions before any action is taken.

## 2. Dataset Curation

This version uses a large custom dataset included inside the project:

- `data/campuscare_large_dataset.csv`: 60,000 messages.
- `data/campuscare_challenge_test.csv`: 1,200 hard/ambiguous messages for error analysis.

### 2.1 Class distribution

| Label | Rows | Percentage |
|---|---:|---:|
| normal | 24,000 | 40% |
| academic_stress | 18,000 | 30% |
| emotional_distress | 12,000 | 20% |
| urgent_human_review | 6,000 | 10% |

The distribution is intentionally imbalanced because real student-support systems usually receive many routine messages and fewer urgent messages.

### 2.2 Dataset design

The data is custom synthetic. It was generated with `src/generate_big_dataset.py` using:

- university course names,
- academic task types,
- support channels,
- message templates,
- wording noise,
- optional metadata,
- class-specific severity ranges.

The challenge set was added because normal random test splits from synthetic data can be too easy. It includes ambiguous messages such as:

- “I was stressed before the quiz, but I am okay now...” → normal,
- “I cannot go on with these deadlines, meaning academically...” → academic stress,
- “I do not plan to hurt myself, but I feel worthless...” → emotional distress,
- “I am writing about the exam, but I do not feel safe tonight...” → urgent review.

## 3. Preprocessing

Implemented in `src/data_utils.py`:

1. Load the big dataset automatically.
2. Clean line breaks and extra spaces.
3. Replace URLs with `<url>`.
4. Replace user mentions with `<user>`.
5. Drop empty texts and duplicate texts.
6. Keep only the four approved labels.
7. Create `label_id` for neural models.
8. Create text statistics: `text_length` and `word_count`.
9. Use stratified train/validation/test split.
10. Build vocabulary and pad/truncate to `MAX_LEN = 96` for PyTorch models.

## 4. Model Selection and Architectural Justification

### 4.1 TF-IDF + SGD Logistic Regression baseline

This is a strong classical NLP baseline. It is fast on large text datasets and helps check whether deep learning adds value. The baseline uses word unigrams and bigrams with a logistic-loss SGD classifier.

### 4.2 TextCNN

TextCNN applies convolution filters over word embeddings. It is suitable for detecting short local phrases such as:

- “not safe tonight”,
- “deadline pressure”,
- “need counseling”,
- “nothing urgent”.

This model covers the CNN part of the course using text instead of images.

### 4.3 BiLSTM

BiLSTM is a recurrent sequence model. It reads the sentence in both directions, so it can use context before and after important words. This is useful when negation or clarification changes meaning, for example: “not in danger, but emotionally struggling.”

### 4.4 DistilBERT Transformer

The project includes `src/train_transformer.py` for optional pretrained Transformer fine-tuning. DistilBERT uses attention and pretrained language representations, so it is expected to perform better on ambiguous natural language when enough compute is available.

## 5. Training Setup

| Item | Choice |
|---|---|
| Main metric | Macro F1 |
| Secondary metric | Accuracy |
| Split method | Stratified train/validation/test |
| Baseline model | TF-IDF + SGD Logistic Regression |
| Deep models | TextCNN, BiLSTM |
| Loss | Cross-entropy |
| Optimizer | AdamW for deep models |
| Regularization | Class weights, dropout, weight decay |
| Reproducibility | Fixed seed, saved reports, saved model files |

Macro F1 is important because the `urgent_human_review` class is smaller than the normal class. Accuracy alone can hide poor minority-class performance.

## 6. Experiments and Results

### 6.1 Main test split results

| Model | Test Accuracy | Test Macro F1 | Notes |
|---|---:|---:|---|
| TF-IDF + SGD Logistic Regression | 1.000 | 1.000 | Very strong on synthetic-distribution test split |
| TextCNN | 1.000 | 1.000 | Quickly learns local phrase cues |
| BiLSTM | 0.535 | 0.475 | Undertrained CPU demo; needs more epochs/data |

### 6.2 Challenge split results

| Model | Challenge Accuracy | Challenge Macro F1 | Notes |
|---|---:|---:|---|
| TF-IDF + SGD Logistic Regression | 0.865 | 0.864 | Best included model on hard ambiguous set |
| TextCNN | 0.744 | 0.723 | Good, but confused by negation and mixed context |
| BiLSTM | 0.381 | 0.339 | Needs stronger training to learn difficult context |

### 6.3 Interpretation

The normal test split is easy because it comes from the same generator distribution as the training data. The challenge split is more meaningful because it tests whether models understand context, negation, and routing rules.

The baseline performs best on the challenge split. This is not a failure; it is an important experimental finding. It means many routing cues are lexical, and a simple model can outperform an undertrained deep model. For the final discussion, this should be explained clearly instead of hidden.

## 7. Error Analysis

Mistake files are saved in `outputs/`:

- `baseline_challenge_mistakes.csv`
- `textcnn_challenge_mistakes.csv`
- `bilstm_challenge_mistakes.csv`

Common error patterns:

1. **Negation errors:** models may see “hurt myself” and ignore “I do not plan to.”
2. **Academic vs emotional overlap:** grade stress can use emotional words such as “worthless” or “hopeless.”
3. **Urgent phrases mixed with academic context:** urgent messages may also mention exams or grades.
4. **Routine messages with stress words:** a normal message may mention stress historically but ask a routine question.
5. **Undertrained sequence model:** BiLSTM needs more epochs and/or GPU training.

## 8. Innovation and Intellectual Contribution

The project is innovative because it reformulates text classification as a **student-support routing problem**, not just sentiment analysis. It includes:

- a meaningful education/wellbeing use case,
- a large custom dataset,
- a separate challenge set for harder evaluation,
- multiple model families: baseline, CNN, RNN, Transformer script,
- human-in-the-loop safety framing,
- error analysis focused on real routing risks.

## 9. Limitations and Ethics

The dataset is synthetic and cannot represent all student cultures, languages, slang, or mental-health expressions. A real deployment would require institution-approved data, privacy review, counselor involvement, bias testing, threshold calibration, and emergency escalation protocols.

The model must never be used to automatically diagnose students or replace human review. False negatives in urgent messages are high-risk, so urgent recall should be prioritized over overall accuracy in any real system.

## 10. Conclusion

CampusCare AI demonstrates a complete deep learning project workflow: problem framing, data curation, preprocessing, baseline modeling, CNN/RNN experiments, optional Transformer training, metrics, comparison, challenge-set evaluation, and error analysis. The current strongest model is the TF-IDF baseline on the challenge set, while TextCNN performs strongly on the standard test split. The BiLSTM result shows the importance of training time, model tuning, and not assuming deep models always win.

## References

- PyTorch documentation.
- scikit-learn documentation.
- Hugging Face Transformers documentation for optional DistilBERT script.
