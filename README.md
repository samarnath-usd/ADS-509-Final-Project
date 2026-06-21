# Financial News Sentiment Classification using Fine-tuned DistilBERT

**ADS-509 — Applied Text Mining | Team 1**  
**Team Members:** Zadafiya Bhrami · Amarnath Sreekara  
**University of San Diego, Shiley-Marcos School of Engineering**

---

## Project Overview

This project develops a text classification pipeline to automatically detect sentiment (Positive, Negative, Neutral) in financial news headlines. General-purpose sentiment tools like VADER perform poorly on financial text because domain-specific language (e.g., "impairment charges", "missed guidance") lacks clear polarity in standard lexicons.

We fine-tune **DistilBERT** (`distilbert-base-uncased`) — a 66M-parameter distilled variant of BERT that is 40% smaller and 60% faster while retaining ~97% of BERT-base performance — and compare it against a VADER rule-based baseline.

---

## Repository Structure

```
ADS-Final-Project/
├── data/
│   ├── all-data.csv              # Financial PhraseBank dataset (Malo et al., 2014)
│   ├── train.csv                 # 70% split (3,386 rows)
│   ├── val.csv                   # 15% split (726 rows)
│   ├── test.csv                  # 15% split (726 rows)
│   ├── vader_results.csv         # VADER baseline metrics
│   └── final_results.csv         # Model comparison summary
├── notebooks/
│   ├── 01_EDA_preprocessing.ipynb   # Data cleaning, EDA, word clouds, VADER baseline
│   └── 02_FinBERT_finetuning.ipynb  # DistilBERT fine-tuning, evaluation, error analysis
├── figures/
│   ├── class_distribution.png
│   ├── sentence_length_hist.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── wordcloud_neutral.png
│   ├── wordclouds_combined.png
│   ├── top_words_by_class.png
│   ├── vader_confusion_matrix.png
│   ├── distilbert_confusion_matrix.png
│   ├── training_curves.png
│   ├── model_comparison_bar.png
│   ├── per_class_f1.png
│   ├── error_types.png
│   └── error_length_dist.png
├── requirements.txt
└── README.md
```

---

## Dataset

**Financial PhraseBank** (Malo et al., 2014)  
- Source: [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)  
- 4,840 English sentences from financial news articles  
- 2 columns: `text` (raw sentence), `sentiment` (Positive / Negative / Neutral)  
- Class distribution: Neutral ~59.5%, Positive ~28.2%, Negative ~12.5% (imbalanced)  
- Format: CSV (`all-data.csv`), Latin-1 encoded, no missing values after cleaning

---

## Notebooks

### 01_EDA_preprocessing.ipynb
**Author: Zadafiya Bhrami**

Covers:
- Data loading and cleaning (whitespace normalization, label encoding)
- Train / validation / test split (70 / 15 / 15)
- Exploratory Data Analysis: class distribution chart, sentence length histogram, top words per class
- Word cloud visualizations per class
- VADER rule-based baseline: implementation and evaluation (accuracy, macro F1, classification report)

### 02_FinBERT_finetuning.ipynb
**Author: Amarnath Sreekara**

Covers:
- DistilBERT model setup (`distilbert-base-uncased`, 66M parameters) via HuggingFace Transformers
- Tokenization (max length 128) and HuggingFace `Dataset` conversion
- Fine-tuning with `TrainingArguments` and `WeightedTrainer` (inverse-frequency class weights, early stopping with patience=5, max 25 epochs)
- Model evaluation: confusion matrix, full classification report, per-class F1
- Training curves: loss over steps, validation macro F1 over epochs
- Error analysis: misclassification patterns, representative examples, sentence length distribution

---

## Methods

| Step | Tool / Library |
|------|----------------|
| Preprocessing | pandas, scikit-learn |
| Baseline | vaderSentiment |
| Model | `distilbert-base-uncased` (HuggingFace Transformers) |
| Class imbalance | Inverse-frequency class weights (Negative: 2.67×, Positive: 1.18×, Neutral: 0.56×) |
| Fine-tuning | HuggingFace `Trainer` API, early stopping (patience=5) |
| Evaluation | scikit-learn (F1, confusion matrix, classification report) |
| Visualization | matplotlib, seaborn, wordcloud |

---

## Results

### Overall Performance (Test Set, n=726)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| VADER (baseline) | 0.551 | 0.482 |
| Fine-tuned DistilBERT | **0.850** | **0.834** |

Improvement: **+29.9 pp accuracy**, **+35.2 pp macro F1**

### Per-Class F1 — Fine-tuned DistilBERT

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.83 | 0.87 | **0.85** | 91 |
| Neutral | 0.86 | 0.91 | **0.89** | 431 |
| Positive | 0.83 | 0.71 | **0.77** | 204 |

### Training Details

- Stopped at ~epoch 11 (2,332 steps) out of 25 max — early stopping triggered after validation macro F1 plateaued
- Final training loss: 0.2255

### Key Error Patterns (109 misclassified, 15.0%)

| Error Type | Count |
|------------|-------|
| Positive → Neutral | 54 |
| Neutral → Positive | 27 |
| Neutral → Negative | 11 |
| Negative → Neutral | 10 |
| Positive → Negative | 5 |
| Negative → Positive | 2 |

The dominant failure mode is **Positive → Neutral** (49.5% of errors): positive financial news written in factual or procedural language lacks strong sentiment signals and gets classified as Neutral.

---

## How to Run

1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset from Kaggle and place `all-data.csv` in the `data/` folder.

3. Run notebooks in order:
   - `01_EDA_preprocessing.ipynb` — generates splits, figures, and VADER baseline
   - `02_FinBERT_finetuning.ipynb` — fine-tunes DistilBERT and produces all evaluation figures

> **Note:** The fine-tuned model weights are not included in this repository due to size (~260 MB). Re-run `02_FinBERT_finetuning.ipynb` to reproduce them.

---

## References

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.
