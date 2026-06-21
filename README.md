# Financial News Sentiment Classification using Fine-tuned FinBERT

**ADS-509 — Applied Text Mining | Team 1**  
**Team Members:** Zadafiya Bhrami · Amarnath Sreekara  
**University of San Diego, Shiley-Marcos School of Engineering**

---

## Project Overview

This project develops a text classification pipeline to automatically detect sentiment (Positive, Negative, Neutral) in financial news headlines. General-purpose sentiment tools like VADER perform poorly on financial text because domain-specific language (e.g., "impairment charges", "missed guidance") lacks clear polarity in standard lexicons.

We fine-tune **FinBERT** — a BERT model pre-trained on financial corpora — and compare it against a VADER rule-based baseline.

---

## Repository Structure

```
final_project/
├── data/
│   └── all-data.csv              # Financial PhraseBank dataset (Malo et al., 2014)
├── notebooks/
│   ├── 01_EDA_preprocessing.ipynb   # Data cleaning, EDA, word clouds, VADER baseline
│   └── 02_FinBERT_finetuning.ipynb  # FinBERT fine-tuning, evaluation, error analysis
├── figures/                      # All saved visualizations (PNG)
│   ├── class_distribution.png
│   ├── sentence_length_hist.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── wordcloud_neutral.png
│   ├── vader_confusion_matrix.png
│   ├── finbert_confusion_matrix.png
│   ├── model_comparison_bar.png
│   └── per_class_f1.png
├── requirements.txt              # Python dependencies
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
- Exploratory Data Analysis: class distribution chart, sentence length histogram
- Word cloud visualizations per class
- VADER rule-based baseline: implementation and evaluation (accuracy, macro F1, classification report)

### 02_FinBERT_finetuning.ipynb
**Author: Amarnath Sreekara**

Covers:
- DistilBERT model setup (`distilbert-base-uncased`) via HuggingFace Transformers
- Tokenization and HuggingFace `Dataset` conversion
- Fine-tuning with `TrainingArguments` and `Trainer` (early stopping, class weights)
- Model evaluation: confusion matrix, classification report, per-class F1
- Figures: model comparison bar chart, per-class F1 comparison chart
- Error analysis: review of misclassified examples and pattern identification

---

## Methods

| Step | Tool / Library |
|------|----------------|
| Preprocessing | pandas, scikit-learn |
| Baseline | vaderSentiment |
| Model | `distilbert-base-uncased` (HuggingFace Transformers) |
| Fine-tuning | HuggingFace `Trainer` API |
| Evaluation | scikit-learn (F1, confusion matrix) |
| Visualization | matplotlib, seaborn, wordcloud |

---

## Results Summary

*(To be updated after training)*

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| VADER (baseline) | — | — |
| Fine-tuned FinBERT | — | — |

---

## How to Run

1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset from Kaggle and place `all-data.csv` in the `data/` folder.

3. Run notebooks in order:
   - `01_EDA_preprocessing.ipynb` — generates splits and baseline results
   - `02_FinBERT_finetuning.ipynb` — fine-tunes FinBERT and produces all figures

---

## References

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.

Yang, Y., Uy, M. C. S., & Huang, A. (2020). FinBERT: A Pretrained Language Model for Financial Communications. *arXiv preprint arXiv:2006.08097*.
