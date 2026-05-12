# Hybrid Feature-Enhanced IndoBERT with Controlled Semi-Supervised Learning

This repository provides the reproducibility package for the manuscript:

**Hybrid Feature-Enhanced IndoBERT with Controlled Semi-Supervised Learning for Low-Resource Indonesian Hate Speech Detection**

## Contents

- Main experiment notebooks for preprocessing, hybrid feature extraction, Hybrid IndoBERT, and controlled SSL.
- Baseline notebooks for SVM, Logistic Regression, Decision Tree, Random Forest, CNN, LSTM, IndoBERT, mBERT, XLM-RoBERTa, IndoBERTweet/XLM-IndoBERT, and naive pseudo-labeling.
- LLM evaluation template requested by the editor.
- Result tables and figures corresponding to the manuscript.
- Generative AI declaration.

## Structure

```text
configs/                  YAML configuration
data/raw/                 Raw datasets; not redistributed
data/processed/           Processed outputs
notebooks/main/           Main paper experiment notebooks
notebooks/baselines/      Baseline experiment notebooks
scripts/                  Script entry points
src/                      Modular utilities
results/tables/           CSV versions of manuscript tables
results/figures/          Manuscript figures
manuscript_snippets/      LaTeX snippets for manuscript revision
```

## Raw Data

Place the following files in `data/raw/`:

```text
IDHSD_RIO_unbalanced_713_2017.txt
572-hate-speech-dataset.csv
re_dataset.csv
new_kamusalay.csv
abusive.csv
```

Raw datasets are not redistributed because they may be governed by their original licenses.

## Main Notebooks

```text
notebooks/main/01_data_preparation_and_main_pipeline.ipynb
notebooks/main/02_hybrid_feature_experiments.ipynb
notebooks/main/03_hybrid_indobert_experiment.ipynb
notebooks/main/04_extended_hybrid_experiment.ipynb
```

## Baseline Notebooks

```text
notebooks/baselines/SVM-TFIDF.ipynb
notebooks/baselines/LR-TFIDF.ipynb
notebooks/baselines/DT-TFIDF.ipynb
notebooks/baselines/RF-TFIDF.ipynb
notebooks/baselines/CNN-text-Classifier.ipynb
notebooks/baselines/LSTM-text-Classifier.ipynb
notebooks/baselines/IndoBERT-text-Classifier.ipynb
notebooks/baselines/mBERT-text-Classifier.ipynb
notebooks/baselines/XLM-RoBERTa-text-Classifier.ipynb
notebooks/baselines/IndoBERTweet-or-XLM-IndoBERT-text-Classifier.ipynb
notebooks/baselines/Naive-PseudoLabeling-text-Classifier.ipynb
```

## Result Tables

```text
results/tables/sota_baseline_comparison.csv
results/tables/internal_variant_results.csv
results/tables/llm_evaluation_results_TEMPLATE.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## LLM Evaluation

```bash
export OPENAI_API_KEY="your_api_key"
python scripts/04_run_llm_eval.py --config configs/default.yaml --model gpt-4o-mini
```

## Code Availability Statement

Use this GitHub URL in the manuscript after upload:

```latex
\url{https://github.com/shoffan28/hybrid-indobert-ssl-hatespeech}
```
