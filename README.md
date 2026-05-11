# Hybrid Feature-Enhanced IndoBERT with Controlled Semi-Supervised Learning

Reproducibility package for the manuscript:

**Hybrid Feature-Enhanced IndoBERT with Controlled Semi-Supervised Learning for Low-Resource Indonesian Hate Speech Detection**

## Contents

- Dataset preparation from IDHSD, HS-572, and RE datasets.
- Baseline experiments: SVM, Logistic Regression, Random Forest, RFDT, CNN, LSTM, IndoBERT, mBERT, XLM-RoBERTa, IndoBERTweet.
- Proposed Hybrid IndoBERT + handcrafted + TF-IDF--SVD features.
- Controlled semi-supervised learning with adaptive thresholding and class-balanced pseudo-labeling.
- LLM-based zero-shot evaluation script.

## Repository Structure

```text
configs/                  YAML configuration
data/raw/                 Raw datasets; not included
data/processed/           Processed splits
notebooks/                Annotated notebooks
src/                      Python source code
scripts/                  Experiment entry points
results/                  Tables and figures
```

## Data

Place these files in `data/raw/`:

```text
IDHSD_RIO_unbalanced_713_2017.txt
572-hate-speech-dataset.csv
re_dataset.csv
new_kamusalay.csv
abusive.csv
```

## Run

```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py --config configs/default.yaml
python scripts/02_run_baselines.py --config configs/default.yaml
python scripts/03_run_hybrid_ssl.py --config configs/default.yaml
```

## LLM Evaluation

```bash
export OPENAI_API_KEY="your_api_key"
python scripts/04_run_llm_eval.py --config configs/default.yaml --model gpt-4o-mini
```

## Notes

- Primary metric: Macro-F1.
- Low-resource settings: 5%, 10%, and 20% labeled data.
- Threshold selection is performed on validation Macro-F1.
- Raw datasets are not redistributed in this repository.
