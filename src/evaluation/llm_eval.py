import os, time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from .metrics import evaluate_predictions

PROMPT_TEMPLATE = """Classify the following Indonesian social media text as exactly one of:
Hate Speech
Non-Hate Speech

Text:
{text}

Answer only one label:
"""

def parse_label(response):
    r = str(response).lower()
    if "non" in r and "hate" in r:
        return 0
    if "hate" in r:
        return 1
    return 0

def evaluate_openai_llm(df, text_col="clean_text", label_col="label", model="gpt-4o-mini", sleep=0.0):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    preds, raw = [], []
    for text in tqdm(df[text_col].astype(str).tolist(), desc=f"LLM eval: {model}"):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
            temperature=0,
            max_tokens=10,
        )
        content = resp.choices[0].message.content.strip()
        raw.append(content)
        preds.append(parse_label(content))
        if sleep:
            time.sleep(sleep)
    y_true = df[label_col].astype(int).values
    metrics = evaluate_predictions(y_true, preds, y_prob=None)
    pred_df = pd.DataFrame({text_col: df[text_col].values, label_col: y_true, "prediction": preds, "raw_response": raw})
    return metrics, pred_df
