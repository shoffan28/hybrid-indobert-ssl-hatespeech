import argparse, os, sys
import pandas as pd
sys.path.append(".")
from src.evaluation.llm_eval import evaluate_openai_llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--input_csv", default="data/processed/test.csv")
    parser.add_argument("--text_col", default="clean_text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)
    metrics, pred_df = evaluate_openai_llm(df, args.text_col, args.label_col, args.model)
    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(f"results/tables/llm_{args.model}_metrics.csv", index=False)
    pred_df.to_csv(f"results/tables/llm_{args.model}_predictions.csv", index=False)
    print(metrics)

if __name__ == "__main__":
    main()
