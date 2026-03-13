"""
Run inference on new data.

Usage:
    python scripts/predict.py --input path/to/data.csv --model lightgbm
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("predict")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict alarm categories")
    parser.add_argument("--input", required=True, help="CSV file to predict")
    parser.add_argument("--model", default="lightgbm",
                        choices=["random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--output", default="predictions.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)

    # Load model & encoder
    with open(models_dir / f"{args.model}.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(models_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Load input
    df = pd.read_csv(args.input, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()

    # Drop non-feature columns if present
    drop_cols = ["Label", "alarm_category", "Timestamp"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Predict
    y_pred_enc = pipeline.predict(X)
    y_pred = le.inverse_transform(y_pred_enc)
    y_prob = pipeline.predict_proba(X)

    out = df.copy()
    out["predicted_category"] = y_pred
    out["confidence"] = y_prob.max(axis=1)
    for i, cls in enumerate(le.classes_):
        out[f"prob_{cls}"] = y_prob[:, i]

    out.to_csv(args.output, index=False)
    logger.info(f"Predictions saved -> {args.output}")
    print(pd.Series(y_pred).value_counts().to_string())


if __name__ == "__main__":
    main()
