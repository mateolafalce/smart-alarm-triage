"""
Evaluate saved models on a test CSV.

Usage:
    python scripts/evaluate.py --input data/test.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.preprocessor import CICIDSPreprocessor
from src.models.evaluator import AlarmModelEvaluator
from src.config import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models_dir", default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    models_dir = Path(args.models_dir)

    df = pd.read_csv(args.input, encoding="latin-1", low_memory=False)
    preprocessor = CICIDSPreprocessor(label_mapping=cfg["label_mapping"])
    X, y = preprocessor.process(df)

    with open(models_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    models = {}
    for name in cfg["models"]:
        path = models_dir / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            logger.info(f"Loaded {name}")

    evaluator = AlarmModelEvaluator(le, reports_dir=cfg["output"]["reports_dir"])
    evaluator.evaluate(models, X, y)


if __name__ == "__main__":
    main()
