"""
Training entrypoint.

Usage:
    python scripts/train.py [--config config.yaml] [--sample 100000]
"""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.loader import CICIDSLoader
from src.data.preprocessor import CICIDSPreprocessor
from src.data.synthesizer import AlarmSynthesizer
from src.models.trainer import AlarmModelTrainer
from src.models.evaluator import AlarmModelEvaluator
from src.utils.logger import get_logger

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train alarm triage models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N rows from CICIDS (for quick testing)")
    parser.add_argument("--models", nargs="+",
                        choices=["random_forest", "xgboost", "lightgbm"],
                        default=None, help="Models to train (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.sample:
        cfg["cicids"]["sample_size"] = args.sample

    # 1. Load
    loader = CICIDSLoader(
        raw_dir=cfg["data"]["raw_dir"],
        encoding=cfg["cicids"]["encoding"],
        sample_size=cfg["cicids"]["sample_size"],
        random_state=cfg["data"]["random_state"],
    )
    df = loader.load_all()

    # 2. Preprocess
    preprocessor = CICIDSPreprocessor(label_mapping=cfg["label_mapping"])
    X, y = preprocessor.process(df)

    # 3. Augment with synthetic categories
    synth_cfg = cfg.get("synthesis", {})
    synthesizer = AlarmSynthesizer(random_state=cfg["data"]["random_state"])
    X, y = synthesizer.augment(
        X, y,
        fire_samples=synth_cfg.get("fire_samples", 5000),
        medical_emergency_samples=synth_cfg.get("medical_emergency_samples", 3000),
    )

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        stratify=y,
        random_state=cfg["data"]["random_state"],
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 5. Train
    trainer = AlarmModelTrainer(cfg, output_dir=cfg["output"]["models_dir"])
    cv_results = trainer.train(X_train, y_train, model_names=args.models)
    logger.info(f"CV results: {cv_results}")

    # 6. Evaluate on hold-out test set
    evaluator = AlarmModelEvaluator(
        label_encoder=trainer.label_encoder,
        reports_dir=cfg["output"]["reports_dir"],
    )
    evaluator.evaluate(trainer.trained_models, X_test, y_test)


if __name__ == "__main__":
    main()
