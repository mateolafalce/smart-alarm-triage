import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from src.models.pipelines import PIPELINE_BUILDERS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlarmModelTrainer:
    """Trains and cross-validates alarm triage classifiers."""

    def __init__(self, config: dict, output_dir: str = "models"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = LabelEncoder()
        self.trained_models: Dict[str, object] = {}

    def _encode_labels(self, y: pd.Series) -> np.ndarray:
        return self.label_encoder.fit_transform(y)

    def cross_validate_model(
        self,
        name: str,
        pipeline,
        X: pd.DataFrame,
        y_encoded: np.ndarray,
        cv_folds: int = 5,
        scoring: str = "f1_weighted",
    ) -> Dict[str, float]:
        logger.info(f"Cross-validating {name} ({cv_folds} folds)...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_validate(
            pipeline,
            X,
            y_encoded,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        result = {
            f"val_{scoring}_mean": scores[f"test_score"].mean(),
            f"val_{scoring}_std": scores[f"test_score"].std(),
            f"train_{scoring}_mean": scores[f"train_score"].mean(),
        }
        logger.info(
            f"  {name}: val {scoring} = "
            f"{result[f'val_{scoring}_mean']:.4f} +/- {result[f'val_{scoring}_std']:.4f}"
        )
        return result

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_names: list = None,
    ) -> Dict[str, dict]:
        y_encoded = self._encode_labels(y_train)
        n_classes = len(self.label_encoder.classes_)
        model_names = model_names or list(self.config["models"].keys())

        results = {}
        for name in model_names:
            params = self.config["models"][name]

            if name == "xgboost":
                pipeline = PIPELINE_BUILDERS[name](params, n_classes)
            else:
                pipeline = PIPELINE_BUILDERS[name](params)

            cv_results = self.cross_validate_model(
                name,
                pipeline,
                X_train,
                y_encoded,
                cv_folds=self.config["training"]["cv_folds"],
                scoring=self.config["training"]["scoring"],
            )

            logger.info(f"Fitting {name} on full training set...")
            pipeline.fit(X_train, y_encoded)
            self.trained_models[name] = pipeline

            self._save_model(name, pipeline)
            results[name] = cv_results

        self._save_label_encoder()
        return results

    def _save_model(self, name: str, pipeline) -> None:
        path = self.output_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Saved {name} -> {path}")

    def _save_label_encoder(self) -> None:
        path = self.output_dir / "label_encoder.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Saved label encoder -> {path}")
