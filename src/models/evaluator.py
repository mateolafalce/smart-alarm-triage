import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlarmModelEvaluator:

    def __init__(self, label_encoder, reports_dir: str = "reports"):
        self.label_encoder = label_encoder
        self.reports_dir = Path(reports_dir)
        (self.reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        models: dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        y_encoded = self.label_encoder.transform(y_test)
        classes = self.label_encoder.classes_
        all_results = {}

        for name, pipeline in models.items():
            logger.info(f"Evaluating {name}...")
            y_pred = pipeline.predict(X_test)
            y_prob = (
                pipeline.predict_proba(X_test)
                if hasattr(pipeline, "predict_proba")
                else None
            )

            report = classification_report(
                y_encoded, y_pred, target_names=classes, output_dict=True
            )
            f1_w = f1_score(y_encoded, y_pred, average="weighted")

            roc_auc = None
            if y_prob is not None:
                y_bin = label_binarize(y_encoded, classes=list(range(len(classes))))
                try:
                    roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
                except Exception:
                    pass

            all_results[name] = {
                "f1_weighted": f1_w,
                "roc_auc_weighted": roc_auc,
                "classification_report": report,
            }

            self._plot_confusion_matrix(name, y_encoded, y_pred, classes)
            if roc_auc:
                logger.info(f"  {name}: F1={f1_w:.4f}, AUC={roc_auc:.4f}")
            else:
                logger.info(f"  {name}: F1={f1_w:.4f}")

        self._save_results(all_results)
        self._print_leaderboard(all_results)
        return all_results

    def _plot_confusion_matrix(self, name, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax
        )
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        path = self.reports_dir / "figures" / f"cm_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved confusion matrix -> {path}")

    def _save_results(self, results: dict):
        path = self.reports_dir / "evaluation_results.json"
        serializable = {}
        for name, r in results.items():
            serializable[name] = {
                "f1_weighted": r["f1_weighted"],
                "roc_auc_weighted": r["roc_auc_weighted"],
            }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved -> {path}")

    def _print_leaderboard(self, results: dict):
        rows = [
            {"model": k, "F1 (weighted)": v["f1_weighted"],
             "ROC-AUC": v["roc_auc_weighted"]}
            for k, v in results.items()
        ]
        df = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)
        print("\n=== Model Leaderboard ===")
        print(df.to_string(index=False))
