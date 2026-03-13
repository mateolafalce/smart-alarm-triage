import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlarmFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derives higher-level features relevant to alarm triage from raw
    CICIDS2017 network flow statistics.
    """

    # Feature groups (adjust if using a different CICIDS variant)
    FWD_COLS = ["Total Fwd Packets", "Total Length of Fwd Packets", "Fwd Packets/s"]
    BWD_COLS = ["Total Backward Packets", "Total Length of Bwd Packets", "Bwd Packets/s"]
    RATE_COLS = ["Flow Bytes/s", "Flow Packets/s"]
    DURATION_COL = "Flow Duration"

    def fit(self, X, y=None):
        self._columns = list(X.columns) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self._columns).copy()

        # 1. Asymmetry ratio: fwd vs bwd packets
        if "Total Fwd Packets" in X and "Total Backward Packets" in X:
            total = (X["Total Fwd Packets"] + X["Total Backward Packets"]).replace(0, 1)
            X["fwd_bwd_ratio"] = X["Total Fwd Packets"] / total

        # 2. Bytes per packet
        if "Total Length of Fwd Packets" in X and "Total Fwd Packets" in X:
            X["bytes_per_fwd_pkt"] = (
                X["Total Length of Fwd Packets"]
                / X["Total Fwd Packets"].replace(0, 1)
            )

        # 3. Log-transform high-variance rate features
        for col in self.RATE_COLS:
            if col in X:
                X[f"log_{col.replace('/', '_').replace(' ', '_')}"] = np.log1p(
                    np.clip(X[col], 0, None)
                )

        # 4. Duration bucket (short / medium / long)
        if self.DURATION_COL in X:
            X["duration_log"] = np.log1p(np.clip(X[self.DURATION_COL], 0, None))

        return X
