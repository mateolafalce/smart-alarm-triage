import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical CICIDS label -> alarm category mapping
# Web Attack labels may use en-dash (--) or a special char depending on encoding
_CICIDS_TO_ALARM: Dict[str, str] = {
    "BENIGN": "false_alarm",
    # DoS / DDoS -> panic (overwhelming, high-volume events)
    "DoS Hulk": "panic",
    "DDoS": "panic",
    "DoS GoldenEye": "panic",
    "DoS slowloris": "panic",
    "DoS Slowhttptest": "panic",
    # Targeted intrusions
    "PortScan": "intrusion_real",
    "FTP-Patator": "intrusion_real",
    "SSH-Patator": "intrusion_real",
    "Bot": "intrusion_real",
    "Web Attack \u2013 Brute Force": "intrusion_real",
    "Web Attack \u2013 XSS": "intrusion_real",
    "Web Attack \u2013 Sql Injection": "intrusion_real",
    # Legacy encoding variant (latin-1 en-dash)
    "Web Attack \x96 Brute Force": "intrusion_real",
    "Web Attack \x96 XSS": "intrusion_real",
    "Web Attack \x96 Sql Injection": "intrusion_real",
    "Infiltration": "intrusion_real",
    "Heartbleed": "intrusion_real",
}


class CICIDSPreprocessor:
    """
    Cleans CICIDS2017 data and maps original labels to alarm categories.
    """

    LABEL_COL = "Label"
    TARGET_COL = "alarm_category"

    def __init__(self, label_mapping: Dict[str, str] = None):
        self.label_mapping = label_mapping or _CICIDS_TO_ALARM

    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Replace infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)

        # Fill NaN in numeric columns with 0
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)

        logger.info(f"After cleaning: {df.shape}")
        return df

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.LABEL_COL] = df[self.LABEL_COL].str.strip()
        df[self.TARGET_COL] = df[self.LABEL_COL].map(self.label_mapping)

        unmapped = df.loc[df[self.TARGET_COL].isna(), self.LABEL_COL].unique()
        if len(unmapped):
            logger.warning(f"Dropping {len(unmapped)} unmapped label(s): {unmapped}")
            df = df[df[self.TARGET_COL].notna()]

        logger.info(
            f"Label distribution:\n{df[self.TARGET_COL].value_counts().to_string()}"
        )
        return df

    def split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        drop_cols = [self.LABEL_COL, self.TARGET_COL, "Timestamp"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=[np.number])
        y = df[self.TARGET_COL]
        logger.info(f"Features: {X.shape[1]}, samples: {len(y)}")
        return X, y

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.clean(df)
        df = self.map_labels(df)
        return self.split_xy(df)
