import pandas as pd
from pathlib import Path
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CICIDSLoader:
    """
    Loads one or more CICIDS2017 CSV files and concatenates them into a
    single DataFrame.

    CICIDS2017 files can be downloaded from:
    https://www.unb.ca/cic/datasets/ids-2017.html
    """

    def __init__(
        self,
        raw_dir: str,
        encoding: str = "latin-1",
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ):
        self.raw_dir = Path(raw_dir)
        self.encoding = encoding
        self.sample_size = sample_size
        self.random_state = random_state

    def load_file(self, filepath) -> pd.DataFrame:
        logger.info(f"Reading {filepath}")
        df = pd.read_csv(filepath, encoding=self.encoding, low_memory=False)
        df.columns = df.columns.str.strip()
        logger.info(f"  -> {len(df):,} rows, {len(df.columns)} columns")
        return df

    def load_all(self, files: Optional[List[str]] = None) -> pd.DataFrame:
        if files is None:
            files = sorted(self.raw_dir.glob("*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_dir}. "
                "Download CICIDS2017 and place the CSV files in data/raw/."
            )

        combined = pd.concat(
            [self.load_file(f) for f in files], ignore_index=True
        )

        if self.sample_size:
            combined = combined.sample(
                n=min(self.sample_size, len(combined)),
                random_state=self.random_state,
            )
            logger.info(f"Sampled {len(combined):,} rows")

        logger.info(f"Dataset loaded: {combined.shape}")
        return combined
