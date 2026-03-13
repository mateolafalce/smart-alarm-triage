import numpy as np
import pandas as pd
import pytest
from src.data.preprocessor import CICIDSPreprocessor


def _make_df(labels):
    n = len(labels)
    data = {
        "Flow Duration": np.random.rand(n) * 1000,
        "Total Fwd Packets": np.random.randint(1, 100, n).astype(float),
        "Total Backward Packets": np.random.randint(1, 50, n).astype(float),
        "Label": labels,
    }
    return pd.DataFrame(data)


def test_clean_removes_inf():
    df = _make_df(["BENIGN", "PortScan"])
    df.loc[0, "Flow Duration"] = np.inf
    prep = CICIDSPreprocessor()
    cleaned = prep.clean(df)
    assert not np.isinf(cleaned["Flow Duration"]).any()


def test_map_labels():
    df = _make_df(["BENIGN", "PortScan", "DDoS"])
    prep = CICIDSPreprocessor()
    df = prep.clean(df)
    df = prep.map_labels(df)
    assert set(df["alarm_category"].unique()) == {"false_alarm", "intrusion_real", "panic"}


def test_unmapped_labels_dropped():
    df = _make_df(["BENIGN", "UNKNOWN_LABEL"])
    prep = CICIDSPreprocessor()
    df = prep.clean(df)
    df = prep.map_labels(df)
    assert "UNKNOWN_LABEL" not in df.get("Label", pd.Series()).values
    assert len(df) == 1


def test_process_returns_xy_shapes():
    df = _make_df(["BENIGN"] * 10 + ["PortScan"] * 5)
    prep = CICIDSPreprocessor()
    X, y = prep.process(df)
    assert len(X) == len(y) == 15
    assert "Label" not in X.columns
    assert "alarm_category" not in X.columns
