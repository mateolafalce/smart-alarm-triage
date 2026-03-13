import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data.loader import CICIDSLoader


def test_load_file_strips_columns(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text(" Flow Duration , Label \n1.0,BENIGN\n2.0,PortScan\n")
    loader = CICIDSLoader(raw_dir=str(tmp_path))
    df = loader.load_file(csv)
    assert "Flow Duration" in df.columns
    assert "Label" in df.columns


def test_load_all_raises_when_no_files(tmp_path):
    loader = CICIDSLoader(raw_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load_all()


def test_load_all_samples(tmp_path):
    csv = tmp_path / "test.csv"
    rows = "\n".join(["A,B"] + [f"{i},{i}" for i in range(100)])
    csv.write_text(rows)
    loader = CICIDSLoader(raw_dir=str(tmp_path), sample_size=10)
    df = loader.load_all()
    assert len(df) == 10
