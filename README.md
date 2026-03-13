```
┌─────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│  data/raw/  │────>│  CICIDSLoader     │────>│  CICIDSPreprocessor  │
│  *.csv      │     │  (load + concat)  │     │  (clean + map labels)│
└─────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                          │
                                              ┌───────────▼───────────┐
                                              │  AlarmSynthesizer     │
                                              │  (add fire + medical) │
                                              └───────────┬───────────┘
                                                          │
                                          ┌───────────────▼──────────────────┐
                                          │  sklearn Pipeline                 │
                                          │  ┌──────────────────────────┐    │
                                          │  │ AlarmFeatureEngineer     │    │
                                          │  │ (ratios, logs, durations)│    │
                                          │  └──────────────────────────┘    │
                                          │  ┌──────────────────────────┐    │
                                          │  │ SimpleImputer + Scaler   │    │
                                          │  └──────────────────────────┘    │
                                          │  ┌──────────────────────────┐    │
                                          │  │ Classifier               │    │
                                          │  │ RF / XGBoost / LightGBM  │    │
                                          │  └──────────────────────────┘    │
                                          └───────────────┬──────────────────┘
                                                          │
                                              ┌───────────▼───────────┐
                                              │  AlarmModelEvaluator  │
                                              │  (F1, AUC, CM plots)  │
                                              └───────────────────────┘
```

<div align="center">

# Smart Alarm Triage

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

</div>

A machine learning system that classifies security and physical alarm events into five actionable categories using network traffic features derived from the CICIDS2017 dataset. Three classifiers are trained and compared: Random Forest, XGBoost, and LightGBM.

---

## Overview

Traditional intrusion detection systems generate a flat stream of alerts — a single binary "benign / malicious" signal that leaves security operators without context about the nature and urgency of each alarm. **Smart Alarm Triage** re-frames this problem as a 5-class classification task:

| Category | Description |
|---|---|
| `false_alarm` | Benign traffic incorrectly flagged |
| `intrusion_real` | Confirmed network intrusion (port scans, brute force, bots, etc.) |
| `panic` | Overwhelming volumetric event (DoS / DDoS) |
| `fire` | Physical fire event (synthetic; requires real sensor data in production) |
| `medical_emergency` | Physical medical event (synthetic; requires real sensor data in production) |

The pipeline is fully reproducible: download the CICIDS2017 CSVs, run `make train`, and receive trained model artifacts plus evaluation reports.

---

## Dataset

**CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017) contains 2.8 million network flows generated over five weekdays, labelled with 15 attack types and a BENIGN class.

Download instructions:
1. Visit https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the **MachineLearningCSV.zip** file
3. Extract the CSV files into `data/raw/`

The loader accepts any subset of the files — you can start with a single CSV for fast iteration.

---

## Quick Start

### Prerequisites

- Python 3.10+
- CICIDS2017 CSV files in `data/raw/`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd smart-alarm-triage

# Install dependencies (editable mode with dev extras)
make install
# or: pip install -e ".[dev]"
```

### Training

```bash
# Train all models on the full dataset
make train

# Quick test with 50k rows, LightGBM only
make train-sample

# Custom options
python scripts/train.py \
    --config config.yaml \
    --sample 100000 \
    --models lightgbm xgboost
```

### Evaluation

```bash
# Evaluate saved models against a labelled CSV
python scripts/evaluate.py --input data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### Prediction

```bash
# Run inference on unlabelled data
python scripts/predict.py \
    --input data/raw/sample.csv \
    --model lightgbm \
    --output predictions.csv
```

### Running Tests

```bash
make test
# or: pytest tests/ -v --tb=short
```

---

## Docker

### Build the image

```bash
make docker-build
# or: docker compose build
```

### Run training

```bash
make docker-train
# or: docker compose run --rm train
```

### Run training with custom arguments

```bash
docker compose run --rm train python scripts/train.py --sample 50000 --models lightgbm
```

### Run prediction

```bash
# Uses the `predict` profile defined in docker-compose.yml
docker compose --profile predict up predict
```

## Project Structure

```
smart-alarm-triage/
├── data/
│   ├── raw/              # Place CICIDS2017 CSV files here
│   └── processed/        # Intermediate processed data
├── models/               # Saved model artifacts (.pkl)
├── reports/
│   └── figures/          # Confusion matrix plots
├── src/
│   ├── config.py         # Config loader (config.yaml -> dict)
│   ├── data/
│   │   ├── loader.py       # CICIDSLoader: reads and concatenates CSVs
│   │   ├── preprocessor.py # CICIDSPreprocessor: clean + map labels
│   │   └── synthesizer.py  # AlarmSynthesizer: synthetic fire/medical
│   ├── features/
│   │   └── engineering.py  # AlarmFeatureEngineer (sklearn Transformer)
│   ├── models/
│   │   ├── pipelines.py    # Pipeline builders for each model
│   │   ├── trainer.py      # AlarmModelTrainer with CV
│   │   └── evaluator.py    # AlarmModelEvaluator: metrics + plots
│   └── utils/
│       └── logger.py       # Structured logging helper
├── scripts/
│   ├── train.py            # Training entrypoint
│   ├── evaluate.py         # Evaluation on saved models
│   └── predict.py          # Inference entrypoint
├── tests/
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── setup.py
├── config.yaml
├── Makefile
└── README.md
```

## Synthetic Categories

The CICIDS2017 dataset covers network-layer events only. The `fire` and `medical_emergency` categories represent physical-world alarms that have no equivalent in the dataset.

To enable a fully functional 5-class classifier for research and development, `src/data/synthesizer.py` generates synthetic samples using Gaussian perturbation of the real feature space:

- **fire**: derived from high-throughput flow statistics (elevated `Flow Bytes/s`, `Flow Packets/s`) representing sensor flooding during a fire event.
- **medical_emergency**: derived from low-traffic BENIGN-like flows (small packet counts) representing a wearable or panic button trigger.

The default synthesis volumes are controlled via `config.yaml`:

```yaml
synthesis:
  fire_samples: 5000
  medical_emergency_samples: 3000
  noise_std: 0.1
```

