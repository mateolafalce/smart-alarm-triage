# Smart Alarm Triage

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Status](https://img.shields.io/badge/status-experimental-orange)

A machine learning system that classifies security and physical alarm events into five actionable categories using network traffic features derived from the CICIDS2017 dataset. Three classifiers are trained and compared: Random Forest, XGBoost, and LightGBM.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Label Mapping](#label-mapping)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Synthetic Categories](#synthetic-categories)
- [Production Considerations](#production-considerations)
- [License](#license)

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

The loader accepts any subset of the files — you can start with a single CSV (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) for fast iteration.

### Label Mapping

| CICIDS2017 Label | Alarm Category |
|---|---|
| BENIGN | false_alarm |
| DoS Hulk | panic |
| DDoS | panic |
| DoS GoldenEye | panic |
| DoS slowloris | panic |
| DoS Slowhttptest | panic |
| PortScan | intrusion_real |
| FTP-Patator | intrusion_real |
| SSH-Patator | intrusion_real |
| Bot | intrusion_real |
| Web Attack - Brute Force | intrusion_real |
| Web Attack - XSS | intrusion_real |
| Web Attack - Sql Injection | intrusion_real |
| Infiltration | intrusion_real |
| Heartbleed | intrusion_real |

---

## Architecture

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

The pipeline uses `StratifiedKFold` cross-validation (5 folds) during training, then evaluates on a held-out 20% test split. All preprocessing steps are encapsulated inside the sklearn `Pipeline` object so they are applied consistently at inference time.

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

### Volume mounts

The `docker-compose.yml` mounts three local directories into the container:

| Host | Container | Purpose |
|---|---|---|
| `./data` | `/app/data` | CICIDS CSV files + processed data |
| `./models` | `/app/models` | Saved model artifacts (`.pkl`) |
| `./reports` | `/app/reports` | Evaluation JSON + confusion matrix PNGs |

---

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

---

## Model Performance

> Placeholder — run `make train` with the full CICIDS2017 dataset to populate.

| Model | F1 (weighted) | ROC-AUC (weighted) |
|---|---|---|
| LightGBM | — | — |
| XGBoost | — | — |
| Random Forest | — | — |

Evaluation results are saved to `reports/evaluation_results.json` and confusion matrix plots to `reports/figures/cm_<model>.png` after training.

---

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

**Important:** These synthetic samples are statistical approximations. The classifier's behaviour on `fire` and `medical_emergency` inputs reflects the synthetic distribution, not real-world sensor data.

---

## Production Considerations

Before deploying this system in a real environment, the following changes are strongly recommended:

1. **Replace synthetic categories with real data.** Integrate actual smoke detector telemetry, wearable health sensor streams, and physical panic button events to train the `fire` and `medical_emergency` classes.

2. **Retrain on domain-specific traffic.** CICIDS2017 was captured in a lab environment. Real deployment networks may have different baseline traffic patterns. Consider fine-tuning or retraining on network captures from the target environment.

3. **Add a confidence threshold.** The `predict.py` script outputs per-class probabilities. In production, low-confidence predictions should be escalated to human review rather than acted upon automatically.

4. **Monitor for drift.** Network traffic distributions change over time. Implement monitoring to detect model degradation and trigger retraining.

5. **Harden the feature pipeline.** Edge cases such as completely empty flows, non-numeric columns, and unseen feature names should be handled gracefully in the preprocessing layer.

6. **Secure model artifacts.** Serialised `.pkl` files can execute arbitrary code when loaded. Use signed model registries and verify checksums before loading models in production.

---

## License

MIT License. See `LICENSE` for details.
