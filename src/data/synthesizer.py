"""
Synthetic data generation for alarm categories not present in CICIDS2017.

CICIDS2017 captures network intrusion events. Categories such as 'fire' and
'medical_emergency' are physical-world events that require sensor data not
available in the dataset. To enable a 5-class classifier for demonstration
and development purposes, this module generates synthetic samples with
statistically plausible feature signatures.

NOTE: In a production deployment these categories should be populated with
real sensor data (smoke detectors, wearables, panic buttons, etc.).
"""

import numpy as np
import pandas as pd
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlarmSynthesizer:
    """
    Generates synthetic alarm samples for 'fire' and 'medical_emergency'
    by perturbing a reference subset of the real feature space.
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def _perturb(
        self,
        reference: pd.DataFrame,
        n_samples: int,
        scale: float = 0.1,
    ) -> pd.DataFrame:
        """Draw n_samples from a gaussian centered on reference column means."""
        means = reference.mean(axis=0).values
        stds = reference.std(axis=0).fillna(0).values
        noise = self.rng.normal(0, scale, size=(n_samples, len(means)))
        synthetic = means + stds * noise
        synthetic = np.clip(synthetic, 0, None)  # features are non-negative
        return pd.DataFrame(synthetic, columns=reference.columns)

    def generate_fire(
        self, X: pd.DataFrame, n_samples: int = 5000
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fire events: characterised by short bursts of high-volume traffic
        (sensor flooding) and elevated packet rates. We derive them from
        the DoS subset of the feature space (high packet/byte rates).
        """
        synthetic = self._perturb(X, n_samples, scale=0.15)

        # Amplify rate-related features if present
        for col in ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s"]:
            if col in synthetic.columns:
                synthetic[col] *= self.rng.uniform(1.2, 2.0, n_samples)

        y = pd.Series(["fire"] * n_samples, name="alarm_category")
        logger.info(f"Generated {n_samples} synthetic fire samples")
        return synthetic, y

    def generate_medical_emergency(
        self, X: pd.DataFrame, n_samples: int = 3000
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Medical emergencies: single-zone activations with short durations and
        low packet counts (panic button / wearable trigger). Derived from
        BENIGN-like low-traffic features.
        """
        synthetic = self._perturb(X, n_samples, scale=0.05)

        # Dampen volume features
        for col in ["Total Fwd Packets", "Total Backward Packets",
                    "Total Length of Fwd Packets", "Total Length of Bwd Packets"]:
            if col in synthetic.columns:
                synthetic[col] *= self.rng.uniform(0.05, 0.3, n_samples)

        y = pd.Series(["medical_emergency"] * n_samples, name="alarm_category")
        logger.info(f"Generated {n_samples} synthetic medical_emergency samples")
        return synthetic, y

    def augment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fire_samples: int = 5000,
        medical_emergency_samples: int = 3000,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Append synthetic categories to real dataset."""
        X_fire, y_fire = self.generate_fire(X, fire_samples)
        X_med, y_med = self.generate_medical_emergency(X, medical_emergency_samples)

        X_aug = pd.concat([X, X_fire, X_med], ignore_index=True)
        y_aug = pd.concat([y, y_fire, y_med], ignore_index=True)

        logger.info(
            f"Augmented dataset: {len(X_aug):,} samples\n"
            f"{y_aug.value_counts().to_string()}"
        )
        return X_aug, y_aug
