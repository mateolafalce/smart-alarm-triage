from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features.engineering import AlarmFeatureEngineer


def _base_steps():
    return [
        ("feature_engineering", AlarmFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ]


def build_random_forest_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        _base_steps() + [("classifier", RandomForestClassifier(**params))]
    )


def build_xgboost_pipeline(params: dict, n_classes: int) -> Pipeline:
    xgb_params = {k: v for k, v in params.items()}
    xgb_params["num_class"] = n_classes
    xgb_params["objective"] = "multi:softprob"
    return Pipeline(
        _base_steps() + [("classifier", XGBClassifier(**xgb_params))]
    )


def build_lightgbm_pipeline(params: dict) -> Pipeline:
    lgbm_params = {k: v for k, v in params.items()}
    lgbm_params["objective"] = "multiclass"
    return Pipeline(
        _base_steps() + [("classifier", LGBMClassifier(**lgbm_params))]
    )


PIPELINE_BUILDERS = {
    "random_forest": build_random_forest_pipeline,
    "xgboost": build_xgboost_pipeline,
    "lightgbm": build_lightgbm_pipeline,
}
