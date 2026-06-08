"""Script d'entraînement reproductible — appelé par DVC (dvc repro)."""

import json
import sys
import warnings
from pathlib import Path

import os

import dill
import mlflow
import numpy as np
import pandas as pd
import pendulum
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))

from settings.params import MODEL_NAME, MODEL_PARAMS, SEED, TIMEZONE
from src.make_dataset import load_data
from src.trainer import Trainer

try:
    import ppscore as pps
    USE_PPS = True
except ImportError:
    USE_PPS = False
    logger.warning("ppscore non disponible — utilisation des features par défaut")


def get_features(data: pd.DataFrame, target: str, threshold: float = 0.05):
    if USE_PPS:
        cols_to_drop = ["id", "yrsold", "yearbuilt", "yearremodadd", "garageyrblt"]
        pps_df = pps.predictors(
            df=data.drop([c for c in cols_to_drop if c in data.columns], axis=1),
            y=target, output="df", random_seed=SEED,
        )
        features = pps_df.loc[pps_df.ppscore.round(3) >= threshold, "x"].values.tolist()
        logger.info(f"Features PPS ({threshold}): {len(features)}")
        return features
    return MODEL_PARAMS["FEATURES"]


def main():
    PROJECT_DIR = Path(__file__).parent.parent
    MODEL_DIR   = Path(PROJECT_DIR, "models")
    DATA_DIR    = Path(PROJECT_DIR, "data", "output")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ─────────────────────────────────────────────────────────
    data = load_data(dataset_name="house_prices", columns_to_lower=True)
    data = data.astype({
        "overallqual": str, "overallcond": str, "garageyrblt": str,
        "yearbuilt": str, "yearremodadd": str, "mssubclass": str,
        "mosold": str, "yrsold": str,
    })
    data.to_parquet(Path(DATA_DIR, "house_prices.parquet"), index=False)
    logger.info(f"Données sauvegardées : {data.shape}")

    # ── Features ────────────────────────────────────────────────────────────
    TARGET        = MODEL_PARAMS["TARGET"]
    FEATURE_NAMES = get_features(data, TARGET, threshold=0.05)

    # ── Split ────────────────────────────────────────────────────────────────
    X = data[FEATURE_NAMES]
    y_log = np.log1p(data[TARGET].astype(float))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=SEED
    )

    # ── Meilleurs hyperparamètres (remplacer par les valeurs Optuna) ──────────
    best_params = {"alpha": 10.0}  # à remplacer après notebook 03

    # ── Entraînement ────────────────────────────────────────────────────────
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(str(Path(PROJECT_DIR, "notebooks", "mlruns")))
    mlflow.set_experiment("house_price_train_pipeline")

    numerical_transformer = [SimpleImputer(strategy="median"), RobustScaler()]
    categorical_transformer = [
        SimpleImputer(strategy="constant", fill_value="undefined"),
        OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
    ]

    with mlflow.start_run(run_name="train_pipeline"):
        trainer = Trainer(
            data=data.loc[:, FEATURE_NAMES + [TARGET]],
            target=TARGET,
            estimator=Ridge(**best_params),
            numerical_transformer=numerical_transformer,
            categorical_transformer=categorical_transformer,
            features=FEATURE_NAMES,
            test_size=0.2,
            random_state=SEED,
        )
        pipeline, train_m, test_m = trainer.train()

        preds = pipeline.predict(X_test)
        y_true = np.expm1(y_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "mae":  float(mean_absolute_error(y_true, preds)),
            "r2":   float(r2_score(y_true, preds)),
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # ── Sauvegarde ──────────────────────────────────────────────────────
        EXECUTION_DATE = pendulum.now(tz="UTC")
        model_path = Path(MODEL_DIR, f"{EXECUTION_DATE.strftime('%Y%m%d')}_{MODEL_NAME}")
        with open(model_path, "wb") as f:
            dill.dump(pipeline, f)

        # ── Métriques DVC ───────────────────────────────────────────────────
        with open(Path(PROJECT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info(f"Pipeline entraîné — RMSE={metrics['rmse']:.0f}$ | R²={metrics['r2']:.4f}")
    logger.info(f"Modèle sauvegardé : {model_path}")
    logger.info(f"Métriques DVC écrites : metrics.json")


if __name__ == "__main__":
    main()
