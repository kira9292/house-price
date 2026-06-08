# Laplace Immo — Prédiction des Prix Immobiliers

Projet Data Science — prédiction du prix de vente de maisons individuelles sur le dataset **Ames Housing** (Iowa, USA, 1 460 observations × 81 variables).

[![CI](https://github.com/kira9292/house-price/actions/workflows/ci.yml/badge.svg)](https://github.com/kira9292/house-price/actions/workflows/ci.yml)

---

## Résultats

| Métrique | Valeur |
|----------|--------|
| RMSE (test) | **26 338 $** |
| MAE (test) | 17 058 $ |
| R² (test) | **0.891** |
| RMSE CV (5-fold) | 26 418 $ |

Modèle final : **LightGBM** avec prétraitement avancé (TargetEncoder, interactions de features, Yeo-Johnson).

---

## Structure du projet

```
house_price/
├── .github/workflows/ci.yml        # GitHub Actions — tests + pipeline DVC
├── notebooks/
│   ├── house_price_01_analyse.ipynb       # EDA & analyse exploratoire
│   ├── house_price_02_modeling.ipynb      # Benchmark 20+ modèles (54 runs)
│   ├── house_price_03_optimization.ipynb  # Optimisation Optuna (60 trials/modèle)
│   └── house_price_04_preprocessing.ipynb # 6 axes de prétraitement comparés
├── src/
│   ├── make_dataset.py             # Chargement & feature engineering
│   ├── trainer.py                  # Classe Trainer (pipeline sklearn)
│   └── train_pipeline.py           # Script DVC — entraînement reproductible
├── settings/params.py              # Paramètres centralisés
├── tests/
│   ├── test_make_dataset.py        # 5 tests unitaires
│   └── test_trainer.py             # 6 tests unitaires
├── reports/                        # Graphiques (benchmark, résidus, features)
├── models/                         # Modèles sérialisés (.dill)
├── dvc.yaml                        # Pipeline DVC (load_data → train)
├── metrics.json                    # Métriques DVC (RMSE, MAE, R²)
└── requirements.txt
```

---

## Démarche

### 1. EDA (`notebook 01`)
- Analyse des valeurs manquantes, distributions et corrélations
- Rapport automatique avec `ydata-profiling`
- Identification des variables à fort pouvoir prédictif via **PPS (Predictive Power Score)**

### 2. Benchmark de modèles (`notebook 02`)
54 runs MLFlow comparant 20+ algorithmes sur deux espaces cibles :

| Famille | Modèles |
|---------|---------|
| Linéaires | Ridge, Lasso, ElasticNet, BayesianRidge, Huber |
| Arbres | DecisionTree, ExtraTrees, RandomForest, GradBoost, HistGradBoost |
| Boosting | XGBoost, LightGBM, CatBoost, AdaBoost |
| Autres | SVR, KNN, MLP |
| Ensembles | VotingRegressor, StackingRegressor |

**Découverte clé** : la log-transformation de la cible (`log1p(saleprice)`) améliore SVR de 54 000 $ et XGBoost de 6 700 $.

### 3. Optimisation des hyperparamètres (`notebook 03`)
- **Optuna** — 60 trials par modèle avec pruning Median
- Tracking MLFlow avec nested runs
- Top 5 modèles optimisés : LightGBM, XGBoost, CatBoost, SVR, HistGradBoost

### 4. Ingénierie des prétraitements (`notebook 04`)
6 axes évalués par validation croisée 5-fold :

| Axe | Gain RMSE |
|-----|-----------|
| Imputation sémantique | baseline |
| Winsorizing (outliers) | +129 $ |
| Yeo-Johnson | +102 $ |
| OrdinalEncoder variables ordonnées | +101 $ |
| **TargetEncoder (haute cardinalité)** | **+556 $** |
| **Feature interactions** | **+949 $** |

Pipeline combiné final : CV-RMSE **26 418 $** (+1 237 $ vs baseline).

---

## Installation

```bash
git clone https://github.com/kira9292/house-price.git
cd house-price
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer les tests

```bash
pytest tests/ -v
```

## Reproduire le pipeline d'entraînement

```bash
python src/train_pipeline.py
```

## Visualiser les expériences MLFlow

```bash
cd notebooks
mlflow ui
```

---

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci.yml`) exécute à chaque push :

1. **Tests unitaires** — `pytest tests/ -v` (11 tests)
2. **Pipeline d'entraînement** — `python src/train_pipeline.py` + upload de `metrics.json`

---

## Stack technique

| Catégorie | Outils |
|-----------|--------|
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| Optimisation | Optuna |
| MLOps | MLFlow, DVC |
| Feature selection | ppscore |
| Encodage | category-encoders (TargetEncoder) |
| Sérialisation | dill |
| Tests | pytest |
| CI/CD | GitHub Actions |
