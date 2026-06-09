# Documentation complète — Laplace Immo

## Présentation du projet

**Objectif** : prédire le prix de vente de maisons individuelles à partir de leurs caractéristiques physiques et contractuelles.

**Dataset** : Ames Housing (Iowa, USA) — 1 460 observations, 81 variables (surface habitable, qualité générale, type de garage, année de construction, etc.). Chargé automatiquement via OpenML (`fetch_openml`), sans téléchargement manuel.

**Résultat final** : RMSE = **26 338 $** sur le jeu de test, R² = **0.891** — le modèle explique 89 % de la variance des prix.

---

## Structure du projet

```
house_price/
├── .github/workflows/ci.yml        # Pipeline CI/CD automatique
├── notebooks/                      # Analyse et expérimentations
│   ├── house_price_01_analyse.ipynb
│   ├── house_price_02_modeling.ipynb
│   ├── house_price_03_optimization.ipynb
│   └── house_price_04_preprocessing.ipynb
├── src/                            # Code source réutilisable
│   ├── make_dataset.py
│   ├── trainer.py
│   └── train_pipeline.py
├── settings/
│   └── params.py                   # Paramètres centralisés
├── tests/                          # Tests unitaires automatisés
│   ├── test_make_dataset.py
│   └── test_trainer.py
├── reports/                        # Graphiques de résultats
├── dvc.yaml                        # Pipeline reproductible DVC
├── metrics.json                    # Métriques finales
└── requirements.txt                # Dépendances Python
```

---

## Notebooks — déroulement de la démarche

### Notebook 01 — Analyse exploratoire (`house_price_01_analyse.ipynb`)

**Objectif** : comprendre les données avant de modéliser.

**Ce qui a été fait** :

- **Valeurs manquantes** : visualisation avec `missingno`. Certaines variables comme `PoolQC`, `Alley`, `Fence` ont plus de 80 % de valeurs manquantes — ces absences ont une signification métier (absence de piscine, pas d'allée), pas une erreur de collecte.
- **Variable cible `saleprice`** : distribution fortement asymétrique à droite (quelques maisons très chères tirent la moyenne vers le haut). Cette observation motive la **log-transformation** utilisée dans les notebooks suivants.
- **Variables catégorielles** : analyse des fréquences et boxplots par rapport à `saleprice`. On observe par exemple que `OverallQual` (qualité générale 1-10) est très discriminante.
- **Variables numériques** : distributions univariées (histogrammes) et bivariées (scatter plots vs `saleprice`). `GrLivArea` (surface habitable) est la variable la plus linéairement corrélée au prix.
- **Matrice de corrélation** : identification des variables redondantes et des variables à fort lien avec la cible.
- **Rapport de profiling** : `ydata-profiling` génère automatiquement un rapport HTML complet (`reports/house_price_profiling.html`) avec statistiques, alertes d'interactions, et détection de variables quasi-constantes.

**Motivation** : l'EDA (Exploratory Data Analysis) est indispensable avant la modélisation pour éviter de traiter de façon mécanique des données qu'on ne comprend pas. Elle révèle les transformations nécessaires et les features les plus informatives.

---

### Notebook 02 — Benchmark de modèles (`house_price_02_modeling.ipynb`)

**Objectif** : comparer systématiquement 20+ algorithmes pour identifier les meilleures approches.

**Ce qui a été fait** :

**Sélection des features via PPS (Predictive Power Score)**
Le PPS mesure la capacité prédictive d'une variable X vers une variable Y, en tenant compte des relations non linéaires (contrairement à la corrélation de Pearson qui ne détecte que les relations linéaires). Seules les features avec un score ≥ 0.05 sont conservées.

**Double version de la cible**
Chaque modèle est testé deux fois :
- `y_raw` : `saleprice` brut
- `y_log` : `log1p(saleprice)` — transformation logarithmique pour normaliser la distribution

**Résultat clé** : la log-transformation améliore massivement certains modèles :
- SVR (noyau RBF) : gain de **54 000 $** de RMSE
- XGBoost : gain de **6 700 $** de RMSE

**Modèles testés** :

| Famille | Algorithmes |
|---------|------------|
| Baseline | DummyRegressor (moyenne) |
| Linéaires | Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor |
| Arbres | DecisionTree, ExtraTrees, RandomForest |
| Gradient Boosting | GradientBoosting, HistGradientBoosting, AdaBoost |
| Boosting avancé | XGBoost, LightGBM, CatBoost |
| Autres | SVR (RBF), KNN, MLP (réseau de neurones) |
| Ensembles | VotingRegressor, StackingRegressor |

**Comparaison des scalers** : RobustScaler vs StandardScaler vs MinMaxScaler. Le RobustScaler (utilise médiane et IQR au lieu de moyenne/écart-type) est plus robuste aux outliers — important sur des données immobilières où quelques maisons très chères faussent la normalisation.

**54 runs MLFlow** loggués automatiquement pour traçabilité complète.

**Meilleur résultat** : SVR RBF avec log-transform, RMSE ≈ 28 600 $.

**Motivation** : tester un seul modèle est une erreur méthodologique. Seul un benchmark exhaustif permet de justifier le choix final de l'algorithme avec des données empiriques.

---

### Notebook 03 — Optimisation des hyperparamètres (`house_price_03_optimization.ipynb`)

**Objectif** : pousser les performances des meilleurs modèles en cherchant leurs hyperparamètres optimaux.

**Ce qui a été fait** :

**Optuna** — framework d'optimisation bayésienne des hyperparamètres. Contrairement à une GridSearch qui teste toutes les combinaisons de façon exhaustive, Optuna apprend des essais précédents pour diriger la recherche vers les zones prometteuses de l'espace des hyperparamètres.

- **60 trials** par modèle avec pruning (arrêt anticipé des essais non prometteurs)
- **5 modèles optimisés** : LightGBM, XGBoost, CatBoost, SVR, HistGradientBoosting
- **Nested runs MLFlow** : chaque trial Optuna est un run enfant dans l'expérience du modèle parent

**Exemple d'espace de recherche pour LightGBM** :
- `n_estimators` : entre 200 et 1 200
- `learning_rate` : entre 0.005 et 0.3 (échelle log)
- `max_depth` : entre 3 et 10
- `num_leaves` : entre 20 et 150
- `reg_alpha`, `reg_lambda` : régularisation L1/L2

**Ensembles** :
- `VotingRegressor` : moyenne pondérée des prédictions des meilleurs modèles
- `StackingRegressor` : un méta-modèle (Ridge) apprend à combiner les prédictions des modèles de base

**Analyse du meilleur modèle** :
- Feature importances (gain d'information LightGBM) : `OverallQual`, `GrLivArea`, `TotalBsmtSF` sont les variables les plus importantes
- Analyse des résidus : vérification que les erreurs sont bien distribuées autour de zéro sans pattern systématique

**Modèle sauvegardé** avec `dill` (sérialisation du pipeline sklearn complet, préprocesseur inclus).

**Motivation** : les hyperparamètres par défaut sont rarement optimaux. Optuna permet une exploration intelligente en 10× moins de temps qu'une GridSearch exhaustive.

---

### Notebook 04 — Ingénierie des prétraitements (`house_price_04_preprocessing.ipynb`)

**Objectif** : aller au-delà du prétraitement de base pour extraire plus d'information des données.

**6 axes évalués par validation croisée 5-fold** (CV-RMSE = métrique objective, indépendante du jeu de test) :

| Axe | Description | Gain RMSE |
|-----|-------------|-----------|
| **Baseline** | Imputation médiane + OneHotEncoder | référence |
| Axe 1 | Imputation sémantique (NaN = "absent" pour PoolQC, etc.) | +129 $ |
| Axe 2 | Winsorizing — écrêtage des outliers à P1/P99 | +102 $ |
| Axe 3 | Yeo-Johnson — correction d'asymétrie des variables numériques | +101 $ |
| Axe 4 | OrdinalEncoder pour variables de qualité (Ex > Gd > TA > Fa > Po) | +101 $ |
| **Axe 5** | **TargetEncoder pour variables haute cardinalité (Neighborhood, etc.)** | **+556 $** |
| **Axe 6** | **Feature interactions : `qual_x_surface`, `total_sf`, `bath_score`** | **+949 $** |

**Pipeline combiné final** : CV-RMSE **26 418 $** — gain de **+1 237 $** par rapport au baseline.

**Détail des motivations** :

- *Imputation sémantique* : pour `PoolQC`, une valeur manquante signifie "pas de piscine", pas une donnée inconnue. Imputer par "absent" est plus fidèle à la réalité qu'imputer par la valeur la plus fréquente.
- *Winsorizing* : les outliers (maisons à 750k$ dans un dataset médian à 180k$) perturbent l'apprentissage. Les écrêter à P99 limite leur influence sans les supprimer.
- *Yeo-Johnson* : les variables numériques comme `LotArea` ou `GrLivArea` sont très asymétriques. Les normaliser aide les modèles linéaires et le SVR.
- *OrdinalEncoder* : `OverallQual` = "Excellent" > "Good" > "Average" a un ordre naturel. L'encoder numériquement (5 > 4 > 3) est plus informative qu'un OneHotEncoding.
- *TargetEncoder* : `Neighborhood` a 25 modalités. OneHotEncoding crée 25 colonnes sparse. TargetEncoder remplace chaque modalité par la moyenne de `saleprice` dans ce quartier — une information directement pertinente.
- *Feature interactions* : `qual_x_surface = OverallQual × GrLivArea` capture le fait qu'une grande maison de mauvaise qualité peut valoir moins qu'une petite maison de très bonne qualité.

---

## Code source

### `src/make_dataset.py` — Chargement des données

Fonction unique `load_data()` qui :
1. Valide que le nom du dataset n'est pas vide (sinon `ValueError`)
2. Charge les données depuis OpenML via `fetch_openml`
3. Met les colonnes en minuscules si demandé
4. Ajoute 3 features temporelles calculées :
   - `building_age` = année de vente − année de construction
   - `remodel_age` = année de vente − année de la dernière rénovation
   - `garage_age` = année de vente − année de construction du garage

**Motivation** : centraliser le chargement dans une fonction testable évite la duplication de code entre les notebooks et le pipeline de production. Les features temporelles captent la dépréciation des biens.

---

### `src/trainer.py` — Classe `Trainer`

Abstraction du pipeline d'entraînement sklearn. Reçoit un estimateur quelconque + des transformeurs + les données, et gère automatiquement :

- Détection des colonnes numériques vs catégorielles
- Construction d'un `ColumnTransformer` (préprocesseur séparé par type de variable)
- Assemblage du `Pipeline` (préprocesseur → estimateur)
- Split train/test
- Calcul des métriques (RMSE, MAE, R²)

**Méthodes** :
- `define_pipeline()` : construit le pipeline sans entraîner
- `fit()` : entraîne le pipeline
- `train()` : `fit()` + retourne `(pipeline, métriques_train, métriques_test)`
- `predict(X)` : prédit sur de nouvelles données
- `score(X, y)` : retourne le R²

**Motivation** : éviter de réécrire le même code sklearn à chaque notebook. Un seul appel `Trainer(...).train()` remplace 30 lignes de boilerplate.

---

### `src/train_pipeline.py` — Script DVC

Script autonome exécuté par DVC (`dvc repro` ou directement `python src/train_pipeline.py`). Reproduit l'entraînement de bout en bout :

1. Charge les données avec `load_data()`
2. Sauvegarde les données en Parquet (`data/output/house_prices.parquet`)
3. Sélectionne les features via PPS
4. Entraîne un pipeline Ridge avec `Trainer`
5. Calcule les métriques et les log dans MLFlow
6. Sauvegarde le modèle avec `dill`
7. Écrit `metrics.json` (lu par DVC pour le suivi des métriques)

**Motivation** : séparer l'expérimentation (notebooks) de la reproductibilité (script). DVC peut comparer les métriques entre deux runs et détecter si un changement de code dégrade le modèle.

---

### `settings/params.py` — Paramètres centralisés

Contient toutes les constantes partagées :
- `SEED = 43` : graine aléatoire pour la reproductibilité
- `TARGET = "saleprice"` : variable cible
- `MODEL_NAME` : nom du fichier modèle
- `MODEL_PARAMS["FEATURES"]` : liste des features par défaut (si PPS non disponible)

**Motivation** : un seul fichier à modifier pour changer un paramètre global. Évite les "magic numbers" éparpillés dans le code.

---

## Tests unitaires

11 tests répartis en 2 fichiers, exécutés automatiquement à chaque push.

### `tests/test_make_dataset.py` (5 tests)

| Test | Ce qu'il vérifie |
|------|-----------------|
| `test_load_data_returns_dataframe` | La fonction retourne bien un `pd.DataFrame` |
| `test_load_data_lowercase_columns` | Les colonnes sont en minuscules si `columns_to_lower=True` |
| `test_load_data_feature_engineering` | Les colonnes `building_age`, `remodel_age`, `garage_age` sont présentes |
| `test_load_data_raises_on_empty_name` | `ValueError` levée si `dataset_name=""` |
| `test_load_data_raises_on_none_name` | `ValueError` levée si `dataset_name=None` |

Les tests utilisent `unittest.mock.patch` pour simuler `fetch_openml` sans appel réseau — les tests restent rapides et indépendants d'internet.

### `tests/test_trainer.py` (6 tests)

| Test | Ce qu'il vérifie |
|------|-----------------|
| `test_define_pipeline_returns_pipeline` | `define_pipeline()` retourne un objet sklearn `Pipeline` |
| `test_fit_splits_data` | Après `fit()`, `X_train` et `X_test` sont non nuls |
| `test_train_returns_metrics` | `train()` retourne un dict avec `rmse`, `mae`, `r2` |
| `test_predict_returns_correct_shape` | `predict()` retourne un tableau de la bonne longueur |
| `test_train_test_size_respected` | La taille du test set correspond bien au `test_size` configuré |
| `test_score_returns_float` | `score()` retourne un flottant (le R²) |

**Motivation** : les tests unitaires détectent les régressions (quand une modification casse quelque chose qui fonctionnait). Sans tests, un refactoring peut introduire des bugs silencieux.

---

## MLOps

### MLFlow — Tracking des expériences

MLFlow enregistre automatiquement pour chaque run :
- Les **hyperparamètres** (`alpha`, `n_estimators`, `learning_rate`, etc.)
- Les **métriques** (RMSE, MAE, R²)
- Les **artefacts** (graphiques, modèles)

L'interface graphique (`mlflow ui --port 5001` depuis le dossier `notebooks/`) permet de :
- Comparer visuellement tous les runs
- Filtrer par métrique pour trouver le meilleur modèle
- Visualiser l'évolution des métriques pendant l'optimisation Optuna

Les runs Optuna utilisent des **nested runs** : chaque trial est un run enfant dans le run parent du modèle, ce qui permet de voir à la fois la progression de l'optimisation et le résultat final.

**Motivation** : sans tracking, on perd les résultats des expériences passées. MLFlow est l'équivalent d'un cahier de laboratoire structuré et automatique.

### DVC — Versionnage des données et du pipeline

DVC (Data Version Control) fonctionne comme git mais pour les fichiers lourds (données, modèles) et les pipelines.

Le fichier `dvc.yaml` définit deux étapes :

```yaml
stages:
  load_data:
    cmd: python src/make_dataset.py
    deps: [src/make_dataset.py, settings/params.py]
    outs: [data/output/house_prices.parquet]

  train:
    cmd: python src/train_pipeline.py
    deps: [src/train_pipeline.py, src/trainer.py, data/output/house_prices.parquet]
    outs: [models/]
    metrics: [metrics.json]
```

DVC :
- Détecte automatiquement si les dépendances ont changé et ne ré-exécute que ce qui est nécessaire
- Permet de comparer `metrics.json` entre deux branches git (`dvc metrics diff`)
- Assure la reproductibilité : en clonant le repo et en exécutant `dvc repro`, on obtient exactement le même modèle

**Motivation** : garantir que les résultats sont reproductibles, même 6 mois plus tard sur une autre machine.

---

## CI/CD — GitHub Actions

Le fichier `.github/workflows/ci.yml` déclenche automatiquement à chaque `git push` deux jobs séquentiels :

### Job 1 — Tests unitaires
```
pytest tests/ -v
```
Si un test échoue, le job 2 ne démarre pas.

### Job 2 — Pipeline d'entraînement (conditionnel)
```
python src/train_pipeline.py
```
Exécute l'entraînement complet et uploade `metrics.json` comme artefact du run GitHub.

**Motivation** : la CI garantit que le code reste fonctionnel à tout moment. Si un collaborateur pousse une modification qui casse les tests ou fait planter le pipeline, l'équipe est alertée immédiatement avant que le problème n'atteigne la production.

---

## Choix techniques et leurs motivations

| Choix | Alternative | Motivation |
|-------|-------------|------------|
| `log1p(saleprice)` comme cible | cible brute | Normalise la distribution asymétrique, améliore SVR de 54k$ |
| PPS pour la sélection de features | corrélation Pearson | Capture les relations non linéaires |
| RobustScaler | StandardScaler | Plus résistant aux outliers immobiliers |
| TargetEncoder pour haute cardinalité | OneHotEncoder | Évite l'explosion dimensionnelle (25 colonnes → 1) |
| `dill` pour la sérialisation | `pickle` | Supporte les lambdas et objets Python complexes que pickle refuse |
| `loguru` pour les logs | `print()` | Format structuré avec niveau (INFO/WARNING), timestamp automatique |
| `pendulum` pour les dates | `datetime` | API plus intuitive, gestion des fuseaux horaires simplifiée |

---

## Résultats détaillés

### Évolution des performances

| Étape | RMSE | R² |
|-------|------|-----|
| Baseline (DummyRegressor — moyenne) | ~79 000 $ | 0.00 |
| Meilleur modèle brut (SVR RBF, sans log) | ~83 000 $ | — |
| Meilleur modèle avec log-transform (SVR RBF) | ~28 600 $ | 0.857 |
| Après optimisation Optuna (LightGBM) | ~28 624 $ | 0.871 |
| Après ingénierie prétraitement (LightGBM) | **26 338 $** | **0.891** |

### Features les plus importantes (LightGBM, gain d'information)

1. `OverallQual` — Qualité générale de la maison (1-10)
2. `GrLivArea` — Surface habitable au-dessus du sol
3. `TotalBsmtSF` — Surface totale du sous-sol
4. `qual_x_surface` — Interaction qualité × surface (feature engineered)
5. `Neighborhood_enc` — Quartier (TargetEncoded)
6. `GarageCars` — Capacité du garage

---

## Comment reproduire

### Installation
```bash
git clone https://github.com/kira9292/house-price.git
cd house-price
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Lancer les tests
```bash
pytest tests/ -v
```

### Réentraîner le modèle
```bash
python src/train_pipeline.py
```

### Visualiser les expériences MLFlow
```bash
cd notebooks
mlflow ui --port 5001
# Ouvrir http://127.0.0.1:5001
```

### Vérifier les métriques
```bash
cat metrics.json
```
