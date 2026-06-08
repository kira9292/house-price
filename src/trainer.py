"""Model training pipeline builder."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline


class Trainer:
    """Builds and trains a sklearn Pipeline with separate numeric/categorical preprocessing."""

    def __init__(self, data, target, estimator,
                 numerical_transformer=None,
                 categorical_transformer=None,
                 features=None,
                 test_size=0.25,
                 cv=5,
                 random_state=43):
        self.data = data
        self.target = target
        self.estimator = estimator
        self.numerical_transformer = numerical_transformer
        self.categorical_transformer = categorical_transformer
        self.features = features
        self.test_size = test_size
        self.cv = cv
        self.random_state = random_state
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def define_pipeline(self):
        feature_cols = (self.features
                        if self.features is not None
                        else [c for c in self.data.columns if c != self.target])

        X = self.data[feature_cols]
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

        transformers = []
        if numeric_cols and self.numerical_transformer:
            steps = self.numerical_transformer if isinstance(self.numerical_transformer, list) else [self.numerical_transformer]
            transformers.append(("num", make_pipeline(*steps), numeric_cols))

        if categorical_cols and self.categorical_transformer:
            steps = self.categorical_transformer if isinstance(self.categorical_transformer, list) else [self.categorical_transformer]
            transformers.append(("cat", make_pipeline(*steps), categorical_cols))

        preprocessor = ColumnTransformer(transformers, remainder="passthrough")

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", self.estimator),
        ])
        return self.pipeline

    def fit(self):
        if self.pipeline is None:
            self.define_pipeline()

        feature_cols = (self.features
                        if self.features is not None
                        else [c for c in self.data.columns if c != self.target])

        X = self.data[feature_cols]
        y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        self.pipeline.fit(self.X_train, self.y_train)
        return self

    def train(self):
        """Fit the pipeline and return (pipeline, train_metrics, test_metrics)."""
        self.fit()

        def _metrics(X, y):
            preds = self.pipeline.predict(X)
            return {
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "mae":  float(mean_absolute_error(y, preds)),
                "r2":   float(r2_score(y, preds)),
            }

        return self.pipeline, _metrics(self.X_train, self.y_train), _metrics(self.X_test, self.y_test)

    def predict(self, X=None):
        X = X if X is not None else self.X_test
        return self.pipeline.predict(X)

    def score(self, X=None, y=None):
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        return self.pipeline.score(X, y)
