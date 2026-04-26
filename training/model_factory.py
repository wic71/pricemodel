"""
Model factory.

Creates sklearn-compatible models by name.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


def get_model(
    model_name: str,
    random_state: int = 42,
):
    """
    Returns a model instance by name.
    """
    if model_name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=random_state)

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=25,
            min_samples_split=50,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=25,
            min_samples_split=50,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            min_samples_leaf=50,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


SUPPORTED_MODELS = [
    "dummy_most_frequent",
    "dummy_stratified",
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]