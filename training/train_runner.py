"""
Training runner.

Loads a model dataset, splits it, trains selected models, evaluates them,
saves models and logs experiment results.
"""

import pandas as pd
from pathlib import Path

from config.experiment_config import TARGET_CONFIG, TRAINING_CONFIG

from features.balancing import balance_by_undersampling

from training.splitting import (
    split_by_time_fraction,
    split_features_and_target,
)
from training.model_factory import get_model
from training.evaluation import (
    evaluate_classifier,
    print_evaluation_report,
    build_prediction_frame,
)
from training.model_store import save_model
from training.experiment_logger import log_experiment

from training.feature_importance import (
    get_builtin_feature_importance,
    get_permutation_feature_importance,
    save_feature_importance,
    print_top_features,
)


def train_single_model(
    model_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    model_name: str,
    target_column: str,
    train_fraction: float = 2 / 3,
    balance_train: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Trains and evaluates one model.
    """
    metadata_for_split_df = metadata_df.drop(
        columns=[target_column],
        errors="ignore",
    )

    train_model_df, test_model_df, train_metadata_df, test_metadata_df = split_by_time_fraction(
        model_df=model_df,
        metadata_df=metadata_for_split_df,
        train_fraction=train_fraction,
    )

    if balance_train:
        metadata_without_target_df = train_metadata_df.drop(
            columns=[target_column],
            errors="ignore",
        )

        train_with_metadata_df = pd.concat(
            [
                metadata_without_target_df.reset_index(drop=True),
                train_model_df.reset_index(drop=True),
            ],
            axis=1,
        )

        balanced_train_df = balance_by_undersampling(
            df=train_with_metadata_df,
            target_column=target_column,
            random_state=random_state,
        )

        metadata_columns = list(metadata_without_target_df.columns)

        train_metadata_df = balanced_train_df[metadata_columns].copy()
        train_model_df = balanced_train_df.drop(columns=metadata_columns)

    X_train, y_train = split_features_and_target(
        model_df=train_model_df,
        target_column=target_column,
    )

    X_test, y_test = split_features_and_target(
        model_df=test_model_df,
        target_column=target_column,
    )

    model = get_model(
        model_name=model_name,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    print_evaluation_report(model, X_train, y_train, label=f"{model_name} train")
    print_evaluation_report(model, X_test, y_test, label=f"{model_name} test")

    train_metrics = evaluate_classifier(
        model=model,
        X=X_train,
        y=y_train,
        label="train",
    )

    test_metrics = evaluate_classifier(
        model=model,
        X=X_test,
        y=y_test,
        label="test",
    )

    feature_names = list(X_train.columns)

    builtin_importance_df = get_builtin_feature_importance(
        model=model,
        feature_names=feature_names,
    )

    if not builtin_importance_df.empty:
        print(f"\nTop built-in feature importance: {model_name}")
        print_top_features(
            importance_df=builtin_importance_df,
            value_column="importance",
            top_n=20,
        )

        builtin_importance_path = save_feature_importance(
            importance_df=builtin_importance_df,
            model_name=model_name,
            importance_type="builtin",
        )
    else:
        builtin_importance_path = None

    permutation_importance_df = get_permutation_feature_importance(
        model=model,
        X=X_test,
        y=y_test,
        scoring="balanced_accuracy",
        n_repeats=10,
        random_state=random_state,
    )

    print(f"\nTop permutation feature importance: {model_name}")
    print_top_features(
        importance_df=permutation_importance_df,
        value_column="importance_mean",
        top_n=20,
    )

    permutation_importance_path = save_feature_importance(
        importance_df=permutation_importance_df,
        model_name=model_name,
        importance_type="permutation_test",
    )

    model_path = save_model(
        model=model,
        model_name=model_name,
    )

    log_experiment(
        model_name=model_name,
        target_column=target_column,
        model_path=model_path,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_count=X_train.shape[1],
    )

    prediction_df = build_prediction_frame(
        model=model,
        X=X_test,
        y=y_test,
        metadata_df=test_metadata_df,
        target_column=target_column,
    )

    prediction_dir = Path("artifacts/predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = prediction_dir / f"{model_name}_test_predictions.parquet"
    prediction_df.to_parquet(prediction_path, index=False)

    return {
        "model_name": model_name,
        "model": model,
        "model_path": model_path,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "prediction_path": str(prediction_path),
        "builtin_importance_path": builtin_importance_path,
        "permutation_importance_path": permutation_importance_path,
    }


def train_many_models(
    model_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    model_names: list[str],
    target_column: str,
    train_fraction: float = 2 / 3,
    balance_train: bool = True,
    random_state: int = 42,
) -> list[dict]:
    """
    Trains and evaluates multiple models.
    """
    results = []

    for model_name in model_names:
        print("\n" + "=" * 100)
        print(f"Training model: {model_name}")
        print("=" * 100)

        result = train_single_model(
            model_df=model_df,
            metadata_df=metadata_df,
            model_name=model_name,
            target_column=target_column,
            train_fraction=train_fraction,
            balance_train=balance_train,
            random_state=random_state,
        )

        results.append(result)

    return results


def main() -> None:
    dataset_dir = Path("datasets")

    model_df = pd.read_parquet(dataset_dir / "model_dataset.parquet")
    metadata_df = pd.read_parquet(dataset_dir / "model_metadata.parquet")

    target_column = TARGET_CONFIG.target_column

    results = train_many_models(
        model_df=model_df,
        metadata_df=metadata_df,
        model_names=TRAINING_CONFIG.model_names,
        target_column=target_column,
        train_fraction=TRAINING_CONFIG.train_fraction,
        balance_train=TRAINING_CONFIG.balance_train,
        random_state=TRAINING_CONFIG.random_state,
    )

    print("\nTraining complete.")
    for result in results:
        print(
            result["model_name"],
            "test balanced accuracy:",
            result["test_metrics"]["balanced_accuracy"],
        )


if __name__ == "__main__":
    main()