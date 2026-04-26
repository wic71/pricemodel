"""
Experiment configuration.

Single source of truth for dataset target configuration and training settings.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DatasetConfig:
    tickers: list[str]
    benchmark_ticker: str | None
    start_date: date
    end_date: date
    interval: str
    provider_name: str


@dataclass(frozen=True)
class TargetConfig:
    target_horizon: int
    target_type: str
    threshold: float
    relative_to_benchmark: bool

    @property
    def target_column(self) -> str:
        """
        Returns the expected target column name.
        """
        if self.relative_to_benchmark:
            return f"target_{self.target_horizon}d_relative_binary"

        if self.target_type == "binary":
            return f"target_{self.target_horizon}d_binary"

        if self.target_type == "three_class":
            return f"target_{self.target_horizon}d_3class"

        raise ValueError(f"Unknown target_type: {self.target_type}")


@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float
    balance_train: bool
    random_state: int
    model_names: list[str]


DATASET_CONFIG = DatasetConfig(
    tickers=[
        "VOLV-B.ST",
        "SAAB-B.ST",
        "INVE-B.ST",
        "ERIC-B.ST",
        "ATCO-A.ST",
    ],
    benchmark_ticker="^OMX",
    start_date=date(2000, 1, 1),
    end_date=date(2026, 4, 1),
    interval="1d",
    provider_name="yfinance",
    
)


TARGET_CONFIG = TargetConfig(
    target_horizon=20,
    target_type="binary",
    threshold=0.005,
    relative_to_benchmark=False,
)


TRAINING_CONFIG = TrainingConfig(
    train_fraction=2 / 3,
    balance_train=False,
    random_state=42,
    model_names=[
        "dummy_most_frequent",
        "dummy_stratified",
        "random_forest",
        "hist_gradient_boosting",
        "extra_trees",
    ],
)