#!/usr/bin/env python3
"""Training script for Snow Forecast model.

This script provides a reproducible training pipeline.
Run with: python scripts/train.py --help
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}")


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data.

    Args:
        data_path: Path to data directory

    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = data_path / "train.parquet"
    test_path = data_path / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        logger.warning("Data files not found. Creating synthetic data for demo.")
        # Create synthetic data for demonstration
        n_train = 10000
        n_test = 2000
        n_features = 20

        train_df = pd.DataFrame(
            np.random.randn(n_train, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        train_df["target"] = (
            train_df["feature_0"] * 10
            + train_df["feature_1"] * 5
            + np.random.randn(n_train) * 2
        )

        test_df = pd.DataFrame(
            np.random.randn(n_test, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        test_df["target"] = (
            test_df["feature_0"] * 10
            + test_df["feature_1"] * 5
            + np.random.randn(n_test) * 2
        )

        return train_df, test_df

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def train_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs,
):
    """Train a model.

    Args:
        model_type: Type of model ('linear', 'gradient_boosting', 'lstm')
        X_train: Training features
        y_train: Training targets
        **kwargs: Additional model parameters

    Returns:
        Trained model
    """
    if model_type == "linear":
        from snowforecast.models import LinearRegressionModel
        model = LinearRegressionModel(**kwargs)

    elif model_type == "gradient_boosting":
        try:
            from snowforecast.models import GradientBoostingModel
            model = GradientBoostingModel(**kwargs)
        except ImportError:
            logger.error("GradientBoostingModel requires [models] dependencies")
            raise

    elif model_type == "lstm":
        try:
            from snowforecast.models import SequenceModel
            model = SequenceModel(model_type="lstm", **kwargs)
        except ImportError:
            logger.error("SequenceModel requires [deeplearning] dependencies")
            raise

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    logger.info("Training complete")

    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a trained model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary of metrics
    """
    from snowforecast.evaluation import compute_prd_metrics

    y_pred = model.predict(X_test)
    metrics = compute_prd_metrics(y_test.values, y_pred)

    logger.info(f"RMSE: {metrics.rmse:.2f}")
    logger.info(f"MAE: {metrics.mae:.2f}")
    logger.info(f"Bias: {metrics.bias:+.2f}")

    return {
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "bias": metrics.bias,
        "f1": metrics.f1,
    }


def save_model(model, output_path: Path) -> None:
    """Save trained model.

    Args:
        model: Trained model
        output_path: Path to save model
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Snow Forecast model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing training data",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="linear",
        choices=["linear", "gradient_boosting", "lstm"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Set random seeds
    set_seeds(args.seed)

    # Load data
    logger.info("Loading data...")
    train_df, test_df = load_data(args.data_dir)

    # Prepare features and targets
    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {len(feature_cols)}")

    # Train model
    model = train_model(args.model_type, X_train, y_train)

    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"{args.model_type}_{timestamp}.pkl"
    save_model(model, model_path)

    logger.info("Training complete!")
    logger.info(f"Final metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    main()
