"""Evaluation metrics for snowfall prediction.

This module provides metrics tailored for evaluating snow depth and
snowfall predictions:

- RMSE: Root Mean Square Error for continuous snow depth predictions
- MAE: Mean Absolute Error for continuous predictions
- Bias: Mean Error (systematic over/under prediction)
- F1-score: For binary snowfall event classification (>2.5cm / 1 inch)

All metrics handle NaN values appropriately by excluding them from calculations.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Snowfall event threshold (2.5cm = ~1 inch)
SNOWFALL_EVENT_THRESHOLD_CM = 2.5


def rmse(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Calculate Root Mean Square Error.

    RMSE gives higher weight to large errors, making it useful for
    detecting significant prediction failures.

    Args:
        y_true: Actual values (ground truth).
        y_pred: Predicted values.

    Returns:
        RMSE value. Returns NaN if no valid pairs.

    Example:
        >>> y_true = np.array([10.0, 20.0, 30.0])
        >>> y_pred = np.array([12.0, 18.0, 32.0])
        >>> rmse(y_true, y_pred)
        2.309...
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        logger.warning("No valid values for RMSE calculation")
        return np.nan

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    squared_errors = (y_true_valid - y_pred_valid) ** 2
    return float(np.sqrt(np.mean(squared_errors)))


def mae(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Calculate Mean Absolute Error.

    MAE treats all errors equally and is more robust to outliers than RMSE.
    Useful for understanding typical prediction error magnitude.

    Args:
        y_true: Actual values (ground truth).
        y_pred: Predicted values.

    Returns:
        MAE value. Returns NaN if no valid pairs.

    Example:
        >>> y_true = np.array([10.0, 20.0, 30.0])
        >>> y_pred = np.array([12.0, 18.0, 32.0])
        >>> mae(y_true, y_pred)
        2.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        logger.warning("No valid values for MAE calculation")
        return np.nan

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    absolute_errors = np.abs(y_true_valid - y_pred_valid)
    return float(np.mean(absolute_errors))


def bias(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Calculate Mean Error (Bias).

    Positive bias indicates systematic over-prediction.
    Negative bias indicates systematic under-prediction.

    Args:
        y_true: Actual values (ground truth).
        y_pred: Predicted values.

    Returns:
        Mean error (y_pred - y_true). Returns NaN if no valid pairs.

    Example:
        >>> y_true = np.array([10.0, 20.0, 30.0])
        >>> y_pred = np.array([12.0, 22.0, 32.0])  # Over-predicting by 2
        >>> bias(y_true, y_pred)
        2.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        logger.warning("No valid values for Bias calculation")
        return np.nan

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    return float(np.mean(y_pred_valid - y_true_valid))


def f1_score_snowfall(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
) -> float:
    """Calculate F1-score for snowfall events.

    Converts continuous predictions to binary (snowfall event vs no event)
    using the specified threshold, then calculates F1-score.

    F1-score is the harmonic mean of precision and recall, balancing
    both false positives (predicting snow when there wasn't) and
    false negatives (missing actual snow events).

    Args:
        y_true: Actual snowfall values in cm.
        y_pred: Predicted snowfall values in cm.
        threshold_cm: Threshold for defining a snowfall event.
            Default is 2.5cm (~1 inch).

    Returns:
        F1-score between 0 and 1. Returns NaN if no valid pairs
        or if there are no positive cases.

    Example:
        >>> y_true = np.array([0.0, 5.0, 10.0, 1.0])  # Events: 2
        >>> y_pred = np.array([1.0, 6.0, 8.0, 3.0])   # Correct: 2, FP: 1
        >>> f1_score_snowfall(y_true, y_pred)
        0.8
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        logger.warning("No valid values for F1 calculation")
        return np.nan

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    # Convert to binary
    y_true_binary = (y_true_valid >= threshold_cm).astype(int)
    y_pred_binary = (y_pred_valid >= threshold_cm).astype(int)

    # Calculate confusion matrix components
    true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    # Calculate precision and recall
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        # No actual positive cases
        logger.warning("No positive cases in y_true for F1 calculation")
        return np.nan

    recall = true_positives / (true_positives + false_negatives)

    # Calculate F1
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1)


def precision_snowfall(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
) -> float:
    """Calculate precision for snowfall events.

    Precision = TP / (TP + FP): Of all predicted snow events, how many
    were correct?

    Args:
        y_true: Actual snowfall values in cm.
        y_pred: Predicted snowfall values in cm.
        threshold_cm: Threshold for defining a snowfall event.

    Returns:
        Precision between 0 and 1. Returns NaN if no predictions.

    Example:
        >>> y_true = np.array([0.0, 5.0, 10.0])
        >>> y_pred = np.array([3.0, 6.0, 8.0])  # 3 predictions, 2 correct
        >>> precision_snowfall(y_true, y_pred)
        0.666...
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return np.nan

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    y_true_binary = (y_true_valid >= threshold_cm).astype(int)
    y_pred_binary = (y_pred_valid >= threshold_cm).astype(int)

    true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))

    if true_positives + false_positives == 0:
        logger.warning("No positive predictions for precision calculation")
        return np.nan

    return float(true_positives / (true_positives + false_positives))


def recall_snowfall(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
) -> float:
    """Calculate recall for snowfall events.

    Recall = TP / (TP + FN): Of all actual snow events, how many did
    we correctly predict?

    Args:
        y_true: Actual snowfall values in cm.
        y_pred: Predicted snowfall values in cm.
        threshold_cm: Threshold for defining a snowfall event.

    Returns:
        Recall between 0 and 1. Returns NaN if no actual events.

    Example:
        >>> y_true = np.array([0.0, 5.0, 10.0])  # 2 events
        >>> y_pred = np.array([0.0, 6.0, 1.0])   # Caught 1, missed 1
        >>> recall_snowfall(y_true, y_pred)
        0.5
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return np.nan

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    y_true_binary = (y_true_valid >= threshold_cm).astype(int)
    y_pred_binary = (y_pred_valid >= threshold_cm).astype(int)

    true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    if true_positives + false_negatives == 0:
        logger.warning("No positive cases for recall calculation")
        return np.nan

    return float(true_positives / (true_positives + false_negatives))


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    snowfall_threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
) -> dict[str, float]:
    """Compute all standard metrics for snowfall predictions.

    Convenience function that computes RMSE, MAE, Bias, and F1-score
    in one call.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        snowfall_threshold_cm: Threshold for F1-score calculation.

    Returns:
        Dictionary with all metrics:
            - rmse: Root Mean Square Error
            - mae: Mean Absolute Error
            - bias: Mean Error
            - f1: F1-score for snowfall events
            - precision: Precision for snowfall events
            - recall: Recall for snowfall events

    Example:
        >>> metrics = compute_all_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.2f}")
        >>> print(f"F1: {metrics['f1']:.2f}")
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "f1": f1_score_snowfall(y_true, y_pred, snowfall_threshold_cm),
        "precision": precision_snowfall(y_true, y_pred, snowfall_threshold_cm),
        "recall": recall_snowfall(y_true, y_pred, snowfall_threshold_cm),
    }


def compute_metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    snowfall_threshold_cm: float = SNOWFALL_EVENT_THRESHOLD_CM,
) -> pd.DataFrame:
    """Compute metrics grouped by a column (e.g., station_id).

    Useful for understanding model performance across different
    locations or time periods.

    Args:
        df: DataFrame with predictions.
        group_col: Column to group by (e.g., "station_id", "month").
        y_true_col: Column name for true values.
        y_pred_col: Column name for predicted values.
        snowfall_threshold_cm: Threshold for F1-score.

    Returns:
        DataFrame with one row per group and columns for each metric.

    Example:
        >>> df = pd.DataFrame({
        ...     "station_id": ["A", "A", "B", "B"],
        ...     "y_true": [10, 20, 15, 25],
        ...     "y_pred": [12, 18, 14, 26],
        ... })
        >>> metrics_by_station = compute_metrics_by_group(df, "station_id")
    """
    results = []

    for group_val, group_df in df.groupby(group_col):
        y_true = group_df[y_true_col].values
        y_pred = group_df[y_pred_col].values

        metrics = compute_all_metrics(y_true, y_pred, snowfall_threshold_cm)
        metrics[group_col] = group_val
        metrics["n_samples"] = len(group_df)
        results.append(metrics)

    result_df = pd.DataFrame(results)

    # Reorder columns to put group_col first
    cols = [group_col, "n_samples"] + [
        c for c in result_df.columns if c not in [group_col, "n_samples"]
    ]
    return result_df[cols]


# Metric registry for dynamic access
METRICS: dict[str, Callable] = {
    "rmse": rmse,
    "mae": mae,
    "bias": bias,
    "f1": f1_score_snowfall,
    "precision": precision_snowfall,
    "recall": recall_snowfall,
}
