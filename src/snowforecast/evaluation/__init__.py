"""Evaluation module for final model assessment.

This module provides:

- HoldoutEvaluator: Run inference on temporal holdout set
- MetricsBreakdown: Compute metrics by grouping (month, station, elevation)
- ResidualAnalyzer: Analyze prediction residuals for patterns
- ConfidenceInterval: Bootstrap confidence intervals for metrics
- PRDMetrics: Container for PRD-defined success metrics
"""

from snowforecast.evaluation.evaluation import (
    ConfidenceInterval,
    HoldoutEvaluator,
    MetricsBreakdown,
    PRDMetrics,
    ResidualAnalyzer,
    check_prd_targets,
    compute_prd_metrics,
    evaluate_holdout,
)

__all__ = [
    "HoldoutEvaluator",
    "MetricsBreakdown",
    "ResidualAnalyzer",
    "ConfidenceInterval",
    "PRDMetrics",
    "evaluate_holdout",
    "compute_prd_metrics",
    "check_prd_targets",
]
