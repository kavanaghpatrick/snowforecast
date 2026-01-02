"""Evaluation module for final model assessment.

This module provides:

- HoldoutEvaluator: Run inference on temporal holdout set
- MetricsBreakdown: Compute metrics by grouping (month, station, elevation)
- ResidualAnalyzer: Analyze prediction residuals for patterns
- ConfidenceInterval: Bootstrap confidence intervals for metrics
- PRDMetrics: Container for PRD-defined success metrics
"""

from snowforecast.evaluation.evaluation import (
    HoldoutEvaluator,
    MetricsBreakdown,
    ResidualAnalyzer,
    ConfidenceInterval,
    PRDMetrics,
    evaluate_holdout,
    compute_prd_metrics,
    check_prd_targets,
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
