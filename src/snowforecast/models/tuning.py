"""Hyperparameter tuning framework using Optuna.

This module provides a flexible hyperparameter tuning framework for ML models
in the snowforecast project.
"""

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
    from optuna.samplers import GridSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning.

    Attributes:
        n_trials: Number of trials to run.
        timeout: Maximum time in seconds for optimization (None = no limit).
        n_jobs: Number of parallel jobs (-1 for all CPUs).
        sampler: Sampling strategy ('tpe', 'random', 'grid').
        pruner: Pruning strategy ('median', 'hyperband', 'none').
        direction: Optimization direction ('minimize' or 'maximize').
        metric: Metric name to optimize.
        study_name: Optional name for the Optuna study.
        storage: Optional storage URL for study persistence.
        load_if_exists: Whether to load existing study with same name.
    """

    n_trials: int = 100
    timeout: int | None = None
    n_jobs: int = 1
    sampler: str = "tpe"
    pruner: str = "median"
    direction: str = "minimize"
    metric: str = "rmse"
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = False


# Predefined search spaces for common model types
SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "gradient_boosting": {
        "n_estimators": ("int", 100, 2000),
        "learning_rate": ("float", 0.01, 0.3, "log"),
        "max_depth": ("int", 3, 12),
        "num_leaves": ("int", 15, 127),
        "min_child_samples": ("int", 5, 100),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("float", 1e-8, 10.0, "log"),
        "reg_lambda": ("float", 1e-8, 10.0, "log"),
    },
    "lstm": {
        "hidden_size": ("categorical", [32, 64, 128, 256]),
        "num_layers": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.5),
        "learning_rate": ("float", 1e-4, 1e-2, "log"),
        "batch_size": ("categorical", [16, 32, 64, 128]),
    },
    "transformer": {
        "d_model": ("categorical", [64, 128, 256, 512]),
        "n_heads": ("categorical", [2, 4, 8]),
        "n_layers": ("int", 1, 6),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("float", 1e-5, 1e-3, "log"),
        "batch_size": ("categorical", [16, 32, 64]),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 5, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2", None]),
    },
    "linear": {
        "alpha": ("float", 1e-6, 10.0, "log"),
        "l1_ratio": ("float", 0.0, 1.0),
    },
}


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning framework.

    This class wraps Optuna to provide a simple interface for hyperparameter
    optimization of ML models.

    Attributes:
        config: TuningConfig instance with optimization settings.
        study: The Optuna study object (created during tune()).
        best_params: Best hyperparameters found (set after tune()).

    Example:
        >>> tuner = HyperparameterTuner()
        >>> best_params = tuner.tune(
        ...     model_class=GradientBoostingModel,
        ...     X_train=X_train, y_train=y_train,
        ...     X_val=X_val, y_val=y_val,
        ...     param_space=SEARCH_SPACES["gradient_boosting"],
        ... )
        >>> print(f"Best params: {best_params}")
    """

    def __init__(self, config: TuningConfig | None = None):
        """Initialize the tuner.

        Args:
            config: TuningConfig instance. Uses defaults if None.

        Raises:
            ImportError: If optuna is not installed.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )

        self.config = config or TuningConfig()
        self.study: optuna.Study | None = None
        self.best_params: dict[str, Any] | None = None
        self._objective_fn: Callable | None = None

    def _create_sampler(self) -> Any:
        """Create the appropriate Optuna sampler based on config."""
        if self.config.sampler == "tpe":
            return TPESampler()
        elif self.config.sampler == "random":
            return RandomSampler()
        elif self.config.sampler == "grid":
            # Grid sampler requires search_space, will be set during tune()
            return None
        else:
            raise ValueError(f"Unknown sampler: {self.config.sampler}")

    def _create_pruner(self) -> Any:
        """Create the appropriate Optuna pruner based on config."""
        if self.config.pruner == "median":
            return MedianPruner()
        elif self.config.pruner == "hyperband":
            return HyperbandPruner()
        elif self.config.pruner == "none":
            return NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.config.pruner}")

    def _suggest_param(
        self, trial: "optuna.Trial", name: str, spec: tuple
    ) -> Any:
        """Suggest a parameter value based on the specification.

        Args:
            trial: Optuna trial object.
            name: Parameter name.
            spec: Parameter specification tuple.
                Format depends on type:
                - ("int", low, high) for integers
                - ("float", low, high) for floats
                - ("float", low, high, "log") for log-uniform floats
                - ("categorical", [choices]) for categorical

        Returns:
            Suggested parameter value.

        Raises:
            ValueError: If parameter type is unknown.
        """
        param_type = spec[0]

        if param_type == "int":
            low, high = spec[1], spec[2]
            log = len(spec) > 3 and spec[3] == "log"
            return trial.suggest_int(name, low, high, log=log)

        elif param_type == "float":
            low, high = spec[1], spec[2]
            log = len(spec) > 3 and spec[3] == "log"
            return trial.suggest_float(name, low, high, log=log)

        elif param_type == "categorical":
            choices = spec[1]
            return trial.suggest_categorical(name, choices)

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _create_objective(
        self,
        model_class: type,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_space: dict[str, tuple],
        fit_kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        """Create the objective function for Optuna.

        Args:
            model_class: Model class to instantiate and train.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            param_space: Parameter search space specification.
            fit_kwargs: Additional kwargs to pass to model.fit().

        Returns:
            Objective function for Optuna optimization.
        """
        fit_kwargs = fit_kwargs or {}
        metric = self.config.metric

        def objective(trial: "optuna.Trial") -> float:
            # Suggest parameters based on search space
            params = {}
            for name, spec in param_space.items():
                params[name] = self._suggest_param(trial, name, spec)

            # Create and train model
            try:
                model = model_class(**params)
                model.fit(X_train, y_train, **fit_kwargs)

                # Evaluate on validation set
                predictions = model.predict(X_val)

                # Calculate metric
                if metric == "rmse":
                    import numpy as np

                    score = np.sqrt(np.mean((y_val - predictions) ** 2))
                elif metric == "mae":
                    import numpy as np

                    score = np.mean(np.abs(y_val - predictions))
                elif metric == "mse":
                    import numpy as np

                    score = np.mean((y_val - predictions) ** 2)
                elif metric == "r2":
                    from sklearn.metrics import r2_score

                    score = r2_score(y_val, predictions)
                else:
                    # Try to get metric from model if it has score method
                    if hasattr(model, "score"):
                        score = model.score(X_val, y_val)
                    else:
                        raise ValueError(f"Unknown metric: {metric}")

                return score

            except Exception as e:
                # If trial fails, report as pruned
                raise optuna.TrialPruned(f"Trial failed: {e}")

        return objective

    def tune(
        self,
        model_class: type,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        param_space: dict[str, tuple],
        fit_kwargs: dict[str, Any] | None = None,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            model_class: The model class to tune (must have fit/predict methods).
            X_train: Training features DataFrame.
            y_train: Training target Series.
            X_val: Validation features DataFrame.
            y_val: Validation target Series.
            param_space: Dict mapping param names to specifications.
                Example: {
                    "n_estimators": ("int", 100, 2000),
                    "learning_rate": ("float", 0.01, 0.3, "log"),
                    "max_depth": ("int", 3, 12),
                }
            fit_kwargs: Additional keyword arguments for model.fit().
            callbacks: Optional list of Optuna callback functions.

        Returns:
            Best hyperparameters found during optimization.

        Raises:
            ImportError: If optuna is not installed.
            ValueError: If invalid sampler/pruner specified.
        """
        # Create study
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
        )

        # Create objective function
        objective = self._create_objective(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_space=param_space,
            fit_kwargs=fit_kwargs,
        )
        self._objective_fn = objective

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            callbacks=callbacks,
            show_progress_bar=False,  # Avoid cluttering logs
        )

        # Store best params
        self.best_params = self.study.best_params

        return self.best_params

    def tune_with_custom_objective(
        self,
        objective: Callable[["optuna.Trial"], float],
        callbacks: list[Callable] | None = None,
    ) -> dict[str, Any]:
        """Run optimization with a custom objective function.

        This allows full control over the optimization process for complex
        training procedures (e.g., cross-validation, custom metrics).

        Args:
            objective: Custom objective function taking an Optuna trial
                and returning a float metric value.
            callbacks: Optional list of Optuna callback functions.

        Returns:
            Best hyperparameters found during optimization.

        Example:
            >>> def custom_objective(trial):
            ...     lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            ...     # Custom training logic here
            ...     return validation_score
            >>> tuner.tune_with_custom_objective(custom_objective)
        """
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
        )

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            callbacks=callbacks,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        return self.best_params

    def get_optimization_history(self) -> pd.DataFrame:
        """Return trial history as a DataFrame.

        Returns:
            DataFrame with columns for trial number, parameters, and values.

        Raises:
            ValueError: If tune() has not been called yet.
        """
        if self.study is None:
            raise ValueError("No study available. Call tune() first.")

        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "datetime_start": trial.datetime_start,
                    "datetime_complete": trial.datetime_complete,
                    **trial.params,
                }
                trials_data.append(row)

        if not trials_data:
            return pd.DataFrame()

        return pd.DataFrame(trials_data)

    def get_best_trial(self) -> dict[str, Any]:
        """Get details of the best trial.

        Returns:
            Dict with best trial info including params, value, and metadata.

        Raises:
            ValueError: If tune() has not been called yet.
        """
        if self.study is None:
            raise ValueError("No study available. Call tune() first.")

        best = self.study.best_trial
        return {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "datetime_start": best.datetime_start,
            "datetime_complete": best.datetime_complete,
            "duration": (
                (best.datetime_complete - best.datetime_start).total_seconds()
                if best.datetime_complete
                else None
            ),
        }

    def get_param_importances(self) -> dict[str, float]:
        """Calculate parameter importance scores.

        Uses Optuna's fANOVA-based importance evaluation.

        Returns:
            Dict mapping parameter names to importance scores (0-1).

        Raises:
            ValueError: If tune() has not been called yet.
        """
        if self.study is None:
            raise ValueError("No study available. Call tune() first.")

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception:
            # May fail if not enough trials completed
            return {}

    def plot_optimization_history(self) -> None:
        """Plot optimization progress.

        Displays a plot showing objective value over trials.
        Requires optuna.visualization to be available.

        Raises:
            ValueError: If tune() has not been called yet.
            ImportError: If plotly is not installed.
        """
        if self.study is None:
            raise ValueError("No study available. Call tune() first.")

        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self.study)
            fig.show()
        except ImportError:
            raise ImportError(
                "Plotly is required for visualization. "
                "Install with: pip install plotly"
            )

    def plot_param_importances(self) -> None:
        """Plot parameter importance visualization.

        Raises:
            ValueError: If tune() has not been called yet.
            ImportError: If plotly is not installed.
        """
        if self.study is None:
            raise ValueError("No study available. Call tune() first.")

        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self.study)
            fig.show()
        except ImportError:
            raise ImportError(
                "Plotly is required for visualization. "
                "Install with: pip install plotly"
            )

    def plot_contour(self, params: list[str] | None = None) -> None:
        """Plot contour of parameter relationships.

        Args:
            params: List of parameter names to include. Uses all if None.

        Raises:
            ValueError: If tune() has not been called yet.
            ImportError: If plotly is not installed.
        """
        if self.study is None:
            raise ValueError("No study available. Call tune() first.")

        try:
            from optuna.visualization import plot_contour

            fig = plot_contour(self.study, params=params)
            fig.show()
        except ImportError:
            raise ImportError(
                "Plotly is required for visualization. "
                "Install with: pip install plotly"
            )


def get_search_space(model_type: str) -> dict[str, tuple]:
    """Get a predefined search space for a model type.

    Args:
        model_type: One of 'gradient_boosting', 'lstm', 'transformer',
            'random_forest', 'linear'.

    Returns:
        Dict of parameter specifications.

    Raises:
        KeyError: If model_type is not recognized.
    """
    if model_type not in SEARCH_SPACES:
        available = ", ".join(SEARCH_SPACES.keys())
        raise KeyError(
            f"Unknown model type: {model_type}. "
            f"Available types: {available}"
        )
    return SEARCH_SPACES[model_type].copy()


def create_tuner(
    n_trials: int = 100,
    timeout: int | None = None,
    metric: str = "rmse",
    direction: str = "minimize",
    **kwargs,
) -> HyperparameterTuner:
    """Factory function to create a tuner with common settings.

    Args:
        n_trials: Number of optimization trials.
        timeout: Maximum time in seconds.
        metric: Metric to optimize.
        direction: 'minimize' or 'maximize'.
        **kwargs: Additional TuningConfig parameters.

    Returns:
        Configured HyperparameterTuner instance.
    """
    config = TuningConfig(
        n_trials=n_trials,
        timeout=timeout,
        metric=metric,
        direction=direction,
        **kwargs,
    )
    return HyperparameterTuner(config)
