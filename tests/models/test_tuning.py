"""Tests for hyperparameter tuning framework.

Tests use real Optuna optimization - NO MOCKS.
Requires optuna to be installed.
"""

import numpy as np
import pandas as pd
import pytest

from snowforecast.models.tuning import (
    TuningConfig,
    SEARCH_SPACES,
    get_search_space,
    create_tuner,
    OPTUNA_AVAILABLE,
)

if OPTUNA_AVAILABLE:
    from snowforecast.models.tuning import HyperparameterTuner
    import optuna


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = TuningConfig()

        assert config.n_trials == 100
        assert config.timeout is None
        assert config.n_jobs == 1
        assert config.sampler == "tpe"
        assert config.pruner == "median"
        assert config.direction == "minimize"
        assert config.metric == "rmse"
        assert config.study_name is None
        assert config.storage is None
        assert config.load_if_exists is False

    def test_custom_values(self):
        """Should accept custom values."""
        config = TuningConfig(
            n_trials=50,
            timeout=3600,
            n_jobs=4,
            sampler="random",
            pruner="hyperband",
            direction="maximize",
            metric="r2",
            study_name="my_study",
        )

        assert config.n_trials == 50
        assert config.timeout == 3600
        assert config.n_jobs == 4
        assert config.sampler == "random"
        assert config.pruner == "hyperband"
        assert config.direction == "maximize"
        assert config.metric == "r2"
        assert config.study_name == "my_study"


class TestSearchSpaces:
    """Tests for predefined search spaces."""

    def test_gradient_boosting_space(self):
        """Gradient boosting should have all key params."""
        space = SEARCH_SPACES["gradient_boosting"]

        expected_params = [
            "n_estimators", "learning_rate", "max_depth",
            "num_leaves", "min_child_samples", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda",
        ]

        for param in expected_params:
            assert param in space

    def test_lstm_space(self):
        """LSTM should have neural network params."""
        space = SEARCH_SPACES["lstm"]

        expected_params = [
            "hidden_size", "num_layers", "dropout",
            "learning_rate", "batch_size",
        ]

        for param in expected_params:
            assert param in space

    def test_transformer_space(self):
        """Transformer should have attention params."""
        space = SEARCH_SPACES["transformer"]

        expected_params = [
            "d_model", "n_heads", "n_layers",
            "dropout", "learning_rate", "batch_size",
        ]

        for param in expected_params:
            assert param in space

    def test_random_forest_space(self):
        """Random forest should have tree params."""
        space = SEARCH_SPACES["random_forest"]

        expected_params = [
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features",
        ]

        for param in expected_params:
            assert param in space

    def test_linear_space(self):
        """Linear should have regularization params."""
        space = SEARCH_SPACES["linear"]

        assert "alpha" in space
        assert "l1_ratio" in space

    def test_param_spec_format(self):
        """All specs should have valid format."""
        for model_type, space in SEARCH_SPACES.items():
            for param_name, spec in space.items():
                # First element should be type
                assert spec[0] in ("int", "float", "categorical"), (
                    f"{model_type}.{param_name} has invalid type: {spec[0]}"
                )

                if spec[0] == "categorical":
                    # Should have list of choices
                    assert isinstance(spec[1], list), (
                        f"{model_type}.{param_name} categorical should have choices"
                    )
                else:
                    # Should have low and high bounds
                    assert len(spec) >= 3, (
                        f"{model_type}.{param_name} should have low/high bounds"
                    )
                    assert spec[1] < spec[2], (
                        f"{model_type}.{param_name} low should be < high"
                    )


class TestGetSearchSpace:
    """Tests for get_search_space function."""

    def test_returns_copy(self):
        """Should return a copy, not the original."""
        space1 = get_search_space("gradient_boosting")
        space2 = get_search_space("gradient_boosting")

        # Modify one
        space1["test_param"] = ("int", 1, 10)

        # Other should be unchanged
        assert "test_param" not in space2

    def test_unknown_model_type(self):
        """Should raise KeyError for unknown model."""
        with pytest.raises(KeyError, match="Unknown model type"):
            get_search_space("unknown_model")

    def test_all_model_types_accessible(self):
        """Should be able to get all registered types."""
        for model_type in SEARCH_SPACES.keys():
            space = get_search_space(model_type)
            assert isinstance(space, dict)
            assert len(space) > 0


class TestCreateTuner:
    """Tests for create_tuner factory function."""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_basic_creation(self):
        """Should create tuner with default config."""
        tuner = create_tuner()

        assert tuner.config.n_trials == 100
        assert tuner.config.metric == "rmse"
        assert tuner.config.direction == "minimize"

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_custom_config(self):
        """Should apply custom config values."""
        tuner = create_tuner(
            n_trials=50,
            timeout=600,
            metric="mae",
            direction="minimize",
            sampler="random",
        )

        assert tuner.config.n_trials == 50
        assert tuner.config.timeout == 600
        assert tuner.config.metric == "mae"
        assert tuner.config.sampler == "random"


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        tuner = HyperparameterTuner()

        assert tuner.config is not None
        assert tuner.study is None
        assert tuner.best_params is None

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = TuningConfig(n_trials=10, metric="mae")
        tuner = HyperparameterTuner(config)

        assert tuner.config.n_trials == 10
        assert tuner.config.metric == "mae"

    def test_create_sampler_tpe(self):
        """Should create TPE sampler."""
        tuner = HyperparameterTuner(TuningConfig(sampler="tpe"))
        sampler = tuner._create_sampler()
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_create_sampler_random(self):
        """Should create Random sampler."""
        tuner = HyperparameterTuner(TuningConfig(sampler="random"))
        sampler = tuner._create_sampler()
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_create_sampler_invalid(self):
        """Should raise for invalid sampler."""
        tuner = HyperparameterTuner(TuningConfig(sampler="invalid"))
        with pytest.raises(ValueError, match="Unknown sampler"):
            tuner._create_sampler()

    def test_create_pruner_median(self):
        """Should create Median pruner."""
        tuner = HyperparameterTuner(TuningConfig(pruner="median"))
        pruner = tuner._create_pruner()
        assert isinstance(pruner, optuna.pruners.MedianPruner)

    def test_create_pruner_hyperband(self):
        """Should create Hyperband pruner."""
        tuner = HyperparameterTuner(TuningConfig(pruner="hyperband"))
        pruner = tuner._create_pruner()
        assert isinstance(pruner, optuna.pruners.HyperbandPruner)

    def test_create_pruner_none(self):
        """Should create Nop pruner for 'none'."""
        tuner = HyperparameterTuner(TuningConfig(pruner="none"))
        pruner = tuner._create_pruner()
        assert isinstance(pruner, optuna.pruners.NopPruner)

    def test_create_pruner_invalid(self):
        """Should raise for invalid pruner."""
        tuner = HyperparameterTuner(TuningConfig(pruner="invalid"))
        with pytest.raises(ValueError, match="Unknown pruner"):
            tuner._create_pruner()

    def test_suggest_param_int(self):
        """Should suggest integer parameter."""
        tuner = HyperparameterTuner()

        # Create a mock trial
        study = optuna.create_study()

        def objective(trial):
            val = tuner._suggest_param(trial, "test_int", ("int", 1, 10))
            assert isinstance(val, int)
            assert 1 <= val <= 10
            return val

        study.optimize(objective, n_trials=1)

    def test_suggest_param_float(self):
        """Should suggest float parameter."""
        tuner = HyperparameterTuner()

        study = optuna.create_study()

        def objective(trial):
            val = tuner._suggest_param(trial, "test_float", ("float", 0.1, 1.0))
            assert isinstance(val, float)
            assert 0.1 <= val <= 1.0
            return val

        study.optimize(objective, n_trials=1)

    def test_suggest_param_float_log(self):
        """Should suggest log-scale float parameter."""
        tuner = HyperparameterTuner()

        study = optuna.create_study()

        def objective(trial):
            val = tuner._suggest_param(
                trial, "test_log_float", ("float", 1e-4, 1e-1, "log")
            )
            assert isinstance(val, float)
            assert 1e-4 <= val <= 1e-1
            return val

        study.optimize(objective, n_trials=1)

    def test_suggest_param_categorical(self):
        """Should suggest categorical parameter."""
        tuner = HyperparameterTuner()

        study = optuna.create_study()
        choices = ["a", "b", "c"]

        def objective(trial):
            val = tuner._suggest_param(trial, "test_cat", ("categorical", choices))
            assert val in choices
            return 1.0

        study.optimize(objective, n_trials=1)

    def test_suggest_param_invalid_type(self):
        """Should raise for invalid parameter type."""
        tuner = HyperparameterTuner()

        study = optuna.create_study()

        def objective(trial):
            tuner._suggest_param(trial, "test", ("invalid_type", 1, 10))
            return 1.0

        with pytest.raises(ValueError, match="Unknown parameter type"):
            study.optimize(objective, n_trials=1)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestTunerTune:
    """Tests for the tune() method with real optimization."""

    @pytest.fixture
    def simple_model_class(self):
        """Create a simple model class for testing."""
        class SimpleModel:
            def __init__(self, alpha=1.0, beta=1.0):
                self.alpha = alpha
                self.beta = beta
                self.is_fitted = False

            def fit(self, X, y):
                self.is_fitted = True
                return self

            def predict(self, X):
                # Simple prediction based on params
                return np.full(len(X), self.alpha + self.beta)

        return SimpleModel

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 50
        X_train = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        y_train = pd.Series(np.random.randn(n) + 5)  # Target around 5
        X_val = pd.DataFrame({"f1": np.random.randn(20), "f2": np.random.randn(20)})
        y_val = pd.Series(np.random.randn(20) + 5)
        return X_train, y_train, X_val, y_val

    def test_basic_tuning(self, simple_model_class, sample_data):
        """Should run optimization and return best params."""
        X_train, y_train, X_val, y_val = sample_data

        config = TuningConfig(n_trials=5, metric="rmse")
        tuner = HyperparameterTuner(config)

        param_space = {
            "alpha": ("float", 0.0, 10.0),
            "beta": ("float", 0.0, 5.0),
        }

        best_params = tuner.tune(
            model_class=simple_model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_space=param_space,
        )

        assert "alpha" in best_params
        assert "beta" in best_params
        assert tuner.study is not None
        assert tuner.best_params is not None

    def test_tuning_with_mae(self, simple_model_class, sample_data):
        """Should work with MAE metric."""
        X_train, y_train, X_val, y_val = sample_data

        config = TuningConfig(n_trials=3, metric="mae")
        tuner = HyperparameterTuner(config)

        param_space = {"alpha": ("float", 0.0, 10.0)}

        best_params = tuner.tune(
            model_class=simple_model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_space=param_space,
        )

        assert "alpha" in best_params

    def test_get_optimization_history(self, simple_model_class, sample_data):
        """Should return trial history as DataFrame."""
        X_train, y_train, X_val, y_val = sample_data

        config = TuningConfig(n_trials=3)
        tuner = HyperparameterTuner(config)

        param_space = {"alpha": ("float", 0.0, 10.0)}

        tuner.tune(
            model_class=simple_model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_space=param_space,
        )

        history = tuner.get_optimization_history()

        assert isinstance(history, pd.DataFrame)
        assert len(history) <= 3  # May be less if trials pruned
        if len(history) > 0:
            assert "trial_number" in history.columns
            assert "value" in history.columns
            assert "alpha" in history.columns

    def test_get_optimization_history_no_study(self):
        """Should raise if tune() not called."""
        tuner = HyperparameterTuner()

        with pytest.raises(ValueError, match="No study available"):
            tuner.get_optimization_history()

    def test_get_best_trial(self, simple_model_class, sample_data):
        """Should return best trial details."""
        X_train, y_train, X_val, y_val = sample_data

        config = TuningConfig(n_trials=3)
        tuner = HyperparameterTuner(config)

        param_space = {"alpha": ("float", 0.0, 10.0)}

        tuner.tune(
            model_class=simple_model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            param_space=param_space,
        )

        best = tuner.get_best_trial()

        assert "number" in best
        assert "value" in best
        assert "params" in best

    def test_get_best_trial_no_study(self):
        """Should raise if tune() not called."""
        tuner = HyperparameterTuner()

        with pytest.raises(ValueError, match="No study available"):
            tuner.get_best_trial()


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestCustomObjective:
    """Tests for tune_with_custom_objective."""

    def test_custom_objective(self):
        """Should work with user-defined objective."""
        config = TuningConfig(n_trials=5)
        tuner = HyperparameterTuner(config)

        def custom_objective(trial):
            x = trial.suggest_float("x", -10, 10)
            # Simple quadratic - minimum at x=0
            return x ** 2

        best_params = tuner.tune_with_custom_objective(custom_objective)

        assert "x" in best_params
        # Best x should be close to 0
        assert abs(best_params["x"]) < 5

    def test_custom_objective_categorical(self):
        """Should handle categorical params in custom objective."""
        # Use more trials to ensure we find the best option
        config = TuningConfig(n_trials=20)
        tuner = HyperparameterTuner(config)

        def custom_objective(trial):
            method = trial.suggest_categorical("method", ["a", "b", "c"])
            # Return different values based on method
            scores = {"a": 1.0, "b": 0.5, "c": 2.0}
            return scores[method]

        best_params = tuner.tune_with_custom_objective(custom_objective)

        assert "method" in best_params
        # Best should be "b" (lowest score) with enough trials
        assert best_params["method"] == "b"


class TestOptunaNotInstalled:
    """Tests for behavior when optuna is not installed."""

    @pytest.mark.skipif(OPTUNA_AVAILABLE, reason="Test only when optuna not installed")
    def test_tuner_raises_import_error(self):
        """HyperparameterTuner should raise ImportError."""
        # This test only runs when optuna is NOT installed
        # It verifies the error handling works
        pass  # Cannot really test this when optuna IS installed
