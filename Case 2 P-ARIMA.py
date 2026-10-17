from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for the ARIMA baseline. "
        "Install it with: pip install statsmodels"
    ) from exc


# Configuration

class Config:
    data_dir = Path("path/to/your/data")

    train_file = "train.xlsx"
    val_file = "validation.xlsx"
    test_file = "test.xlsx"
    output_dir = data_dir / "results_arima"

    rolling_stride = 1
    forecast_horizon = 1

    time_column_index = 0
    target_column_index = 57
    target_column_name = "BF"

    arima_order = (2, 0, 1)
    arima_trend = "c"
    update_with_observations = True

    eps_var = 1e-6
    z_value = 1.959963984540054


cfg = Config()


# Data loading and preprocessing

def read_target_series(file_path, cfg):
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    required_columns = max(
        cfg.time_column_index,
        cfg.target_column_index,
    ) + 1

    if df.shape[1] < required_columns:
        raise ValueError(
            f"{file_path} contains {df.shape[1]} columns; "
            f"at least {required_columns} are required."
        )

    time = df.iloc[:, cfg.time_column_index].to_numpy()
    target = pd.to_numeric(
        df.iloc[:, cfg.target_column_index], errors="raise"
    ).to_numpy(dtype=np.float64).reshape(-1)

    if len(time) != len(target):
        raise ValueError(f"Inconsistent row counts in {file_path}.")
    if target.size == 0:
        raise ValueError(f"No samples found in {file_path}.")
    if not np.isfinite(target).all():
        raise ValueError(f"Non-finite target values found in {file_path}.")

    return {
        "time": time,
        "target": target,
        "sample_count": target.size,
    }


def prepare_data(cfg):
    if cfg.forecast_horizon != 1 or cfg.rolling_stride != 1:
        raise ValueError(
            "The Case II ARIMA baseline requires hourly rolling "
            "one-step forecasting."
        )

    split_files = {
        "Train": cfg.data_dir / cfg.train_file,
        "Validation": cfg.data_dir / cfg.val_file,
        "Test": cfg.data_dir / cfg.test_file,
    }

    return {
        name: read_target_series(path, cfg)
        for name, path in split_files.items()
    }


# Probabilistic ARIMA baseline

class ProbabilisticARIMA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.result = None
        self.aic = None
        self.bic = None

    def fit(self, y_train):
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        if y_train.size == 0:
            raise ValueError("The training series is empty.")

        model = ARIMA(
            y_train,
            order=self.cfg.arima_order,
            trend=self.cfg.arima_trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = model.fit()
        self.aic = float(self.result.aic)
        self.bic = float(self.result.bic)
        return self

    def _extract_forecast(self, forecast_result):
        mean = float(
            np.asarray(forecast_result.predicted_mean).reshape(-1)[0]
        )

        try:
            variance = float(
                np.asarray(forecast_result.var_pred_mean).reshape(-1)[0]
            )
        except AttributeError:
            frame = forecast_result.summary_frame()
            variance = float(
                np.asarray(frame["mean_se"]).reshape(-1)[0] ** 2
            )

        variance = max(variance, self.cfg.eps_var)
        return mean, variance

    def predict_in_sample_train(self, y_train):
        if self.result is None:
            raise RuntimeError(
                "The ARIMA model must be fitted before prediction."
            )

        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        prediction_result = self.result.get_prediction(
            start=0,
            end=len(y_train) - 1,
            dynamic=False,
        )

        mean = np.asarray(
            prediction_result.predicted_mean, dtype=np.float64
        ).reshape(-1)
        try:
            variance = np.asarray(
                prediction_result.var_pred_mean, dtype=np.float64
            ).reshape(-1)
        except AttributeError:
            frame = prediction_result.summary_frame()
            variance = np.asarray(frame["mean_se"], dtype=np.float64) ** 2

        variance = np.maximum(variance, self.cfg.eps_var)
        return y_train.copy(), mean, variance

    def rolling_one_step_forecast(
        self,
        eval_series,
        pre_eval_observations=None,
    ):
        if self.result is None:
            raise RuntimeError(
                "The ARIMA model must be fitted before prediction."
            )

        eval_series = np.asarray(
            eval_series, dtype=np.float64
        ).reshape(-1)
        current_result = self.result

        if pre_eval_observations is not None:
            previous_values = np.asarray(
                pre_eval_observations, dtype=np.float64
            ).reshape(-1)

            if previous_values.size > 0:
                current_result = current_result.append(
                    previous_values,
                    refit=False,
                )

        means = []
        variances = []

        for observed_value in eval_series:
            forecast_result = current_result.get_forecast(steps=1)
            mean, variance = self._extract_forecast(forecast_result)

            means.append(mean)
            variances.append(variance)

            if self.cfg.update_with_observations:
                current_result = current_result.append(
                    [float(observed_value)],
                    refit=False,
                )

        return (
            eval_series.copy(),
            np.asarray(means, dtype=np.float64),
            np.asarray(variances, dtype=np.float64),
        )


# Output

def save_predictions(output_dir, split_name, time, observed, mean, variance, cfg):
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    variance = np.maximum(
        np.asarray(variance, dtype=np.float64).reshape(-1),
        cfg.eps_var,
    )
    time = np.asarray(time).reshape(-1)

    if not (
        len(time) == len(observed) == len(mean) == len(variance)
    ):
        raise ValueError(
            f"Inconsistent prediction lengths for the {split_name} split."
        )

    std = np.sqrt(variance)
    lower = mean - cfg.z_value * std
    upper = mean + cfg.z_value * std

    prediction_df = pd.DataFrame({
        "Time": time,
        "Observed": observed,
        "Predicted_mean": mean,
        "Predicted_variance": variance,
        "Predicted_std": std,
        "Lower_95PI": lower,
        "Upper_95PI": upper,
        "Residual": observed - mean,
    })
    prediction_path = output_dir / f"{split_name}_prediction_results.xlsx"
    prediction_df.to_excel(prediction_path, index=False)

    print(
        f"{split_name}: saved {len(prediction_df)} sample-level predictions "
        f"to {prediction_path}"
    )
    return variance


def train_model(cfg):
    split_data = prepare_data(cfg)
    train_y = split_data["Train"]["target"]
    val_y = split_data["Validation"]["target"]
    test_y = split_data["Test"]["target"]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model = ProbabilisticARIMA(cfg).fit(train_y)

    model_info_df = pd.DataFrame([{
        "Model": "ARIMA",
        "Order": str(cfg.arima_order),
        "Trend": cfg.arima_trend,
        "AIC": model.aic,
        "BIC": model.bic,
        "Rolling_stride_h": cfg.rolling_stride,
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Update_with_observations": cfg.update_with_observations,
        "Parameter_fit_data": "Training set only",
        "Refit_after_training": False,
        "Predictive_variance": "Native ARIMA one-step predictive variance",
    }])
    model_info_df.to_excel(
        cfg.output_dir / "arima_model_info.xlsx",
        index=False,
    )

    fit_summary_df = pd.DataFrame([{
        "Model": "ARIMA",
        "Training_samples": len(train_y),
        "Order": str(cfg.arima_order),
        "Trend": cfg.arima_trend,
        "Parameter_fit_data": "Training set only",
        "Forecast_protocol": "Hourly rolling one-step ahead",
        "State_update": "Append each realized observation with refit=False",
    }])
    fit_summary_df.to_excel(
        cfg.output_dir / "model_fit_summary.xlsx",
        index=False,
    )

    train_observed, train_mean, train_variance = (
        model.predict_in_sample_train(train_y)
    )
    _, val_mean, val_variance = model.rolling_one_step_forecast(val_y)
    _, test_mean, test_variance = model.rolling_one_step_forecast(
        test_y,
        pre_eval_observations=val_y,
    )

    split_outputs = [
        (
            "Train",
            train_observed,
            train_mean,
            train_variance,
            split_data["Train"]["time"],
        ),
        (
            "Validation",
            val_y,
            val_mean,
            val_variance,
            split_data["Validation"]["time"],
        ),
        (
            "Test",
            test_y,
            test_mean,
            test_variance,
            split_data["Test"]["time"],
        ),
    ]

    variance_diagnostics = []
    for split_name, observed, mean, variance, time in split_outputs:
        variance = save_predictions(
            cfg.output_dir,
            split_name,
            time,
            observed,
            mean,
            variance,
            cfg,
        )

        if split_name in ("Validation", "Test"):
            variance_diagnostics.append({
                "Split": split_name,
                "Variance_mean": float(np.mean(variance)),
                "Variance_std": float(np.std(variance)),
                "Variance_min": float(np.min(variance)),
                "Variance_max": float(np.max(variance)),
                "Forecast_horizon_h": cfg.forecast_horizon,
            })

    pd.DataFrame(variance_diagnostics).to_excel(
        cfg.output_dir / "arima_one_step_variance_diagnostic.xlsx",
        index=False,
    )

    protocol_df = pd.DataFrame([{
        "Model": "Probabilistic ARIMA",
        "Rolling_stride_h": cfg.rolling_stride,
        "Forecast_horizon_h": cfg.forecast_horizon,
        "Forecast_call": "get_forecast(steps=1)",
        "State_update": "Append observation after its forecast with refit=False",
        "Parameter_fit_data": "Training set only",
        "Validation_parameter_refit": False,
        "Test_parameter_refit": False,
        "Test_initial_history": (
            "Training-fitted state plus historically available Validation observations"
        ),
        "Predictive_variance": "Native ARIMA one-step predictive variance",
        "Input_series": f"Target column {cfg.target_column_name}",
    }])
    protocol_df.to_excel(
        cfg.output_dir / "forecast_protocol.xlsx",
        index=False,
    )

    input_info_df = pd.DataFrame({
        "Item": [
            "Model",
            "Target series",
            "Target source column",
            "ECMWF predictors used",
            "Historical ANE predictors used as exogenous inputs",
            "Rolling stride (h)",
            "Forecast horizon (h)",
            "Probabilistic output",
            "Parameter estimation",
            "State update",
        ],
        "Value": [
            "ARIMA",
            "Hourly ANE target series",
            cfg.target_column_name,
            "No",
            "No; ARIMA is fitted to the target series only",
            cfg.rolling_stride,
            cfg.forecast_horizon,
            "Predictive mean and native predictive variance",
            "Training series only",
            "Append each realized observation after forecasting, without refitting",
        ],
    })
    input_info_df.to_excel(
        cfg.output_dir / "model_input_information.xlsx",
        index=False,
    )

    print(f"All outputs saved to: {cfg.output_dir}")
    return model


if __name__ == "__main__":
    train_model(cfg)
