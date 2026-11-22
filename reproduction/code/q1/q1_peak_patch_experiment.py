from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import lightgbm as lgb
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "q1_peak_patch_20260810_mplconfig"),
)
import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    ROOT
    / "05_结果与图表"
    / "intermediate"
    / "question1"
    / "experiments"
    / "peak_patch_20260810"
)
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REVIEW_DIR = OUTPUT_DIR / "reviews"
DATA_PATH = ROOT / "02_数据处理" / "processed" / "q1_hourly_gpu_demand.csv"
TARGET = "GPU_Demand_Arrival_Sum"
SEED = 20260808
ALPHA = 0.95
NUM_BOOST_ROUND = 97
SELECTION_WINDOWS = [(2304, 2327), (2328, 2351), (2352, 2375)]
POST_HOC_WINDOW = (2376, 2399)
FEATURE_COLUMNS = [
    "horizon",
    "hour_of_day",
    "day_of_week",
    "day_index",
    "lag24",
    "lag48",
    "lag168",
    "origin_last",
    "origin_mean24",
    "origin_std24",
    "origin_max24",
    "origin_zero_ratio24",
    "origin_mean168",
    "origin_std168",
    "origin_max168",
]
NATIVE_BASE_PARAMS = {
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "learning_rate": 0.05,
    # sklearn's subsample=0.9 is inert with the default subsample_freq=0.
    "bagging_fraction": 0.9,
    "bagging_freq": 0,
    "feature_fraction": 0.9,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "num_threads": min(8, os.cpu_count() or 1),
    "metric": "None",
}
CANDIDATE_CONFIG = {
    "change_level": "FROZEN",
    "m0": {"objective": "regression_l1", "sample_weight": "all_1"},
    "m1": {
        "objective": "regression_l1",
        "sample_weight": "2_if_training_target_gte_per_series_training_P90_else_1",
    },
    "u1": {
        "quantile_objective": "quantile",
        "alpha": ALPHA,
        "quantile_sample_weight": "all_1",
        "upper95": "max(M1,Q95_LightGBM,per_series_training_history_Q95)",
    },
    "num_boost_round": NUM_BOOST_ROUND,
    "native_base_params": NATIVE_BASE_PARAMS,
    "features": FEATURE_COLUMNS,
    "selection_windows": SELECTION_WINDOWS,
    "post_hoc_window": POST_HOC_WINDOW,
    "excluded": ["seasonal_naive_mixture", "task_type_specific_models", "large_search"],
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def config_sha256() -> str:
    encoded = json.dumps(CANDIDATE_CONFIG, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA_PATH,
        usecols=["Hour", "SourceRegion", "TaskType", TARGET],
    )
    if len(frame) != 2400 * 18:
        raise ValueError(f"Q1 输入应为 43200 行，实际为 {len(frame)} 行")
    if frame[TARGET].isna().any() or frame.duplicated(
        ["Hour", "SourceRegion", "TaskType"]
    ).any():
        raise ValueError("Q1 输入存在缺失目标或重复主键")
    counts = frame.groupby(["SourceRegion", "TaskType"])["Hour"].nunique()
    if len(counts) != 18 or not (counts == 2400).all():
        raise ValueError("Q1 输入不是 18 条完整的 2400 小时序列")
    return frame.sort_values(["SourceRegion", "TaskType", "Hour"]).reset_index(drop=True)


def build_feature_rows(
    history: pd.DataFrame,
    first_target: int,
    last_target: int,
) -> pd.DataFrame:
    history_end = int(history.Hour.max())
    rows: list[dict[str, Any]] = []
    for (region, task_type), group in history.groupby(
        ["SourceRegion", "TaskType"], sort=True
    ):
        series = group.set_index("Hour")[TARGET].reindex(range(history_end + 1))
        if series.isna().any():
            raise ValueError(f"{region}|{task_type} 历史序列不完整")
        values = series.to_numpy(dtype=float)
        for target_hour in range(first_target, last_target + 1):
            block_start = (target_hour // 24) * 24
            origin = block_start - 1
            if origin < 167 or origin > history_end:
                raise ValueError(
                    f"目标小时 {target_hour} 的原点 {origin} 超出历史 0–{history_end}"
                )
            history24 = values[origin - 23 : origin + 1]
            history168 = values[origin - 167 : origin + 1]
            rows.append(
                {
                    "SourceRegion": str(region),
                    "TaskType": str(task_type),
                    "target_hour": target_hour,
                    "origin_hour": origin,
                    "horizon": target_hour - origin,
                    "hour_of_day": target_hour % 24,
                    "day_of_week": (target_hour // 24) % 7,
                    "day_index": target_hour // 24,
                    "lag24": values[target_hour - 24],
                    "lag48": values[target_hour - 48],
                    "lag168": values[target_hour - 168],
                    "origin_last": values[origin],
                    "origin_mean24": float(history24.mean()),
                    "origin_std24": float(history24.std()),
                    "origin_max24": float(history24.max()),
                    "origin_zero_ratio24": float(np.mean(history24 == 0)),
                    "origin_mean168": float(history168.mean()),
                    "origin_std168": float(history168.std()),
                    "origin_max168": float(history168.max()),
                }
            )
    return pd.DataFrame(rows)


def encode_features(
    train_rows: pd.DataFrame,
    forecast_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_rows, forecast_rows], ignore_index=True)
    encoded = pd.get_dummies(
        combined[["SourceRegion", "TaskType", *FEATURE_COLUMNS]],
        columns=["SourceRegion", "TaskType"],
        dtype=float,
    )
    return (
        encoded.iloc[: len(train_rows)].reset_index(drop=True),
        encoded.iloc[len(train_rows) :].reset_index(drop=True),
    )


def series_maps(history: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    grouped = history.groupby(["SourceRegion", "TaskType"])[TARGET]
    return grouped.mean(), grouped.quantile(0.90), grouped.quantile(0.95)


def values_from_map(rows: pd.DataFrame, mapping: pd.Series) -> np.ndarray:
    return np.asarray(
        [mapping.loc[(row.SourceRegion, row.TaskType)] for row in rows.itertuples(index=False)],
        dtype=float,
    )


def train_window_models(
    full_data: pd.DataFrame,
    start: int,
    end: int,
) -> dict[str, Any]:
    if end != start + 23:
        raise ValueError("仅支持固定 24 小时窗口")
    train_end = start - 1
    history = full_data.loc[full_data.Hour <= train_end].copy()
    if int(history.Hour.max()) != train_end:
        raise AssertionError("训练历史未严格截止到窗口起点前一小时")
    train_rows = build_feature_rows(history, 168, train_end)
    forecast_rows = build_feature_rows(history, start, end)
    target_lookup = history.set_index(["SourceRegion", "TaskType", "Hour"])[TARGET]
    y_train = np.asarray(
        [
            target_lookup.loc[(row.SourceRegion, row.TaskType, row.target_hour)]
            for row in train_rows.itertuples(index=False)
        ],
        dtype=float,
    )
    x_train, x_forecast = encode_features(train_rows, forecast_rows)
    history_mean, history_p90, history_q95 = series_maps(history)
    p90_for_train = values_from_map(train_rows, history_p90)
    m1_weights = np.where(y_train >= p90_for_train, 2.0, 1.0)

    def fit_predict(
        objective: str,
        weights: np.ndarray | None,
    ) -> tuple[np.ndarray, lgb.Booster]:
        params = {**NATIVE_BASE_PARAMS, "objective": objective}
        if objective == "quantile":
            params["alpha"] = ALPHA
        dataset = lgb.Dataset(
            x_train,
            label=y_train,
            weight=weights,
            feature_name=list(x_train.columns),
            free_raw_data=False,
        )
        model = lgb.train(params, dataset, num_boost_round=NUM_BOOST_ROUND)
        prediction = np.clip(
            model.predict(x_forecast, num_iteration=NUM_BOOST_ROUND), 0.0, None
        )
        return prediction, model

    started = time.perf_counter()
    m0, _ = fit_predict("regression_l1", None)
    m1, m1_model = fit_predict("regression_l1", m1_weights)
    q95_model, _ = fit_predict("quantile", None)
    elapsed = time.perf_counter() - started
    historical_mean = values_from_map(forecast_rows, history_mean)
    historical_q95 = values_from_map(forecast_rows, history_q95)
    p90_for_forecast = values_from_map(forecast_rows, history_p90)
    upper95 = np.maximum.reduce([m1, q95_model, historical_q95])
    return {
        "train_end": train_end,
        "forecast_rows": forecast_rows,
        "m0": m0,
        "m1": m1,
        "q95_model": q95_model,
        "upper95": upper95,
        "historical_mean": historical_mean,
        "historical_q95": historical_q95,
        "series_p90": p90_for_forecast,
        "train_rows": len(train_rows),
        "weighted_train_rows": int(np.sum(m1_weights == 2.0)),
        "weight_2_fraction": float(np.mean(m1_weights == 2.0)),
        "feature_max_origin_hour": int(forecast_rows.origin_hour.max()),
        "runtime_seconds": elapsed,
        "m1_feature_importance": pd.DataFrame(
            {
                "feature": m1_model.feature_name(),
                "importance": m1_model.feature_importance(
                    importance_type="split", iteration=NUM_BOOST_ROUND
                ).astype(int),
            }
        ).sort_values(["importance", "feature"], ascending=[False, True]),
    }


def actual_window(
    full_data: pd.DataFrame,
    start: int,
    end: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    frame = full_data.loc[full_data.Hour.between(start, end)].copy()
    frame = frame.sort_values(["SourceRegion", "TaskType", "Hour"]).reset_index(drop=True)
    meta = frame[["SourceRegion", "TaskType", "Hour"]].rename(
        columns={"Hour": "target_hour"}
    )
    return frame[TARGET].to_numpy(dtype=float), meta


def wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    return float(np.abs(actual - prediction).sum() / denominator)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def median_series_wape(
    actual: np.ndarray,
    prediction: np.ndarray,
    meta: pd.DataFrame,
) -> float:
    frame = meta.copy()
    frame["actual"] = actual
    frame["prediction"] = prediction
    values = [
        wape(group.actual.to_numpy(), group.prediction.to_numpy())
        for _, group in frame.groupby(["SourceRegion", "TaskType"], sort=True)
        if float(np.abs(group.actual).sum()) > 0
    ]
    return float(np.median(values))


def high_load_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    high_mask: np.ndarray,
) -> dict[str, Any]:
    residual = actual[high_mask] - prediction[high_mask]
    if len(residual) == 0:
        return {
            "high_load_count": 0,
            "high_load_underestimation_rate": math.nan,
            "high_load_mean_underestimation": math.nan,
        }
    return {
        "high_load_count": int(len(residual)),
        "high_load_underestimation_rate": float(np.mean(residual > 0)),
        "high_load_mean_underestimation": float(np.maximum(residual, 0).mean()),
    }


def aggregate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, float]:
    frame = meta[["target_hour"]].copy()
    frame["actual"] = actual
    frame["prediction"] = prediction
    aggregate = frame.groupby("target_hour", sort=True)[["actual", "prediction"]].sum()
    if aggregate.actual.std() == 0 or aggregate.prediction.std() == 0:
        correlation = math.nan
    else:
        correlation = float(aggregate.actual.corr(aggregate.prediction))
    actual_peak = float(aggregate.actual.max())
    peak_ratio = float(aggregate.prediction.max() / actual_peak) if actual_peak > 0 else math.nan
    return {
        "aggregate_curve_correlation": correlation,
        "aggregate_actual_peak": actual_peak,
        "aggregate_predicted_peak": float(aggregate.prediction.max()),
        "aggregate_peak_ratio": peak_ratio,
        "aggregate_peak_ratio_abs_error": abs(peak_ratio - 1.0),
    }


def pinball_loss(actual: np.ndarray, prediction: np.ndarray) -> float:
    error = actual - prediction
    return float(np.mean(np.maximum(ALPHA * error, (ALPHA - 1.0) * error)))


def upper_metrics(
    actual: np.ndarray,
    upper: np.ndarray,
    point: np.ndarray,
    high_mask: np.ndarray,
) -> dict[str, Any]:
    residual = actual - upper
    high_residual = residual[high_mask]
    return {
        "pinball_loss_alpha_0_95": pinball_loss(actual, upper),
        "overall_coverage": float(np.mean(actual <= upper)),
        "high_load_coverage": float(np.mean(actual[high_mask] <= upper[high_mask]))
        if high_mask.any()
        else math.nan,
        "high_load_underestimation_rate": float(np.mean(high_residual > 0))
        if high_mask.any()
        else math.nan,
        "high_load_mean_underestimation": float(np.maximum(high_residual, 0).mean())
        if high_mask.any()
        else math.nan,
        "upper95_ge_m1_all": bool(np.all(upper >= point - 1e-12)),
        "upper95_below_m1_count": int(np.sum(upper < point - 1e-12)),
        "minimum_upper95_minus_m1": float(np.min(upper - point)),
        "unique_predictions_rounded_6dp": int(np.unique(np.round(upper, 6)).size),
        "nonfinite_count": int(np.sum(~np.isfinite(upper))),
        "mean_upper95": float(np.mean(upper)),
        "mean_width_above_m1": float(np.mean(upper - point)),
    }


def all_zero_series_count(prediction: np.ndarray, meta: pd.DataFrame) -> int:
    frame = meta[["SourceRegion", "TaskType"]].copy()
    frame["prediction"] = prediction
    return int(
        sum(
            bool(np.all(group.prediction.to_numpy(dtype=float) <= 1e-12))
            for _, group in frame.groupby(["SourceRegion", "TaskType"], sort=True)
        )
    )


def evaluate_one_window(
    full_data: pd.DataFrame,
    start: int,
    end: int,
    role: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fitted = train_window_models(full_data, start, end)
    actual, meta = actual_window(full_data, start, end)
    aligned = fitted["forecast_rows"][["SourceRegion", "TaskType", "target_hour"]]
    if not np.array_equal(meta.to_numpy(), aligned.to_numpy()):
        raise AssertionError("预测与真实值行未对齐")
    high_mask = actual >= fitted["series_p90"]
    m0_high = high_load_metrics(actual, fitted["m0"], high_mask)
    m1_high = high_load_metrics(actual, fitted["m1"], high_mask)
    m0_agg = aggregate_metrics(actual, fitted["m0"], meta)
    m1_agg = aggregate_metrics(actual, fitted["m1"], meta)
    upper = upper_metrics(actual, fitted["upper95"], fitted["m1"], high_mask)
    hist_upper = upper_metrics(
        actual,
        fitted["historical_q95"],
        fitted["m1"],
        high_mask,
    )
    upper["all_zero_series"] = all_zero_series_count(fitted["upper95"], meta)
    hist_upper["all_zero_series"] = all_zero_series_count(fitted["historical_q95"], meta)
    upper_cost = {
        "pinball_loss_delta_vs_historical_q95": upper["pinball_loss_alpha_0_95"]
        - hist_upper["pinball_loss_alpha_0_95"],
        "coverage_delta_vs_historical_q95": upper["overall_coverage"]
        - hist_upper["overall_coverage"],
        "high_load_coverage_delta_vs_historical_q95": upper["high_load_coverage"]
        - hist_upper["high_load_coverage"],
        "mean_upper_delta_vs_historical_q95": float(
            np.mean(fitted["upper95"] - fitted["historical_q95"])
        ),
        "total_upper_delta_vs_historical_q95": float(
            np.sum(fitted["upper95"] - fitted["historical_q95"])
        ),
    }
    summary = {
        "window": f"{start}-{end}",
        "role": role,
        "used_for_selection": role == "selection",
        "evidence_label": "fixed_window_validation"
        if role == "selection"
        else "post-hoc extension; not a new unseen-test proof",
        "train_end": fitted["train_end"],
        "threshold_history_end": fitted["train_end"],
        "feature_max_origin_hour": fitted["feature_max_origin_hour"],
        "train_rows": fitted["train_rows"],
        "weighted_train_rows": fitted["weighted_train_rows"],
        "weight_2_fraction": fitted["weight_2_fraction"],
        "high_load_count": int(high_mask.sum()),
        "m0_micro_wape": wape(actual, fitted["m0"]),
        "m1_micro_wape": wape(actual, fitted["m1"]),
        "historical_mean_micro_wape": wape(actual, fitted["historical_mean"]),
        "m0_rmse": rmse(actual, fitted["m0"]),
        "m1_rmse": rmse(actual, fitted["m1"]),
        "m0_median_series_wape": median_series_wape(actual, fitted["m0"], meta),
        "m1_median_series_wape": median_series_wape(actual, fitted["m1"], meta),
        "historical_mean_median_series_wape": median_series_wape(
            actual, fitted["historical_mean"], meta
        ),
        "m0_high_load_mean_underestimation": m0_high[
            "high_load_mean_underestimation"
        ],
        "m1_high_load_mean_underestimation": m1_high[
            "high_load_mean_underestimation"
        ],
        "m0_high_load_underestimation_rate": m0_high[
            "high_load_underestimation_rate"
        ],
        "m1_high_load_underestimation_rate": m1_high[
            "high_load_underestimation_rate"
        ],
        "m0_aggregate_curve_correlation": m0_agg["aggregate_curve_correlation"],
        "m1_aggregate_curve_correlation": m1_agg["aggregate_curve_correlation"],
        "m0_aggregate_peak_ratio": m0_agg["aggregate_peak_ratio"],
        "m1_aggregate_peak_ratio": m1_agg["aggregate_peak_ratio"],
        "m0_aggregate_peak_ratio_abs_error": m0_agg[
            "aggregate_peak_ratio_abs_error"
        ],
        "m1_aggregate_peak_ratio_abs_error": m1_agg[
            "aggregate_peak_ratio_abs_error"
        ],
        "m1_relative_wape_deterioration_vs_m0": (
            wape(actual, fitted["m1"]) - wape(actual, fitted["m0"])
        )
        / wape(actual, fitted["m0"]),
        "u1_pinball_loss_alpha_0_95": upper["pinball_loss_alpha_0_95"],
        "u1_overall_coverage": upper["overall_coverage"],
        "u1_high_load_coverage": upper["high_load_coverage"],
        "u1_high_load_underestimation_rate": upper[
            "high_load_underestimation_rate"
        ],
        "u1_high_load_mean_underestimation": upper[
            "high_load_mean_underestimation"
        ],
        "u1_upper95_ge_m1_all": upper["upper95_ge_m1_all"],
        "u1_upper95_below_m1_count": upper["upper95_below_m1_count"],
        "u1_minimum_upper95_minus_m1": upper["minimum_upper95_minus_m1"],
        "u1_unique_predictions_rounded_6dp": upper[
            "unique_predictions_rounded_6dp"
        ],
        "u1_all_zero_series": upper["all_zero_series"],
        "u1_nonfinite_count": upper["nonfinite_count"],
        "u1_mean_upper95": upper["mean_upper95"],
        "u1_mean_width_above_m1": upper["mean_width_above_m1"],
        "historical_q95_pinball_loss_alpha_0_95": hist_upper[
            "pinball_loss_alpha_0_95"
        ],
        "historical_q95_overall_coverage": hist_upper["overall_coverage"],
        "historical_q95_high_load_coverage": hist_upper["high_load_coverage"],
        "historical_q95_high_load_underestimation_rate": hist_upper[
            "high_load_underestimation_rate"
        ],
        "historical_q95_high_load_mean_underestimation": hist_upper[
            "high_load_mean_underestimation"
        ],
        "historical_q95_unique_predictions_rounded_6dp": hist_upper[
            "unique_predictions_rounded_6dp"
        ],
        "historical_q95_all_zero_series": hist_upper["all_zero_series"],
        "historical_q95_nonfinite_count": hist_upper["nonfinite_count"],
        **upper_cost,
    }
    detail = meta.copy()
    detail["actual"] = actual
    detail["series_training_p90"] = fitted["series_p90"]
    detail["is_high_load"] = high_mask
    detail["m0_prediction"] = fitted["m0"]
    detail["m1_prediction"] = fitted["m1"]
    detail["q95_lightgbm_prediction"] = fitted["q95_model"]
    detail["historical_mean_prediction"] = fitted["historical_mean"]
    detail["historical_q95_prediction"] = fitted["historical_q95"]
    detail["upper95"] = fitted["upper95"]
    aggregate = detail.groupby("target_hour", sort=True)[
        [
            "actual",
            "m0_prediction",
            "m1_prediction",
            "q95_lightgbm_prediction",
            "historical_q95_prediction",
            "upper95",
        ]
    ].sum()
    aggregate = aggregate.reset_index()
    aggregate["window"] = f"{start}-{end}"
    aggregate["role"] = role
    return summary, detail, aggregate


def add_adoption_gates(rows: list[dict[str, Any]]) -> None:
    official = next(row for row in rows if row["window"] == "2352-2375")
    official_median_gate = (
        official["m1_median_series_wape"]
        <= official["historical_mean_median_series_wape"] + 1e-12
    )
    for row in rows:
        gates = {
            "wape_relative_deterioration_le_0_5pct": row[
                "m1_relative_wape_deterioration_vs_m0"
            ]
            <= 0.005 + 1e-12,
            "high_load_mean_underestimation_lt_m0": row[
                "m1_high_load_mean_underestimation"
            ]
            < row["m0_high_load_mean_underestimation"],
            "aggregate_curve_correlation_gt_m0": row[
                "m1_aggregate_curve_correlation"
            ]
            > row["m0_aggregate_curve_correlation"],
            "aggregate_peak_ratio_closer_to_1_than_m0": row[
                "m1_aggregate_peak_ratio_abs_error"
            ]
            < row["m0_aggregate_peak_ratio_abs_error"],
            "official_validation_median_series_wape_not_worse_than_historical_mean": official_median_gate,
        }
        row.update({f"gate_{key}": value for key, value in gates.items()})
        row["m1_window_adoption_pass"] = bool(all(gates.values()))


def render_figures(aggregates: pd.DataFrame) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "svg.hashsalt": "q1_peak_patch_20260810",
        }
    )
    windows = [f"{a}-{b}" for a, b in [*SELECTION_WINDOWS, POST_HOC_WINDOW]]
    figure_paths: list[Path] = []

    def plot_a() -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
        for axis, window in zip(axes.flat, windows, strict=True):
            frame = aggregates.loc[aggregates.window == window]
            axis.plot(frame.target_hour, frame.actual, label="Actual", color="#222222", lw=2.2)
            axis.plot(frame.target_hour, frame.m0_prediction, label="M0", color="#4C78A8", lw=1.8)
            axis.plot(frame.target_hour, frame.m1_prediction, label="M1", color="#E45756", lw=1.8)
            suffix = " (post-hoc)" if window == "2376-2399" else ""
            axis.set_title(f"{window}{suffix}")
            axis.set_xlabel("Hour")
            axis.set_ylabel("Aggregate GPU demand")
            axis.grid(alpha=0.22)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
        )
        fig.suptitle("A. Aggregate actual curve vs M0 and M1", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        return fig

    def plot_b() -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
        for axis, window in zip(axes.flat, windows, strict=True):
            frame = aggregates.loc[aggregates.window == window]
            axis.plot(frame.target_hour, frame.actual, label="Actual", color="#222222", lw=2.2)
            axis.plot(frame.target_hour, frame.m1_prediction, label="M1 point", color="#E45756", lw=1.8)
            axis.plot(frame.target_hour, frame.upper95, label="upper95", color="#54A24B", lw=1.8)
            axis.fill_between(
                frame.target_hour,
                frame.m1_prediction,
                frame.upper95,
                color="#54A24B",
                alpha=0.12,
            )
            suffix = " (post-hoc)" if window == "2376-2399" else ""
            axis.set_title(f"{window}{suffix}")
            axis.set_xlabel("Hour")
            axis.set_ylabel("Aggregate GPU demand")
            axis.grid(alpha=0.22)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
        )
        fig.suptitle("B. Aggregate actual curve, M1 point forecast and upper95", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        return fig

    for stem, builder in [
        ("figure_A_aggregate_m0_m1", plot_a),
        ("figure_B_aggregate_m1_upper95", plot_b),
    ]:
        figure = builder()
        for suffix in ["png", "svg"]:
            path = FIGURE_DIR / f"{stem}.{suffix}"
            metadata_payload = {"Software": "Q1 peak patch candidate"}
            if suffix == "svg":
                metadata_payload = {"Date": "2026-08-10", "Creator": "Q1 peak patch candidate"}
            figure.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata_payload)
            figure_paths.append(path)
        plt.close(figure)
    return figure_paths


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ["numpy", "pandas", "lightgbm", "matplotlib", "scikit-learn"]:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def run() -> dict[str, Any]:
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    for directory in [TABLE_DIR, FIGURE_DIR, REVIEW_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    full_data = load_data()
    selection_rows: list[dict[str, Any]] = []
    aggregate_frames: list[pd.DataFrame] = []
    output_paths: list[Path] = []

    for start, end in SELECTION_WINDOWS:
        summary, detail, aggregate = evaluate_one_window(
            full_data, start, end, role="selection"
        )
        selection_rows.append(summary)
        aggregate_frames.append(aggregate)
        detail_path = TABLE_DIR / f"selection_predictions_{start}_{end}.csv"
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        output_paths.append(detail_path)

    add_adoption_gates(selection_rows)
    selection_table = pd.DataFrame(selection_rows)
    selection_table_path = TABLE_DIR / "selection_window_summary.csv"
    selection_table.to_csv(selection_table_path, index=False, encoding="utf-8-sig")
    output_paths.append(selection_table_path)

    fixed_config_hash = config_sha256()
    candidate_fixed_before_post_hoc = True
    test_summary, test_detail, test_aggregate = evaluate_one_window(
        full_data, *POST_HOC_WINDOW, role="post_hoc"
    )
    aggregate_frames.append(test_aggregate)
    test_detail_path = TABLE_DIR / "post_hoc_test_predictions_2376_2399.csv"
    test_detail.to_csv(test_detail_path, index=False, encoding="utf-8-sig")
    test_summary_path = TABLE_DIR / "post_hoc_test_summary.csv"
    pd.DataFrame([test_summary]).to_csv(
        test_summary_path, index=False, encoding="utf-8-sig"
    )
    output_paths.extend([test_detail_path, test_summary_path])

    aggregate_table = pd.concat(aggregate_frames, ignore_index=True)
    aggregate_path = TABLE_DIR / "aggregate_curves_all_windows.csv"
    aggregate_table.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    output_paths.append(aggregate_path)
    figure_paths = render_figures(aggregate_table)
    output_paths.extend(figure_paths)

    all_selection_gates_pass = bool(
        all(row["m1_window_adoption_pass"] for row in selection_rows)
    )
    u1_integrity_pass = bool(
        all(
            row["u1_upper95_ge_m1_all"]
            and row["u1_nonfinite_count"] == 0
            and row["u1_unique_predictions_rounded_6dp"] > 1
            and row["u1_all_zero_series"] == 0
            for row in selection_rows
        )
    )
    verdict = "PASS" if all_selection_gates_pass and u1_integrity_pass else "FAIL"
    failed_gates = [
        {"window": row["window"], "gate": key, "observed": value}
        for row in selection_rows
        for key, value in row.items()
        if key.startswith("gate_") and value is False
    ]
    artifact_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in output_paths
    }
    payload = {
        "schema_version": 1,
        "question_id": "Q1",
        "experiment_id": "peak_patch_20260810",
        "change_level": "FROZEN",
        "status": "completed",
        "verdict": verdict,
        "formal_results_modified": False,
        "promotion_authorized": False,
        "candidate_output_only": True,
        "approved_decision_context": "Q1-METHOD-CHOICE-001; user-authorized fixed post-freeze candidate experiment",
        "candidate_config": CANDIDATE_CONFIG,
        "candidate_config_sha256": fixed_config_hash,
        "candidate_fixed_before_post_hoc": candidate_fixed_before_post_hoc,
        "selection_windows": selection_rows,
        "selection_gate_summary": {
            "all_m1_window_adoption_gates_pass": all_selection_gates_pass,
            "u1_integrity_checks_pass": u1_integrity_pass,
            "failed_gates": failed_gates,
            "action": "keep existing formal results unchanged"
            if verdict != "PASS"
            else "candidate may be presented for a new human adoption decision; no automatic promotion",
        },
        "post_hoc_extension": {
            "label": "post-hoc extension; not a new unseen-test proof",
            "used_for_selection": False,
            "evaluated_once_after_config_fixed": True,
            "metrics": test_summary,
        },
        "leakage_evidence": {
            "selection_windows_only": [f"{a}-{b}" for a, b in SELECTION_WINDOWS],
            "each_window_training_thresholds_weights_and_features_use_hours_le_start_minus_1": True,
            "final_test_targets_used_for_training": 0,
            "final_test_targets_used_for_tuning_or_candidate_selection": 0,
            "maximum_history_hour_for_post_hoc_features": POST_HOC_WINDOW[0] - 1,
        },
        "excluded_methods_executed": {
            "seasonal_naive_mixture": False,
            "task_type_specific_models": False,
            "large_search": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "sklearn_wrapper_used": False,
            "lightgbm_interface": "lightgbm.train",
            "input_sha256": sha256(DATA_PATH),
            "random_seed": SEED,
            "run_command": "<.venv python> 04_代码实现/question1/q1_peak_patch_experiment.py",
        },
        "execution_time_seconds": time.perf_counter() - started,
        "artifacts": artifact_hashes,
        "warnings": [
            "2376-2399 was already inspected in the original workflow; its new metrics are post-hoc descriptive only."
        ],
        "errors": [],
    }
    summary_path = OUTPUT_DIR / "candidate_run_summary.json"
    write_json(summary_path, payload)
    return payload


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "execution_time_seconds": result["execution_time_seconds"],
                "failed_gates": result["selection_gate_summary"]["failed_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
