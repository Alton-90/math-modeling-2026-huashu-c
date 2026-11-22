"""Legacy Q1 round1 predictor/ablation helpers and unchanged scheduling implementation.

The approved FROZEN point/risk forecast is implemented in
``q1_peak_patch_experiment.py``.  This module remains authoritative for the
unchanged deterministic scheduling functions and for reproducing the old M0
and 1.25x-correction ablations only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[2]
ROUND_DIR = ROOT / "05_结果与图表" / "intermediate" / "question1" / "experiments" / "round1"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
TARGET = "GPU_Demand_Arrival_Sum"
SEED = 20260808
TRAIN_END = 2351
VALID_START = 2352
VALID_END = 2375
TEST_START = 2376
TEST_END = 2399
HORIZON_END = 2406
STABILITY_WINDOWS = [(2304, 2327), (2328, 2351)]
TASK_WEIGHTS = {"RealTimeInference": 10, "BatchInference": 3, "AITraining": 1}
MODEL_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 97,
    "num_leaves": 15,
    "min_child_samples": 40,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "n_jobs": min(8, os.cpu_count() or 1),
}
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


def prepare_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def set_reproducible_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


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


def wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    return float(np.abs(actual - prediction).sum() / denominator) if denominator > 0 else math.nan


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def prediction_metrics(actual: np.ndarray, prediction: np.ndarray, meta: pd.DataFrame) -> dict:
    frame = meta[["SourceRegion", "TaskType", "target_hour"]].copy()
    frame["actual"] = actual
    frame["prediction"] = prediction
    by_series = {
        f"{key[0]}|{key[1]}": wape(
            group.actual.to_numpy(dtype=float), group.prediction.to_numpy(dtype=float)
        )
        for key, group in frame.groupby(["SourceRegion", "TaskType"], sort=True)
    }
    finite = [value for value in by_series.values() if math.isfinite(value)]
    return {
        "micro_wape": wape(actual, prediction),
        "rmse": rmse(actual, prediction),
        "median_series_wape": float(np.median(finite)) if finite else math.nan,
        "by_series_wape": by_series,
    }


def load_prediction_data() -> pd.DataFrame:
    path = ROOT / "02_数据处理" / "processed" / "q1_hourly_gpu_demand.csv"
    frame = pd.read_csv(path, usecols=["Hour", "SourceRegion", "TaskType", TARGET])
    expected = 2400 * 18
    if len(frame) != expected:
        raise ValueError(f"预测输入应为 {expected} 行，实际为 {len(frame)} 行")
    if frame[TARGET].isna().any():
        raise ValueError("预测目标存在缺失值")
    if frame.duplicated(["Hour", "SourceRegion", "TaskType"]).any():
        raise ValueError("预测输入主键重复")
    return frame.sort_values(["SourceRegion", "TaskType", "Hour"]).reset_index(drop=True)


def build_feature_rows(
    history: pd.DataFrame,
    first_target: int,
    last_target: int,
) -> pd.DataFrame:
    history_end = int(history.Hour.max())
    rows: list[dict] = []
    for (region, task_type), group in history.groupby(["SourceRegion", "TaskType"], sort=True):
        series = group.set_index("Hour")[TARGET].reindex(range(history_end + 1))
        if series.isna().any():
            raise ValueError(f"{region}|{task_type} 的历史序列不完整")
        values = series.to_numpy(dtype=float)
        for target_hour in range(first_target, last_target + 1):
            block_start = (target_hour // 24) * 24
            origin = block_start - 1
            if origin < 167 or origin > history_end:
                raise ValueError(
                    f"目标小时 {target_hour} 的预测原点 {origin} 不在可用历史 0–{history_end} 内"
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


def encode_features(train_rows: pd.DataFrame, forecast_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def fit_fixed_model_window(
    full_data: pd.DataFrame,
    train_end: int,
    forecast_start: int,
    forecast_end: int,
) -> dict:
    if forecast_start != train_end + 1 or forecast_end != forecast_start + 23:
        raise ValueError("正式预测仅支持紧接训练段的完整 24 小时块")
    history = full_data.loc[full_data.Hour <= train_end].copy()
    train_rows = build_feature_rows(history, 168, train_end)
    forecast_rows = build_feature_rows(history, forecast_start, forecast_end)
    target_lookup = history.set_index(["SourceRegion", "TaskType", "Hour"])[TARGET]
    y_train = np.asarray(
        [
            target_lookup.loc[(row.SourceRegion, row.TaskType, row.target_hour)]
            for row in train_rows.itertuples(index=False)
        ],
        dtype=float,
    )
    x_train, x_forecast = encode_features(train_rows, forecast_rows)
    model = LGBMRegressor(**MODEL_PARAMS)
    start = time.perf_counter()
    model.fit(x_train, y_train)
    prediction = np.clip(model.predict(x_forecast), 0, None)
    runtime = time.perf_counter() - start
    importance = pd.DataFrame(
        {"feature": x_train.columns, "importance": model.feature_importances_.astype(int)}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    return {
        "prediction": prediction,
        "forecast_rows": forecast_rows.reset_index(drop=True),
        "importance": importance.reset_index(drop=True),
        "runtime_seconds": runtime,
        "train_rows": len(train_rows),
        "forecast_rows_count": len(forecast_rows),
    }


def actual_for_window(full_data: pd.DataFrame, start: int, end: int) -> tuple[np.ndarray, pd.DataFrame]:
    frame = full_data.loc[full_data.Hour.between(start, end)].copy()
    frame = frame.sort_values(["SourceRegion", "TaskType", "Hour"]).reset_index(drop=True)
    meta = frame[["SourceRegion", "TaskType", "Hour"]].rename(columns={"Hour": "target_hour"})
    return frame[TARGET].to_numpy(dtype=float), meta


def baseline_predictions(
    full_data: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    train_end: int,
) -> dict[str, np.ndarray]:
    history = full_data.loc[full_data.Hour <= train_end]
    mean_map = history.groupby(["SourceRegion", "TaskType"])[TARGET].mean()
    last_map = history.loc[history.Hour == train_end].set_index(["SourceRegion", "TaskType"])[TARGET]
    lookup = history.set_index(["SourceRegion", "TaskType", "Hour"])[TARGET]
    means: list[float] = []
    lasts: list[float] = []
    lag24: list[float] = []
    for row in forecast_rows.itertuples(index=False):
        key = (row.SourceRegion, row.TaskType)
        means.append(float(mean_map.loc[key]))
        lasts.append(float(last_map.loc[key]))
        lag24.append(float(lookup.loc[(key[0], key[1], row.target_hour - 24)]))
    return {
        "historical_mean": np.asarray(means),
        "previous_1_hour": np.asarray(lasts),
        "previous_24_hours": np.asarray(lag24),
    }


def evaluate_window(
    full_data: pd.DataFrame,
    train_end: int,
    forecast_start: int,
    forecast_end: int,
) -> dict:
    fitted = fit_fixed_model_window(full_data, train_end, forecast_start, forecast_end)
    actual, meta = actual_for_window(full_data, forecast_start, forecast_end)
    if not np.array_equal(meta.to_numpy(), fitted["forecast_rows"][["SourceRegion", "TaskType", "target_hour"]].to_numpy()):
        raise AssertionError("预测行与真实值行未对齐")
    baselines = baseline_predictions(full_data, fitted["forecast_rows"], train_end)
    main_metrics = prediction_metrics(actual, fitted["prediction"], meta)
    baseline_metrics = {
        name: prediction_metrics(actual, prediction, meta) for name, prediction in baselines.items()
    }
    best_baseline = min(baseline_metrics, key=lambda name: baseline_metrics[name]["micro_wape"])
    baseline_wape = baseline_metrics[best_baseline]["micro_wape"]
    improvement = (baseline_wape - main_metrics["micro_wape"]) / baseline_wape
    return {
        **fitted,
        "actual": actual,
        "meta": meta,
        "baselines": baselines,
        "main_metrics": main_metrics,
        "baseline_metrics": baseline_metrics,
        "best_baseline": best_baseline,
        "relative_improvement": improvement,
    }


def high_load_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    high_mask: np.ndarray,
) -> dict:
    if not high_mask.any():
        return {
            "high_load_count": 0,
            "high_load_underestimation_rate": math.nan,
            "high_load_mean_underestimation": math.nan,
        }
    residual = actual[high_mask] - prediction[high_mask]
    return {
        "high_load_count": int(high_mask.sum()),
        "high_load_underestimation_rate": float(np.mean(residual > 0)),
        "high_load_mean_underestimation": float(np.maximum(residual, 0).mean()),
    }


def calibrate_conservative_correction(
    full_data: pd.DataFrame,
    validation: dict,
) -> dict:
    thresholds = (
        full_data.loc[full_data.Hour <= TRAIN_END]
        .groupby(["SourceRegion", "TaskType"])[TARGET]
        .quantile(0.9)
    )
    rows = validation["forecast_rows"]
    actual = validation["actual"]
    prediction = validation["prediction"]
    threshold_values = np.asarray(
        [thresholds.loc[(row.SourceRegion, row.TaskType)] for row in rows.itertuples(index=False)],
        dtype=float,
    )
    high_mask = actual >= threshold_values
    risk_flag = np.maximum.reduce(
        [
            prediction,
            rows["lag24"].to_numpy(dtype=float),
            rows["lag168"].to_numpy(dtype=float),
            rows["origin_max24"].to_numpy(dtype=float),
        ]
    ) >= threshold_values
    raw_ratio = (
        float(actual[high_mask].sum() / prediction[high_mask].sum())
        if high_mask.any() and prediction[high_mask].sum() > 0
        else 1.0
    )
    factor = float(np.clip(1.0 + 0.25 * max(0.0, raw_ratio - 1.0), 1.0, 1.25))
    corrected = prediction.copy()
    corrected[risk_flag] *= factor
    original_overall = wape(actual, prediction)
    corrected_overall = wape(actual, corrected)
    relative_degradation = (corrected_overall - original_overall) / original_overall
    original_risk = high_load_metrics(actual, prediction, high_mask)
    corrected_risk = high_load_metrics(actual, corrected, high_mask)
    adopted = bool(
        relative_degradation <= 0.05
        and corrected_risk["high_load_underestimation_rate"]
        <= original_risk["high_load_underestimation_rate"]
        and corrected_risk["high_load_mean_underestimation"]
        < original_risk["high_load_mean_underestimation"]
    )
    return {
        "thresholds": thresholds,
        "raw_high_load_bias_ratio": raw_ratio,
        "factor": factor,
        "risk_flag": risk_flag,
        "high_mask": high_mask,
        "corrected_prediction": corrected,
        "original_overall_wape": original_overall,
        "corrected_overall_wape": corrected_overall,
        "relative_wape_degradation": relative_degradation,
        "original_risk": original_risk,
        "corrected_risk": corrected_risk,
        "risk_flagged_count": int(risk_flag.sum()),
        "adopted_as_conservative_companion": adopted,
    }


def apply_conservative_correction(
    fitted: dict,
    actual: np.ndarray,
    thresholds: pd.Series,
    factor: float,
) -> dict:
    rows = fitted["forecast_rows"]
    prediction = fitted["prediction"]
    threshold_values = np.asarray(
        [thresholds.loc[(row.SourceRegion, row.TaskType)] for row in rows.itertuples(index=False)],
        dtype=float,
    )
    high_mask = actual >= threshold_values
    risk_flag = np.maximum.reduce(
        [
            prediction,
            rows["lag24"].to_numpy(dtype=float),
            rows["lag168"].to_numpy(dtype=float),
            rows["origin_max24"].to_numpy(dtype=float),
        ]
    ) >= threshold_values
    corrected = prediction.copy()
    corrected[risk_flag] *= factor
    return {
        "corrected_prediction": corrected,
        "risk_flag": risk_flag,
        "high_mask": high_mask,
        "risk_flagged_count": int(risk_flag.sum()),
        "original_overall_wape": wape(actual, prediction),
        "corrected_overall_wape": wape(actual, corrected),
        "original_risk": high_load_metrics(actual, prediction, high_mask),
        "corrected_risk": high_load_metrics(actual, corrected, high_mask),
    }


def build_prediction_outputs(full_data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    validation = evaluate_window(full_data, TRAIN_END, VALID_START, VALID_END)
    stability_rows: list[dict] = []
    stability_details: list[dict] = []
    for start, end in STABILITY_WINDOWS:
        result = evaluate_window(full_data, start - 1, start, end)
        stability_rows.append(
            {
                "window": f"{start}-{end}",
                "lightgbm_wape": result["main_metrics"]["micro_wape"],
                "best_simple_baseline": result["best_baseline"],
                "best_simple_baseline_wape": result["baseline_metrics"][result["best_baseline"]]["micro_wape"],
                "relative_improvement": result["relative_improvement"],
                "lightgbm_overall_better": result["relative_improvement"] > 0,
            }
        )
        stability_details.append(result)

    correction = calibrate_conservative_correction(full_data, validation)
    final_test = evaluate_window(full_data, VALID_END, TEST_START, TEST_END)
    test_correction = apply_conservative_correction(
        final_test,
        final_test["actual"],
        correction["thresholds"],
        correction["factor"],
    )

    comparison_rows: list[dict] = []
    for split_name, result in [("validation_2352_2375", validation), ("test_2376_2399", final_test)]:
        comparison_rows.append(
            {
                "split": split_name,
                "method": "LightGBM",
                **{key: value for key, value in result["main_metrics"].items() if key != "by_series_wape"},
            }
        )
        for method, metrics in result["baseline_metrics"].items():
            comparison_rows.append(
                {
                    "split": split_name,
                    "method": method,
                    **{key: value for key, value in metrics.items() if key != "by_series_wape"},
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TABLE_DIR / "prediction_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stability_rows).to_csv(
        TABLE_DIR / "stability_windows.csv", index=False, encoding="utf-8-sig"
    )
    final_test["importance"].to_csv(
        TABLE_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    def save_prediction_detail(
        result: dict,
        corrected: np.ndarray,
        risk_flag: np.ndarray,
        high_mask: np.ndarray,
        filename: str,
    ) -> None:
        detail = result["meta"].copy()
        detail["actual"] = result["actual"]
        detail["lightgbm_prediction"] = result["prediction"]
        for name, prediction in result["baselines"].items():
            detail[name] = prediction
        detail["conservative_prediction"] = corrected
        detail["risk_flag"] = risk_flag
        detail["actual_high_load"] = high_mask
        detail.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")

    save_prediction_detail(
        validation,
        correction["corrected_prediction"],
        correction["risk_flag"],
        correction["high_mask"],
        "prediction_results_validation.csv",
    )
    save_prediction_detail(
        final_test,
        test_correction["corrected_prediction"],
        test_correction["risk_flag"],
        test_correction["high_mask"],
        "prediction_results_test.csv",
    )

    correction_rows: list[dict] = []
    for split_name, result in [("validation_2352_2375", correction), ("test_2376_2399", test_correction)]:
        for version, risk_key, wape_key in [
            ("original", "original_risk", "original_overall_wape"),
            ("corrected", "corrected_risk", "corrected_overall_wape"),
        ]:
            correction_rows.append(
                {
                    "split": split_name,
                    "version": version,
                    "overall_wape": result[wape_key],
                    **result[risk_key],
                    "correction_factor": correction["factor"],
                    "risk_flagged_count": result["risk_flagged_count"],
                    "adopted_as_conservative_companion": correction["adopted_as_conservative_companion"],
                }
            )
    pd.DataFrame(correction_rows).to_csv(
        TABLE_DIR / "high_load_correction.csv", index=False, encoding="utf-8-sig"
    )

    prediction_payload = {
        "official_validation": {
            "train_hours": [0, TRAIN_END],
            "validation_hours": [VALID_START, VALID_END],
            "main": validation["main_metrics"],
            "baselines": validation["baseline_metrics"],
            "best_simple_baseline": validation["best_baseline"],
            "relative_improvement": validation["relative_improvement"],
        },
        "stability_windows": stability_rows,
        "conservative_correction": {
            key: value
            for key, value in correction.items()
            if key not in {"thresholds", "risk_flag", "high_mask", "corrected_prediction"}
        },
        "official_test": {
            "refit_hours": [0, VALID_END],
            "test_hours": [TEST_START, TEST_END],
            "main": final_test["main_metrics"],
            "baselines": final_test["baseline_metrics"],
            "best_simple_baseline": final_test["best_baseline"],
            "relative_improvement": final_test["relative_improvement"],
            "correction": {
                key: value
                for key, value in test_correction.items()
                if key not in {"risk_flag", "high_mask", "corrected_prediction"}
            },
        },
        "leakage_evidence": {
            "final_test_target_rows_used_for_training": 0,
            "final_test_target_rows_used_for_tuning": 0,
            "final_test_target_rows_used_for_correction_calibration": 0,
            "maximum_history_hour_for_test_features": VALID_END,
            "test_targets_used_only_after_predictions_frozen_for_evaluation": True,
        },
        "output_degeneracy": {
            "validation_unique_predictions_rounded_6dp": int(np.unique(np.round(validation["prediction"], 6)).size),
            "test_unique_predictions_rounded_6dp": int(np.unique(np.round(final_test["prediction"], 6)).size),
            "validation_zero_prediction_rate": float(np.mean(validation["prediction"] <= 1e-9)),
            "test_zero_prediction_rate": float(np.mean(final_test["prediction"] <= 1e-9)),
            "all_zero_validation_series": [
                key
                for key, indices in validation["meta"].groupby(["SourceRegion", "TaskType"]).groups.items()
                if np.all(validation["prediction"][list(indices)] <= 1e-9)
            ],
            "all_zero_test_series": [
                key
                for key, indices in final_test["meta"].groupby(["SourceRegion", "TaskType"]).groups.items()
                if np.all(final_test["prediction"][list(indices)] <= 1e-9)
            ],
        },
        "model_params": MODEL_PARAMS,
        "runtime_seconds": {
            "official_validation": validation["runtime_seconds"],
            "stability_windows": [item["runtime_seconds"] for item in stability_details],
            "official_refit_and_test": final_test["runtime_seconds"],
        },
    }
    write_json(METRIC_DIR / "prediction_metrics.json", prediction_payload)
    return prediction_payload, comparison


def task_overlap_minutes(start_hour: int, duration_min: int) -> list[tuple[int, int]]:
    start_minute = start_hour * 60
    end_minute = start_minute + duration_min
    overlaps: list[tuple[int, int]] = []
    for hour in range(start_hour, int(math.ceil(end_minute / 60))):
        overlap = max(0, min(end_minute, (hour + 1) * 60) - max(start_minute, hour * 60))
        if overlap > 0:
            overlaps.append((hour, int(overlap)))
    return overlaps


def load_schedule_inputs(tasks_override: pd.DataFrame | None = None) -> dict:
    data_dir = ROOT / "02_数据处理" / "raw"
    tasks = pd.read_excel(data_dir / "workload_trace.xlsx")
    tasks = tasks.loc[tasks.ArrivalHour.between(TEST_START, TEST_END)].copy()
    tasks["TaskID"] = tasks.TaskID.astype(int)
    tasks["EstimatedDuration_min"] = tasks.EstimatedDuration_min.astype(int)
    if tasks_override is not None:
        tasks = tasks_override.copy()
    gpu = pd.read_excel(data_dir / "GPU_information.xlsx")
    latency = pd.read_excel(data_dir / "network_latency.xlsx")
    power = pd.read_excel(data_dir / "power_mapping.xlsx")
    region_time = pd.read_excel(data_dir / "region_time_data.xlsx")
    region_time = region_time.loc[
        region_time.Hour.between(TEST_START, HORIZON_END - 1),
        ["Hour", "Region", "NonAI_IT_Load_MW"],
    ].copy()
    return {
        "tasks": tasks.reset_index(drop=True),
        "regions": gpu.Region.astype(str).tolist(),
        "gpu_cap": gpu.set_index("Region").Available_GPU.astype(int).to_dict(),
        "max_it": gpu.set_index("Region").Max_IT_Power_MW.astype(float).to_dict(),
        "max_facility": gpu.set_index("Region").Max_Facility_Power_MW.astype(float).to_dict(),
        "pue": gpu.set_index("Region").PUE.astype(float).to_dict(),
        "power_map": power.set_index("TaskType").GPU_Power_MW_per_EquivalentGPU.astype(float).to_dict(),
        "latency": latency.set_index(["FromRegion", "ToRegion"]).NetworkLatency_ms.astype(int).to_dict(),
        "nonai": region_time.set_index(["Region", "Hour"]).NonAI_IT_Load_MW.astype(float).to_dict(),
    }


def candidates_for_task(row: pd.Series, inputs: dict) -> list[tuple[str, int, list[tuple[int, int]]]]:
    arrival = int(row.ArrivalHour)
    duration = int(row.EstimatedDuration_min)
    latest_finish_min = min(int(row.LatestFinishHour) * 60, HORIZON_END * 60)
    max_start = (latest_finish_min - duration) // 60
    if str(row.TaskType) == "RealTimeInference":
        starts = [arrival]
    else:
        starts = range(max(arrival, int(row.EarliestStartHour)), max_start + 1)
    candidates: list[tuple[str, int, list[tuple[int, int]]]] = []
    for region in inputs["regions"]:
        if inputs["latency"][(str(row.SourceRegion), region)] > int(row.MaxLatency_ms):
            continue
        for start in starts:
            if start < TEST_START or start * 60 + duration > latest_finish_min:
                continue
            overlaps = task_overlap_minutes(start, duration)
            if any(hour < TEST_START or hour >= HORIZON_END for hour, _ in overlaps):
                continue
            candidates.append((region, int(start), overlaps))
    return candidates


def deterministic_schedule(inputs: dict) -> tuple[pd.DataFrame, dict]:
    tasks = inputs["tasks"].copy()
    tasks["slack_min"] = (
        np.minimum(tasks.LatestFinishHour, HORIZON_END) * 60
        - tasks.ArrivalHour * 60
        - tasks.EstimatedDuration_min
    )
    tasks["rt_first"] = (tasks.TaskType != "RealTimeInference").astype(int)
    tasks["type_order"] = tasks.TaskType.map({"RealTimeInference": 0, "BatchInference": 1, "AITraining": 2})
    tasks.sort_values(
        ["rt_first", "slack_min", "type_order", "ArrivalHour", "TaskID"], inplace=True
    )
    gpu_load = {(region, hour): 0 for region in inputs["regions"] for hour in range(TEST_START, HORIZON_END)}
    ai_power_energy = {
        (region, hour): 0.0 for region in inputs["regions"] for hour in range(TEST_START, HORIZON_END)
    }
    assignments: list[dict] = []
    candidate_counts: list[int] = []
    for _, row in tasks.iterrows():
        candidates = candidates_for_task(row, inputs)
        candidate_counts.append(len(candidates))
        feasible: list[tuple[tuple, str, int, list[tuple[int, int]]]] = []
        for region, start, overlaps in candidates:
            projected_peak = 0.0
            valid = True
            for hour, overlap in overlaps:
                projected_gpu = gpu_load[(region, hour)] + int(row.GPU_Demand) * overlap
                projected_ai_energy = (
                    ai_power_energy[(region, hour)]
                    + int(row.GPU_Demand)
                    * inputs["power_map"][str(row.TaskType)]
                    * overlap
                )
                projected_ai_power = projected_ai_energy / 60.0
                total_it = inputs["nonai"][(region, hour)] + projected_ai_power
                facility = total_it * inputs["pue"][region]
                if (
                    projected_gpu > inputs["gpu_cap"][region] * 60 + 1e-9
                    or total_it > inputs["max_it"][region] + 1e-9
                    or facility > inputs["max_facility"][region] + 1e-9
                ):
                    valid = False
                    break
                projected_peak = max(
                    projected_peak,
                    projected_gpu / (inputs["gpu_cap"][region] * 60),
                )
            if valid:
                latency = inputs["latency"][(str(row.SourceRegion), region)]
                wait_min = (start - int(row.ArrivalHour)) * 60
                service_time_minutes = wait_min + latency / 60_000.0
                score = (
                    service_time_minutes,
                    int(region != str(row.SourceRegion)),
                    projected_peak,
                    latency,
                    region,
                    start,
                )
                feasible.append((score, region, start, overlaps))
        if not feasible:
            continue
        _, region, start, overlaps = min(feasible, key=lambda item: item[0])
        assignments.append(
            {
                "TaskID": int(row.TaskID),
                "AssignedRegion": region,
                "StartHour": int(start),
                "FinishMinute": int(start * 60 + int(row.EstimatedDuration_min)),
            }
        )
        for hour, overlap in overlaps:
            gpu_load[(region, hour)] += int(row.GPU_Demand) * overlap
            ai_power_energy[(region, hour)] += (
                int(row.GPU_Demand)
                * inputs["power_map"][str(row.TaskType)]
                * overlap
            )
    assignment_frame = pd.DataFrame(
        assignments,
        columns=["TaskID", "AssignedRegion", "StartHour", "FinishMinute"],
    )
    diagnostics = {
        "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
        "candidate_count_median": float(np.median(candidate_counts)) if candidate_counts else 0,
        "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
    }
    return assignment_frame, diagnostics


def validate_schedule(assignments: pd.DataFrame, inputs: dict) -> dict:
    tasks = inputs["tasks"].set_index("TaskID")
    assignment_map = assignments.set_index("TaskID") if not assignments.empty else pd.DataFrame()
    gpu_load = {(region, hour): 0 for region in inputs["regions"] for hour in range(TEST_START, HORIZON_END)}
    ai_power_energy = {
        (region, hour): 0.0 for region in inputs["regions"] for hour in range(TEST_START, HORIZON_END)
    }
    violations = {
        "missing_assignment": 0,
        "duplicate_assignment": int(assignments.TaskID.duplicated().sum()) if not assignments.empty else 0,
        "start_or_realtime": 0,
        "latency": 0,
        "finish": 0,
        "gpu": 0,
        "it_power": 0,
        "facility_power": 0,
    }
    total_wait = 0.0
    weighted_wait = 0.0
    weighted_latency = 0.0
    migrations = 0
    assigned_count = 0
    for task_id, row in tasks.iterrows():
        if assignments.empty or task_id not in assignment_map.index:
            violations["missing_assignment"] += 1
            continue
        assigned = assignment_map.loc[task_id]
        if isinstance(assigned, pd.DataFrame):
            assigned = assigned.iloc[0]
        assigned_count += 1
        region = str(assigned.AssignedRegion)
        start = int(assigned.StartHour)
        duration = int(row.EstimatedDuration_min)
        latency = inputs["latency"][(str(row.SourceRegion), region)]
        wait = (start - int(row.ArrivalHour)) * 60
        total_wait += wait
        weighted_wait += TASK_WEIGHTS[str(row.TaskType)] * wait
        weighted_latency += TASK_WEIGHTS[str(row.TaskType)] * latency
        migrations += int(region != str(row.SourceRegion))
        if start < max(int(row.ArrivalHour), int(row.EarliestStartHour)):
            violations["start_or_realtime"] += 1
        if str(row.TaskType) == "RealTimeInference" and start != int(row.ArrivalHour):
            violations["start_or_realtime"] += 1
        if latency > int(row.MaxLatency_ms):
            violations["latency"] += 1
        finish_minute = start * 60 + duration
        if finish_minute > min(int(row.LatestFinishHour) * 60, HORIZON_END * 60):
            violations["finish"] += 1
        for hour, overlap in task_overlap_minutes(start, duration):
            if hour < TEST_START or hour >= HORIZON_END:
                violations["finish"] += 1
                continue
            gpu_load[(region, hour)] += int(row.GPU_Demand) * overlap
            ai_power_energy[(region, hour)] += (
                int(row.GPU_Demand)
                * inputs["power_map"][str(row.TaskType)]
                * overlap
            )
    utilizations: list[float] = []
    for region in inputs["regions"]:
        for hour in range(TEST_START, HORIZON_END):
            gpu_utilization = gpu_load[(region, hour)] / (inputs["gpu_cap"][region] * 60)
            utilizations.append(gpu_utilization)
            ai_power = ai_power_energy[(region, hour)] / 60.0
            total_it = inputs["nonai"][(region, hour)] + ai_power
            facility = total_it * inputs["pue"][region]
            if gpu_utilization > 1 + 1e-9:
                violations["gpu"] += 1
            if total_it > inputs["max_it"][region] + 1e-9:
                violations["it_power"] += 1
            if facility > inputs["max_facility"][region] + 1e-9:
                violations["facility_power"] += 1
    total_tasks = len(tasks)
    return {
        "assigned_tasks": assigned_count,
        "total_tasks": total_tasks,
        "task_completion_rate": assigned_count / total_tasks if total_tasks else math.nan,
        "constraint_violations": violations,
        "hard_constraint_violation_count": int(sum(violations.values())),
        "migrations": migrations,
        "migration_rate": migrations / total_tasks if total_tasks else math.nan,
        "total_wait_minutes": total_wait,
        "average_wait_minutes_assigned": total_wait / assigned_count if assigned_count else math.nan,
        "weighted_wait_minutes": weighted_wait,
        "weighted_network_latency_ms": weighted_latency,
        "peak_gpu_utilization": max(utilizations) if utilizations else math.nan,
        "gpu_utilization_std": float(statistics.pstdev(utilizations)) if utilizations else math.nan,
    }


def make_stress_tasks(base_tasks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    hourly = base_tasks.groupby("ArrivalHour").GPU_Demand.sum().sort_values(ascending=False)
    peak_hours = sorted(int(hour) for hour in hourly.head(6).index)
    peak_pool = base_tasks.loc[base_tasks.ArrivalHour.isin(peak_hours)].copy()
    additional_count = int(math.floor(len(peak_pool) * 0.20 + 0.5))
    rng = np.random.default_rng(SEED)
    chosen_index = rng.choice(peak_pool.index.to_numpy(), size=additional_count, replace=False)
    duplicates = peak_pool.loc[chosen_index].copy().reset_index(drop=True)
    first_synthetic_id = int(base_tasks.TaskID.max()) + 1
    duplicates["TaskID"] = np.arange(first_synthetic_id, first_synthetic_id + len(duplicates), dtype=int)
    stress = pd.concat([base_tasks, duplicates], ignore_index=True)
    base_gpuh = float((base_tasks.GPU_Demand * base_tasks.EstimatedDuration_min / 60).sum())
    duplicate_gpuh = float((duplicates.GPU_Demand * duplicates.EstimatedDuration_min / 60).sum())
    return stress, {
        "peak_hours": peak_hours,
        "base_peak_tasks": int(len(peak_pool)),
        "additional_tasks": int(len(duplicates)),
        "peak_task_count_increase": len(duplicates) / len(peak_pool) if len(peak_pool) else math.nan,
        "overall_task_count_increase": len(duplicates) / len(base_tasks) if len(base_tasks) else math.nan,
        "overall_gpuh_increase": duplicate_gpuh / base_gpuh if base_gpuh else math.nan,
        "selection_seed": SEED,
    }


def build_schedule_outputs() -> dict:
    base_inputs = load_schedule_inputs()
    base_start = time.perf_counter()
    base_assignments, base_candidates = deterministic_schedule(base_inputs)
    base_runtime = time.perf_counter() - base_start
    base_metrics = validate_schedule(base_assignments, base_inputs)
    base_assignments.to_csv(
        TABLE_DIR / "schedule_assignments_base.csv", index=False, encoding="utf-8-sig"
    )

    stress_tasks, stress_definition = make_stress_tasks(base_inputs["tasks"])
    stress_inputs = load_schedule_inputs(tasks_override=stress_tasks)
    stress_start = time.perf_counter()
    stress_assignments, stress_candidates = deterministic_schedule(stress_inputs)
    stress_runtime = time.perf_counter() - stress_start
    stress_metrics = validate_schedule(stress_assignments, stress_inputs)
    stress_assignments.to_csv(
        TABLE_DIR / "schedule_assignments_stress.csv", index=False, encoding="utf-8-sig"
    )

    schedule_rows = []
    for scenario, metrics, _runtime in [
        ("base_actual_2376_2399", base_metrics, base_runtime),
        ("peak_tasks_plus_20pct", stress_metrics, stress_runtime),
    ]:
        schedule_rows.append(
            {
                "scenario": scenario,
                "total_tasks": metrics["total_tasks"],
                "assigned_tasks": metrics["assigned_tasks"],
                "task_completion_rate": metrics["task_completion_rate"],
                "hard_constraint_violation_count": metrics["hard_constraint_violation_count"],
                "migrations": metrics["migrations"],
                "migration_rate": metrics["migration_rate"],
                "total_wait_minutes": metrics["total_wait_minutes"],
                "average_wait_minutes_assigned": metrics["average_wait_minutes_assigned"],
                "weighted_wait_minutes": metrics["weighted_wait_minutes"],
                "peak_gpu_utilization": metrics["peak_gpu_utilization"],
            }
        )
    pd.DataFrame(schedule_rows).to_csv(
        TABLE_DIR / "schedule_results.csv", index=False, encoding="utf-8-sig"
    )
    changes = {
        "completion_rate_change": stress_metrics["task_completion_rate"] - base_metrics["task_completion_rate"],
        "hard_constraint_violation_change": stress_metrics["hard_constraint_violation_count"] - base_metrics["hard_constraint_violation_count"],
        "migration_change": stress_metrics["migrations"] - base_metrics["migrations"],
        "total_wait_minutes_change": stress_metrics["total_wait_minutes"] - base_metrics["total_wait_minutes"],
        "average_wait_minutes_change": stress_metrics["average_wait_minutes_assigned"] - base_metrics["average_wait_minutes_assigned"],
    }
    payload = {
        "base": {
            "runtime_seconds": base_runtime,
            "candidate_diagnostics": base_candidates,
            "metrics": base_metrics,
        },
        "stress": {
            "definition": stress_definition,
            "runtime_seconds": stress_runtime,
            "candidate_diagnostics": stress_candidates,
            "metrics": stress_metrics,
            "changes_vs_base": changes,
        },
        "prediction_values_used": False,
        "cpsat_executed": False,
        "cpsat_role": "仅在规则调度不可行或未来出现复杂约束时回退",
    }
    write_json(METRIC_DIR / "scheduling_metrics.json", payload)
    return payload


def environment_payload() -> dict:
    packages = ["numpy", "pandas", "lightgbm", "openpyxl"]
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "random_seed": SEED,
        "packages": {name: metadata.version(name) for name in packages},
        "model_params": MODEL_PARAMS,
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in [
                ROOT / "02_数据处理" / "processed" / "q1_hourly_gpu_demand.csv",
                ROOT / "02_数据处理" / "raw" / "workload_trace.xlsx",
                ROOT / "02_数据处理" / "raw" / "GPU_information.xlsx",
                ROOT / "02_数据处理" / "raw" / "network_latency.xlsx",
                ROOT / "02_数据处理" / "raw" / "power_mapping.xlsx",
                ROOT / "02_数据处理" / "raw" / "region_time_data.xlsx",
            ]
        },
        "run_command": "<python> 04_代码实现/question1/run_all.py",
    }


def run_legacy_round1_ablation() -> dict:
    prepare_output_dirs()
    set_reproducible_seed()
    overall_start = time.perf_counter()
    prediction_data = load_prediction_data()
    prediction_payload, _ = build_prediction_outputs(prediction_data)
    schedule_payload = build_schedule_outputs()
    reproducibility = environment_payload()
    write_json(METRIC_DIR / "reproducibility.json", reproducibility)

    validation_improvement = prediction_payload["official_validation"]["relative_improvement"]
    all_zero = (
        prediction_payload["output_degeneracy"]["all_zero_validation_series"]
        or prediction_payload["output_degeneracy"]["all_zero_test_series"]
    )
    base_metrics = schedule_payload["base"]["metrics"]
    fallback_observed = bool(
        validation_improvement < 0.05
        or all_zero
        or base_metrics["task_completion_rate"] < 1.0
        or base_metrics["hard_constraint_violation_count"] > 0
    )
    warning_messages: list[str] = []
    correction = prediction_payload["conservative_correction"]
    if not correction["adopted_as_conservative_companion"]:
        warning_messages.append("高负荷修正未同时满足整体 WAPE 恶化不超过 5% 和尾部风险下降，故仅保留诊断输出。")
    if any(not row["lightgbm_overall_better"] for row in prediction_payload["stability_windows"]):
        warning_messages.append("至少一个相邻稳定性窗口中 LightGBM 未优于最强简单基线；按人工决定如实披露，不扩展模型。")
    if schedule_payload["stress"]["metrics"]["task_completion_rate"] < 1.0:
        warning_messages.append("20% 高峰任务压力场景未完成全部任务，表明当前规则或容量存在承载边界。")

    output_files = sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROUND_DIR.rglob("*")
        if path.is_file() and path.name != "run_summary.json"
    )
    summary = {
        "schema_version": 1,
        "question": "Q1",
        "round": "round1",
        "implementation_target": "python",
        "random_seed": SEED,
        "approved_decision_id": "Q1-METHOD-CHOICE-001",
        "methods": [
            {
                "method_id": "M1",
                "role": "approved_main",
                "script": "04_代码实现/question1/q1_model.py",
                "status": "success",
                "input_files": list(reproducibility["input_sha256"]),
                "output_files": output_files,
                "metrics_summary": {
                    "validation_micro_wape": prediction_payload["official_validation"]["main"]["micro_wape"],
                    "validation_median_series_wape": prediction_payload["official_validation"]["main"]["median_series_wape"],
                    "test_micro_wape": prediction_payload["official_test"]["main"]["micro_wape"],
                    "base_schedule_completion_rate": base_metrics["task_completion_rate"],
                    "base_schedule_hard_constraint_violations": base_metrics["hard_constraint_violation_count"],
                },
                "warnings": warning_messages,
                "errors": [],
            },
            {
                "method_id": "B1",
                "role": "usable_baseline",
                "script": "04_代码实现/question1/q1_model.py",
                "status": "success",
                "input_files": ["02_数据处理/processed/q1_hourly_gpu_demand.csv"],
                "output_files": [
                    "05_结果与图表/intermediate/question1/experiments/round1/tables/prediction_comparison.csv"
                ],
                "metrics_summary": {
                    "validation_historical_mean_wape": prediction_payload["official_validation"]["baselines"]["historical_mean"]["micro_wape"],
                    "test_historical_mean_wape": prediction_payload["official_test"]["baselines"]["historical_mean"]["micro_wape"],
                },
                "warnings": [],
                "errors": [],
            },
        ],
        "comparison": {
            "validation_relative_wape_improvement_vs_best_simple_baseline": validation_improvement,
            "stability_windows": prediction_payload["stability_windows"],
        },
        "fallback_trigger": {
            "fallback_id": "F1",
            "condition": "验证改善低于 5%、全零序列、基础调度未完成全部任务或任一硬约束违规",
            "observed": fallback_observed,
            "evidence": "metrics/prediction_metrics.json; metrics/scheduling_metrics.json",
            "action": "按人工决定不继续开发 CP-SAT；如正式规则调度不可行，再单独申请激活回退。",
        },
        "execution_time_seconds": time.perf_counter() - overall_start,
        "output_files": output_files,
        "warnings": warning_messages,
        "errors": [],
        "environment": reproducibility,
    }
    write_json(ROUND_DIR / "run_summary.json", summary)
    return summary


if __name__ == "__main__":
    run_legacy_round1_ablation()
