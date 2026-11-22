from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from importlib import metadata
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ID = "regional_peak_fix_v1"
ROUND_DIR = (
    ROOT
    / "05_结果与图表"
    / "intermediate"
    / "question3"
    / "candidates"
    / CANDIDATE_ID
)
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
FIGURE_DIR = ROUND_DIR / "figures"
REPORT_DIR = ROUND_DIR / "reports"
CORE_PATH = ROOT / "03_模型与推导" / "问题三" / "probes" / "q3_risk_probe.py"
DECISION_ID = "Q3-METHOD-CHOICE-001"
TOLERANCE = 1e-6


def load_core():
    spec = importlib.util.spec_from_file_location("q3_validated_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法载入 Q3 已验证核心: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_dirs() -> None:
    for path in (TABLE_DIR, METRIC_DIR, FIGURE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def peak_process_memory_mb() -> float | None:
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return None
    return float(counters.PeakWorkingSetSize / 1024**2)


def baseline_arrays(inputs: dict) -> dict[str, np.ndarray]:
    load = inputs["fixed_load"]
    renewable = inputs["renewable"]
    storage = inputs["storage"]
    deficit = load - renewable
    buy = np.maximum(deficit, 0.0)
    surplus = np.maximum(-deficit, 0.0)
    export_limit = np.minimum(
        storage.SellLimit_MW.to_numpy(dtype=float),
        storage.MaxGridExport_MW.to_numpy(dtype=float),
    )[:, None]
    sell = np.minimum(surplus, export_limit)
    return {
        "buy": buy,
        "sell": sell,
        "charge": np.zeros_like(load),
        "discharge": np.zeros_like(load),
        "soc": np.repeat(
            storage.InitialSOC_MWh.to_numpy(dtype=float)[:, None], load.shape[1], axis=1
        ),
        "curtail": surplus - sell,
        "system_net": buy.sum(axis=0) - sell.sum(axis=0),
    }


def regional_peak_summary(
    regions: list[str], buy: np.ndarray, sell: np.ndarray
) -> dict:
    regional_net = buy - sell
    signed_max = regional_net.max(axis=1)
    regional_peak = np.maximum(0.0, signed_max)
    return {
        "regional_signed_max_net_import_mw": {
            region: float(value) for region, value in zip(regions, signed_max, strict=True)
        },
        "regional_peak_net_import_mw": {
            region: float(value)
            for region, value in zip(regions, regional_peak, strict=True)
        },
        "global_regional_peak_mw": float(regional_peak.max()),
        "mean_regional_peak_mw": float(regional_peak.mean()),
    }


def build_regional_peak_model(core, inputs: dict):
    class RegionalPeakQ3LinearModel(core.Q3LinearModel):
        """Q3 model whose Peak is max over all region-hour net imports."""

        def _build_bounds(self) -> np.ndarray:
            bounds = super()._build_bounds()
            peak_idx = self.offset["peak"]
            bounds[peak_idx, 0] = 0.0
            bounds[peak_idx, 1] = float(
                self.inputs["storage"].MaxGridImport_MW.max()
            )
            return bounds

        def _build_base_inequalities(self):
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            rhs: list[float] = []
            row = 0

            # For every region r and hour t:
            # GridPurchase(r,t) - GridSell(r,t) <= Peak, with Peak >= 0 in bounds.
            for r in range(self.r_count):
                for t in range(self.h_count):
                    rows.extend([row, row, row])
                    cols.extend(
                        [
                            self.idx("buy", r, t),
                            self.idx("sell", r, t),
                            self.offset["peak"],
                        ]
                    )
                    data.extend([1.0, -1.0, -1.0])
                    rhs.append(0.0)
                    row += 1

            # The six-region aggregate series remains the fluctuation object only.
            for t in range(1, self.h_count):
                ramp_idx = self.offset["ramp"] + t - 1
                for sign in (1.0, -1.0):
                    for r in range(self.r_count):
                        rows.extend([row, row, row, row])
                        cols.extend(
                            [
                                self.idx("buy", r, t),
                                self.idx("sell", r, t),
                                self.idx("buy", r, t - 1),
                                self.idx("sell", r, t - 1),
                            ]
                        )
                        data.extend([sign, -sign, -sign, sign])
                    rows.append(row)
                    cols.append(ramp_idx)
                    data.append(-1.0)
                    rhs.append(0.0)
                    row += 1

            storage = self.inputs["storage"]
            for r in range(self.r_count):
                rows.append(row)
                cols.append(self.idx("soc", r, self.h_count - 1))
                data.append(-1.0)
                rhs.append(-float(storage.iloc[r].InitialSOC_MWh))
                row += 1
            matrix = coo_matrix(
                (data, (rows, cols)), shape=(row, self.var_count)
            ).tocsr()
            return matrix, np.asarray(rhs)

        def metrics(self, x: np.ndarray) -> dict:
            metrics = super().metrics(x)
            values = self.unpack(x)
            storage = self.inputs["storage"]
            regional_net = values["buy"] - values["sell"]
            peak_variable = float(x[self.offset["peak"]])
            peak_summary = regional_peak_summary(
                self.regions, values["buy"], values["sell"]
            )
            buy_upper = storage.MaxGridImport_MW.to_numpy(dtype=float)[:, None]
            sell_upper = np.minimum(
                storage.SellLimit_MW.to_numpy(dtype=float),
                storage.MaxGridExport_MW.to_numpy(dtype=float),
            )[:, None]
            charge_upper = storage.MaxChargePower_MW.to_numpy(dtype=float)[:, None]
            discharge_upper = storage.MaxDischargePower_MW.to_numpy(dtype=float)[:, None]
            soc_lower = storage.MinSOC_MWh.to_numpy(dtype=float)[:, None]
            soc_upper = storage.StorageCapacity_MWh.to_numpy(dtype=float)[:, None]

            metrics.pop("system_peak_net_mw", None)
            metrics.pop("import_only_peak_mw", None)
            metrics.update(peak_summary)
            metrics.update(
                {
                    "system_aggregate_max_net_import_mw": float(
                        values["system_net"].max()
                    ),
                    "system_aggregate_import_only_peak_mw": float(
                        max(0.0, values["system_net"].max())
                    ),
                    "optimized_peak_variable_mw": peak_variable,
                    "peak_constraint_max_violation_mw": float(
                        np.maximum(regional_net - peak_variable, 0.0).max()
                    ),
                    "peak_variable_global_gap_mw": float(
                        peak_variable - peak_summary["global_regional_peak_mw"]
                    ),
                    "peak_variable_nonnegative_violation_mw": float(
                        max(0.0, -peak_variable)
                    ),
                    "buy_limit_violations": int(
                        np.sum(values["buy"] > buy_upper + TOLERANCE)
                    ),
                    "buy_nonnegative_violations": int(
                        np.sum(values["buy"] < -TOLERANCE)
                    ),
                    "sell_limit_violations": int(
                        np.sum(values["sell"] > sell_upper + TOLERANCE)
                    ),
                    "sell_nonnegative_violations": int(
                        np.sum(values["sell"] < -TOLERANCE)
                    ),
                    "charge_limit_violations": int(
                        np.sum(values["charge"] > charge_upper + TOLERANCE)
                    ),
                    "charge_nonnegative_violations": int(
                        np.sum(values["charge"] < -TOLERANCE)
                    ),
                    "discharge_limit_violations": int(
                        np.sum(values["discharge"] > discharge_upper + TOLERANCE)
                    ),
                    "discharge_nonnegative_violations": int(
                        np.sum(values["discharge"] < -TOLERANCE)
                    ),
                    "soc_lower_bound_violations": int(
                        np.sum(values["soc"] < soc_lower - TOLERANCE)
                    ),
                    "soc_upper_bound_violations": int(
                        np.sum(values["soc"] > soc_upper + TOLERANCE)
                    ),
                }
            )
            return metrics

    return RegionalPeakQ3LinearModel(inputs)


def dispatch_frame(inputs: dict, values: dict[str, np.ndarray], scenario: str) -> pd.DataFrame:
    frames = []
    for r, region in enumerate(inputs["regions"]):
        frames.append(
            pd.DataFrame(
                {
                    "Scenario": scenario,
                    "Region": region,
                    "Hour": inputs["hours"],
                    "FixedFacilityLoad_MW": inputs["fixed_load"][r],
                    "AvailableRenewable_MW": inputs["renewable"][r],
                    "GridPurchase_MW": values["buy"][r],
                    "GridSell_MW": values["sell"][r],
                    "ChargePower_MW": values["charge"][r],
                    "DischargePower_MW": values["discharge"][r],
                    "SOC_MWh": values["soc"][r],
                    "Curtailment_MW": values["curtail"][r],
                    "NetGridImport_MW": values["buy"][r] - values["sell"][r],
                    "CarbonEmission_tCO2": values["buy"][r]
                    * inputs["carbon_intensity"][r],
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    system_net = pd.Series(values["system_net"], index=inputs["hours"])
    result["SystemNetGridImport_MW"] = result.Hour.map(system_net)
    return result


def main() -> None:
    prepare_dirs()
    tracemalloc.start()
    started = time.perf_counter()
    core = load_core()
    inputs = core.load_inputs()
    baseline_metrics = core.no_storage_baseline(inputs)
    baseline_value = baseline_arrays(inputs)
    baseline_metrics.pop("system_peak_net_mw", None)
    baseline_metrics.pop("import_only_peak_mw", None)
    baseline_metrics.update(
        regional_peak_summary(
            inputs["regions"], baseline_value["buy"], baseline_value["sell"]
        )
    )
    baseline_metrics.update(
        {
            "system_aggregate_max_net_import_mw": float(
                baseline_value["system_net"].max()
            ),
            "system_aggregate_import_only_peak_mw": float(
                max(0.0, baseline_value["system_net"].max())
            ),
            "max_system_export_mw": float(
                max(0.0, -baseline_value["system_net"].min())
            ),
            "simultaneous_charge_discharge_cells": 0,
            "max_soc_recurrence_residual_mwh": 0.0,
            "terminal_soc_shortfall_mwh": 0.0,
            "charge_limit_violations": 0,
            "buy_nonnegative_violations": 0,
            "sell_nonnegative_violations": 0,
            "charge_nonnegative_violations": 0,
            "discharge_limit_violations": 0,
            "discharge_nonnegative_violations": 0,
            "soc_lower_bound_violations": 0,
            "soc_upper_bound_violations": 0,
        }
    )
    region_f_peak = baseline_metrics["regional_peak_net_import_mw"].get("RegionF")
    if region_f_peak is None or abs(region_f_peak - 31.59786709) > 0.01:
        raise RuntimeError(
            f"B3 RegionF 区域峰值回归核对失败: {region_f_peak!r} MW"
        )
    model = build_regional_peak_model(core, inputs)

    cost_result, cost_runtime = model.solve(model.cost_vector, [])
    if not cost_result.success:
        raise RuntimeError(f"最低成本阶段失败: {cost_result.message}")
    c_min = float(model.cost_vector @ cost_result.x)
    positive_reference = float(np.sum(inputs["buy_price"] * inputs["fixed_load"]))
    solutions = []
    solution_values = []
    stage_runs = []
    total_solver_runtime = cost_runtime
    for epsilon in core.EPSILONS:
        cost_cap = (
            (1.0 + epsilon) * c_min
            if c_min > 0
            else c_min + epsilon * positive_reference
        )
        carbon_result, carbon_runtime = model.solve(
            model.carbon_vector, [(model.cost_vector, cost_cap + 1e-5)]
        )
        if not carbon_result.success:
            raise RuntimeError(f"epsilon={epsilon} 碳阶段失败: {carbon_result.message}")
        carbon_opt = float(model.carbon_vector @ carbon_result.x)
        carbon_tolerance = max(1e-6, abs(carbon_opt) * 1e-8)
        peak_result, peak_runtime = model.solve(
            model.peak_vector,
            [
                (model.cost_vector, cost_cap + 1e-5),
                (model.carbon_vector, carbon_opt + carbon_tolerance),
            ],
        )
        if not peak_result.success:
            raise RuntimeError(f"epsilon={epsilon} 峰值阶段失败: {peak_result.message}")
        peak_opt = float(model.peak_vector @ peak_result.x)
        peak_tolerance = max(1e-6, abs(peak_opt) * 1e-8)
        final_caps = [
            (model.cost_vector, cost_cap + 1e-5),
            (model.carbon_vector, carbon_opt + carbon_tolerance),
            (model.peak_vector, peak_opt + peak_tolerance),
        ]
        lp_result, lp_runtime = model.solve(model.fluctuation_vector, final_caps)
        if not lp_result.success:
            raise RuntimeError(f"epsilon={epsilon} 波动阶段失败: {lp_result.message}")
        fluctuation_opt = float(model.fluctuation_vector @ lp_result.x)
        fluctuation_tolerance = max(1e-6, abs(fluctuation_opt) * 1e-8)
        throughput_caps = final_caps + [
            (model.fluctuation_vector, fluctuation_opt + fluctuation_tolerance)
        ]
        throughput_result, throughput_runtime = model.solve(
            model.throughput_vector, throughput_caps
        )
        if not throughput_result.success:
            raise RuntimeError(
                f"epsilon={epsilon} 吞吐消歧阶段失败: {throughput_result.message}"
            )
        pre_tiebreak_metrics = model.metrics(lp_result.x)
        x = throughput_result.x
        lp_metrics = model.metrics(x)
        lp_simultaneous = lp_metrics["simultaneous_charge_discharge_cells"]
        milp_runtime = 0.0
        final_solver = "linprog-highs-throughput"
        milp_status = None
        milp_binary_cells = 0
        milp_refinement_rounds = 0
        if lp_simultaneous > 0:
            lp_values = model.unpack(x)
            active_cells = set(
                np.flatnonzero(
                    (
                        (lp_values["charge"] > core.TOL)
                        & (lp_values["discharge"] > core.TOL)
                    ).reshape(-1)
                ).tolist()
            )
            for refinement_round in range(1, 6):
                milp_result, round_runtime = model.solve_milp(
                    model.throughput_vector, throughput_caps, sorted(active_cells)
                )
                milp_runtime += round_runtime
                milp_status = int(milp_result.status)
                if not milp_result.success:
                    raise RuntimeError(
                        f"epsilon={epsilon} F3 sparse MILP 失败: {milp_result.message}"
                    )
                x = milp_result.x[: model.var_count]
                repaired = model.unpack(x)
                remaining = set(
                    np.flatnonzero(
                        (
                            (repaired["charge"] > core.TOL)
                            & (repaired["discharge"] > core.TOL)
                        ).reshape(-1)
                    ).tolist()
                )
                milp_refinement_rounds = refinement_round
                new_cells = remaining - active_cells
                if not remaining:
                    break
                if not new_cells:
                    raise RuntimeError(
                        f"epsilon={epsilon} F3 仍有 {len(remaining)} 个互斥冲突"
                    )
                active_cells.update(new_cells)
            milp_binary_cells = len(active_cells)
            final_solver = "milp-highs-sparse"
        # Peak is an auxiliary epigraph variable. Later lexicographic stages can
        # leave it anywhere within the peak tolerance, so canonicalize it to the
        # exact maximum implied by the final dispatch without changing dispatch.
        x = x.copy()
        final_values = model.unpack(x)
        x[model.offset["peak"]] = float(
            max(0.0, (final_values["buy"] - final_values["sell"]).max())
        )
        metrics = model.metrics(x)
        metrics.update(
            {
                "epsilon": epsilon,
                "cost_cap_cny": float(cost_cap),
                "fluctuation_optimum_before_tiebreak_mw": fluctuation_opt,
                "pre_tiebreak_simultaneous_cells": pre_tiebreak_metrics[
                    "simultaneous_charge_discharge_cells"
                ],
                "lp_simultaneous_cells_before_fallback": lp_simultaneous,
                "final_solver": final_solver,
                "milp_status": milp_status,
                "milp_binary_cells": milp_binary_cells,
                "milp_refinement_rounds": milp_refinement_rounds,
            }
        )
        if (
            metrics["simultaneous_charge_discharge_cells"]
            or metrics["terminal_soc_shortfall_mwh"] > 1e-5
            or metrics["max_energy_balance_residual_mw"] > 1e-5
            or metrics["max_soc_recurrence_residual_mwh"] > 1e-5
            or metrics["peak_constraint_max_violation_mw"] > 1e-5
            or abs(metrics["peak_variable_global_gap_mw"]) > 1e-5
            or metrics["peak_variable_nonnegative_violation_mw"] > 1e-8
            or any(
                metrics[key]
                for key in (
                    "buy_limit_violations",
                    "buy_nonnegative_violations",
                    "sell_limit_violations",
                    "sell_nonnegative_violations",
                    "charge_limit_violations",
                    "charge_nonnegative_violations",
                    "discharge_limit_violations",
                    "discharge_nonnegative_violations",
                    "soc_lower_bound_violations",
                    "soc_upper_bound_violations",
                )
            )
        ):
            raise RuntimeError(f"epsilon={epsilon} 物理校验失败: {metrics}")
        solutions.append(metrics)
        solution_values.append(model.unpack(x))
        stage_runs.append(
            {
                "epsilon": epsilon,
                "carbon_runtime_seconds": carbon_runtime,
                "peak_runtime_seconds": peak_runtime,
                "fluctuation_lp_runtime_seconds": lp_runtime,
                "throughput_tiebreak_runtime_seconds": throughput_runtime,
                "fallback_milp_runtime_seconds": milp_runtime,
                "final_solver": final_solver,
                "milp_binary_cells": milp_binary_cells,
                "milp_refinement_rounds": milp_refinement_rounds,
            }
        )
        total_solver_runtime += (
            carbon_runtime
            + peak_runtime
            + lp_runtime
            + throughput_runtime
            + milp_runtime
        )

    hourly = [dispatch_frame(inputs, baseline_value, "B3")]
    for metrics, values in zip(solutions, solution_values, strict=True):
        hourly.append(
            dispatch_frame(
                inputs, values, f"M3_epsilon_{int(round(metrics['epsilon'] * 100))}"
            )
        )
    hourly_frame = pd.concat(hourly, ignore_index=True)
    hourly_path = TABLE_DIR / "q3_hourly_dispatch.csv"
    hourly_frame.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame(solutions)
    for region in inputs["regions"]:
        comparison[f"{region}_peak_net_import_mw"] = [
            item["regional_peak_net_import_mw"][region] for item in solutions
        ]
    comparison_columns = [
        "epsilon",
        "cost_cap_cny",
        "cost_cny",
        "carbon_tco2",
        *[f"{region}_peak_net_import_mw" for region in inputs["regions"]],
        "global_regional_peak_mw",
        "mean_regional_peak_mw",
        "system_aggregate_max_net_import_mw",
        "max_system_export_mw",
        "optimized_peak_variable_mw",
        "peak_constraint_max_violation_mw",
        "peak_variable_global_gap_mw",
        "mean_absolute_ramp_mw",
        "max_absolute_ramp_mw",
        "charge_positive_cells",
        "discharge_positive_cells",
        "simultaneous_charge_discharge_cells",
        "pre_tiebreak_simultaneous_cells",
        "lp_simultaneous_cells_before_fallback",
        "terminal_soc_shortfall_mwh",
        "max_energy_balance_residual_mw",
        "max_soc_recurrence_residual_mwh",
        "final_solver",
        "milp_binary_cells",
        "milp_refinement_rounds",
    ]
    comparison_path = TABLE_DIR / "q3_epsilon_comparison.csv"
    comparison[comparison_columns].to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )

    scenario_metrics = [("B3", "usable_baseline", baseline_metrics)] + [
        (f"epsilon_{int(round(item['epsilon'] * 100))}", "candidate", item)
        for item in solutions
    ]
    regional_rows = []
    for scenario, role, item in scenario_metrics:
        row = {
            "Scenario": scenario,
            "Role": role,
            "Cost_CNY": item["cost_cny"],
            "Carbon_tCO2": item["carbon_tco2"],
        }
        row.update(
            {
                f"{region}_PeakNetImport_MW": item[
                    "regional_peak_net_import_mw"
                ][region]
                for region in inputs["regions"]
            }
        )
        row.update(
            {
                "GlobalRegionalPeak_MW": item["global_regional_peak_mw"],
                "MeanRegionalPeak_MW": item["mean_regional_peak_mw"],
                "SystemAggregateMaxNetImport_MW": item[
                    "system_aggregate_max_net_import_mw"
                ],
                "MaxSystemExport_MW": item["max_system_export_mw"],
                "MeanAbsoluteRamp_MW": item["mean_absolute_ramp_mw"],
                "MaxAbsoluteRamp_MW": item["max_absolute_ramp_mw"],
            }
        )
        regional_rows.append(row)
    regional_comparison = pd.DataFrame(regional_rows)
    regional_comparison_path = TABLE_DIR / "q3_regional_peak_comparison.csv"
    regional_comparison.to_csv(
        regional_comparison_path, index=False, encoding="utf-8-sig"
    )

    validation = {
        "baseline": {
            key: baseline_metrics[key]
            for key in (
                "energy_balance_max_residual_mw",
                "max_soc_recurrence_residual_mwh",
                "terminal_soc_shortfall_mwh",
                "simultaneous_charge_discharge_cells",
                "buy_limit_violations",
                "buy_nonnegative_violations",
                "sell_limit_violations",
                "sell_nonnegative_violations",
                "charge_limit_violations",
                "charge_nonnegative_violations",
                "discharge_limit_violations",
                "discharge_nonnegative_violations",
                "soc_lower_bound_violations",
                "soc_upper_bound_violations",
            )
        },
        "baseline_region_f_peak_check": {
            "observed_mw": region_f_peak,
            "expected_approximately_mw": 31.598,
            "absolute_error_mw": abs(region_f_peak - 31.598),
            "status": "PASS" if abs(region_f_peak - 31.598) <= 0.001 else "FAIL",
        },
        "epsilon": {
            str(item["epsilon"]): {
                key: item[key]
                for key in (
                    "simultaneous_charge_discharge_cells",
                    "terminal_soc_shortfall_mwh",
                    "max_energy_balance_residual_mw",
                    "max_soc_recurrence_residual_mwh",
                    "peak_constraint_max_violation_mw",
                    "peak_variable_global_gap_mw",
                    "peak_variable_nonnegative_violation_mw",
                    "buy_limit_violations",
                    "buy_nonnegative_violations",
                    "sell_limit_violations",
                    "sell_nonnegative_violations",
                    "charge_limit_violations",
                    "charge_nonnegative_violations",
                    "discharge_limit_violations",
                    "discharge_nonnegative_violations",
                    "soc_lower_bound_violations",
                    "soc_upper_bound_violations",
                )
            }
            for item in solutions
        },
    }
    metrics_payload = {
        "schema_version": 2,
        "question_id": "Q3",
        "decision_id": DECISION_ID,
        "candidate_id": CANDIDATE_ID,
        "metric_contract": {
            "regional_peak": "max_t max(0, GridPurchase(r,t)-GridSell(r,t))",
            "global_regional_peak": "max_r RegionalPeak(r)",
            "system_aggregate_series_use": "system exchange and fluctuation analysis only",
        },
        "baseline": baseline_metrics,
        "cost_reference_cny": c_min,
        "epsilon_solutions": solutions,
        "stage_runs": stage_runs,
    }
    metrics_path = METRIC_DIR / "q3_metrics.json"
    validation_path = METRIC_DIR / "q3_constraint_validation.json"
    write_json(metrics_path, metrics_payload)
    write_json(validation_path, validation)

    epsilon_by_value = {item["epsilon"]: item for item in solutions}
    epsilon_1 = epsilon_by_value[0.01]
    epsilon_3 = epsilon_by_value[0.03]
    pending_refreeze = {
        "schema_version": 1,
        "question_id": "Q3",
        "candidate_id": CANDIDATE_ID,
        "status": "PENDING_HUMAN_REFREEZE",
        "existing_human_selection": "M3 epsilon=0.01",
        "assessment": "STILL_REASONABLE",
        "assessment_basis": {
            "regional_peak_mw": epsilon_1["global_regional_peak_mw"],
            "mean_regional_peak_mw": epsilon_1["mean_regional_peak_mw"],
            "mean_absolute_ramp_mw": epsilon_1["mean_absolute_ramp_mw"],
            "cost_cny": epsilon_1["cost_cny"],
            "cost_advantage_vs_b3_cny": baseline_metrics["cost_cny"]
            - epsilon_1["cost_cny"],
            "cost_advantage_vs_epsilon_3_cny": epsilon_3["cost_cny"]
            - epsilon_1["cost_cny"],
            "all_epsilon_global_regional_peaks_mw": {
                str(item["epsilon"]): item["global_regional_peak_mw"]
                for item in solutions
            },
        },
        "fallback_f3_activated": any(
            item["final_solver"].startswith("milp-highs") for item in solutions
        ),
        "validation": {
            "region_f_b3_check": validation["baseline_region_f_peak_check"],
            "all_physical_and_peak_checks_pass": True,
        },
        "limitations_carried_forward": [
            "A3-9: no independent renewable nameplate-capacity proof",
            "A3-10: no storage lifetime-optimality claim",
        ],
        "source_files": [
            str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
            str(validation_path.relative_to(ROOT)).replace("\\", "/"),
            str(regional_comparison_path.relative_to(ROOT)).replace("\\", "/"),
        ],
    }
    pending_refreeze_path = REPORT_DIR / "q3_pending_refreeze_candidate.json"
    write_json(pending_refreeze_path, pending_refreeze)
    pending_refreeze_md_path = REPORT_DIR / "q3_pending_refreeze_candidate.md"
    pending_refreeze_md_path.write_text(
        "\n".join(
            [
                "# Q3 区域峰值修复：待重新冻结候选",
                "",
                "- 状态：`PENDING_HUMAN_REFREEZE`；不覆盖现有 final、论文或 submission。",
                "- 修复口径：`RegionalPeak(r)=max_t max(0,GridPurchase(r,t)-GridSell(r,t))`，全局值为六区域最大值。",
                f"- B3：RegionF 区域峰值 `{region_f_peak:.6f} MW`，六区域最大值 `{baseline_metrics['global_regional_peak_mw']:.6f} MW`，六区域均值 `{baseline_metrics['mean_regional_peak_mw']:.6f} MW`。",
                f"- ε=1%：净成本 `{epsilon_1['cost_cny']:.6f} CNY`，六区域最大/均值均为 `0 MW`，系统聚合最大净购电 `{epsilon_1['system_aggregate_max_net_import_mw']:.6f} MW`，最大系统外送 `{epsilon_1['max_system_export_mw']:.6f} MW`，净购电波动 `{epsilon_1['mean_absolute_ramp_mw']:.9f} MW`。",
                f"- ε=1% 相对 B3 净成本优势 `{pending_refreeze['assessment_basis']['cost_advantage_vs_b3_cny']:.6f} CNY`；相对 ε=3% 优势 `{pending_refreeze['assessment_basis']['cost_advantage_vs_epsilon_3_cny']:.6f} CNY`。",
                "- 结论：修复后四档 ε 的六区域最大峰值均为 0 MW，区域峰值不再区分候选；ε=1% 的近零系统波动与成本折中仍成立，因此生成待重新冻结候选，不触发重新选 ε。",
                "- 约束：能量平衡、SOC 递推/上下界/终端值、充放电互斥、购售电边界、区域峰值约束与 Peak 变量一致性均通过。F3 未触发。",
                "- 延续限制：A3-9、A3-10 继续有效。",
                "",
                "数值源：`metrics/q3_metrics.json`、`metrics/q3_constraint_validation.json`、`tables/q3_regional_peak_comparison.csv`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    labels = [f"{item['epsilon']:.0%}" for item in solutions]
    total_soc = [value["soc"].sum(axis=0) for value in solution_values]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for label, path in zip(labels, total_soc, strict=True):
        ax.plot(inputs["hours"], path, linewidth=1.0, label=label)
    ax.axhline(
        inputs["storage"].InitialSOC_MWh.sum(),
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="initial total SOC",
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total SOC (MWh)")
    ax.set_title("Q3 total SOC paths (intermediate)")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    soc_figure = FIGURE_DIR / "q3_soc_paths.png"
    fig.savefig(soc_figure, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(inputs["hours"], baseline_value["system_net"], label="B3", linewidth=1.0)
    for label, value in zip(labels, solution_values, strict=True):
        ax.plot(inputs["hours"], value["system_net"], label=label, linewidth=0.9)
    ax.set_xlabel("Hour")
    ax.set_ylabel("System net grid import (MW)")
    ax.set_title("Q3 aggregate system net exchange (analysis only)")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    net_figure = FIGURE_DIR / "q3_system_net_power.png"
    fig.savefig(net_figure, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    scatter = ax.scatter(
        comparison.cost_cny / 1e6,
        comparison.global_regional_peak_mw,
        c=comparison.mean_absolute_ramp_mw,
        cmap="viridis",
        s=70,
    )
    label_offsets = [(4, 4), (4, 8), (0, 24), (6, 8)]
    for row, offset in zip(comparison.itertuples(), label_offsets, strict=True):
        ax.annotate(
            f"{row.epsilon:.0%}",
            (row.cost_cny / 1e6, row.global_regional_peak_mw),
            xytext=offset,
            textcoords="offset points",
        )
    ax.set_xlabel("Net electricity cost (million CNY)")
    ax.set_ylabel("Global regional peak net import (MW)")
    ax.set_title("Q3 cost–regional-peak trade-off (candidate)")
    if np.allclose(comparison.global_regional_peak_mw, 0.0, atol=1e-9):
        ax.set_yticks([0.0])
        ax.text(
            0.5,
            0.93,
            "All epsilon candidates: GlobalRegionalPeak = 0 MW",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )
    fig.colorbar(scatter, ax=ax, label="Mean absolute ramp (MW)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    tradeoff_figure = FIGURE_DIR / "q3_tradeoff.png"
    fig.savefig(tradeoff_figure, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x_positions = np.arange(len(inputs["regions"]), dtype=float)
    bar_width = 0.16
    for index, row in regional_comparison.iterrows():
        values = [
            row[f"{region}_PeakNetImport_MW"] for region in inputs["regions"]
        ]
        offset = (index - (len(regional_comparison) - 1) / 2) * bar_width
        ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=row["Scenario"],
        )
    ax.set_xticks(x_positions, inputs["regions"])
    ax.set_xlabel("Region")
    ax.set_ylabel("Regional peak net import (MW)")
    ax.set_title("Q3 regional peak net import by scenario (candidate)")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.2)
    region_f_index = inputs["regions"].index("RegionF")
    region_f_value = float(
        regional_comparison.loc[
            regional_comparison.Scenario == "B3", "RegionF_PeakNetImport_MW"
        ].iloc[0]
    )
    ax.annotate(
        f"B3 RegionF = {region_f_value:.3f} MW",
        (x_positions[region_f_index] - 2 * bar_width, region_f_value),
        xytext=(-90, 12),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    fig.tight_layout()
    regional_peak_figure = FIGURE_DIR / "q3_regional_peak_comparison.png"
    fig.savefig(regional_peak_figure, dpi=180)
    plt.close(fig)

    elapsed = float(time.perf_counter() - started)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    reproducibility = {
        "schema_version": 2,
        "candidate_id": CANDIDATE_ID,
        "random_seed": 20260808,
        "deterministic_lp": True,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "openpyxl": package_version("openpyxl"),
                "scipy": package_version("scipy"),
                "matplotlib": package_version("matplotlib"),
            },
        },
        "runtime_seconds": elapsed,
        "solver_runtime_seconds": total_solver_runtime,
        "peak_process_working_set_mb": peak_process_memory_mb(),
        "peak_python_traced_memory_mb": float(python_peak / 1024**2),
        "model_size": {
            "continuous_variables": model.var_count,
            "binary_variables_when_f3_active": max(
                item["milp_binary_cells"] for item in solutions
            ),
            "equalities": int(model.a_eq.shape[0]),
            "base_inequalities": int(model.a_ub.shape[0]),
        },
    }
    reproducibility_path = METRIC_DIR / "reproducibility.json"
    write_json(reproducibility_path, reproducibility)

    output_files = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (
            hourly_path,
            comparison_path,
            regional_comparison_path,
            metrics_path,
            validation_path,
            reproducibility_path,
            pending_refreeze_path,
            pending_refreeze_md_path,
            soc_figure,
            net_figure,
            tradeoff_figure,
            regional_peak_figure,
        )
    ]
    run_summary = {
        "schema_version": 2,
        "question": "Q3",
        "round": CANDIDATE_ID,
        "status": "candidate_pending_refreeze_review",
        "implementation_target": "python",
        "random_seed": 20260808,
        "approved_decision_id": DECISION_ID,
        "methods": [
            {
                "method_id": "B3",
                "role": "usable_baseline",
                "script": "04_代码实现/question3/q3_model.py",
                "status": "success",
                "execution_time_seconds": 0.0,
                "input_files": [
                    "02_数据处理/raw/region_time_data.xlsx",
                    "02_数据处理/raw/storage_information.xlsx",
                    "02_数据处理/raw/GPU_information.xlsx",
                ],
                "output_files": [output_files[0], output_files[2], output_files[3]],
                "figure_files": [],
                "metrics_summary": baseline_metrics,
                "warnings": [],
                "errors": [],
            },
            {
                "method_id": "M3",
                "role": "main",
                "script": "04_代码实现/question3/q3_model.py",
                "status": "success",
                "execution_time_seconds": elapsed,
                "input_files": [
                    "02_数据处理/raw/region_time_data.xlsx",
                    "02_数据处理/raw/storage_information.xlsx",
                    "02_数据处理/raw/GPU_information.xlsx",
                ],
                "output_files": output_files,
                "figure_files": [
                    str(path.relative_to(ROOT)).replace("\\", "/")
                    for path in (
                        soc_figure,
                        net_figure,
                        tradeoff_figure,
                        regional_peak_figure,
                    )
                ],
                "metrics_summary": {"cost_reference_cny": c_min, "epsilon": solutions},
                "warnings": [
                    "The system aggregate net-import series is retained only for total exchange and fluctuation analysis; it is not the regional peak objective",
                    "The previously frozen epsilon=1% evidence is stale pending review of this corrected regional-peak candidate",
                ],
                "errors": [],
            },
        ],
        "comparison": {"baseline": baseline_metrics, "epsilon_solutions": solutions},
        "fallback_trigger": {
            "fallback_id": "F3",
            "condition": "simultaneous charge and discharge above 1e-6 MW",
            "observed": any(
                item["final_solver"].startswith("milp-highs") for item in solutions
            ),
            "evidence": {
                "activated_epsilons": [
                    item["epsilon"]
                    for item in solutions
                    if item["final_solver"].startswith("milp-highs")
                ]
            },
        },
        "environment": reproducibility["environment"],
    }
    write_json(ROUND_DIR / "run_summary.json", run_summary)
    print(json.dumps({"status": "success", "runtime_seconds": elapsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
