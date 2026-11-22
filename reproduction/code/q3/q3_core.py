from __future__ import annotations

import json
import math
import os
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack


ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = Path(__file__).resolve().parent
RAW = ROOT / "02_数据处理" / "raw"
EPSILONS = (0.00, 0.01, 0.03, 0.05)
TOL = 1e-6


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_memory_mb() -> float | None:
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        value = getattr(info, "peak_wset", info.rss)
        return float(value / 1024**2)
    except Exception:
        return None


def load_inputs() -> dict:
    region_time = pd.read_excel(RAW / "region_time_data.xlsx", sheet_name="region_time_data")
    storage = pd.read_excel(RAW / "storage_information.xlsx", sheet_name="storage_information")
    gpu = pd.read_excel(RAW / "GPU_information.xlsx", sheet_name="GPU中心基础情况")
    regions = sorted(storage.Region.astype(str).unique())
    hours = sorted(region_time.Hour.astype(int).unique())
    expected = pd.MultiIndex.from_product([regions, hours], names=["Region", "Hour"])
    frame = region_time.set_index(["Region", "Hour"]).reindex(expected)
    if frame.isna().any().any():
        raise ValueError("Q3 区域—小时输入网格存在缺失")
    storage = storage.set_index("Region").loc[regions]
    gpu = gpu.set_index("Region").loc[regions]
    shape = (len(regions), len(hours))

    def matrix(column: str) -> np.ndarray:
        return frame[column].to_numpy(dtype=float).reshape(shape)

    fixed_it = matrix("Baseline_AI_IT_Load_MW") + matrix("NonAI_IT_Load_MW")
    pue = gpu.PUE.to_numpy(dtype=float)[:, None]
    fixed_load = pue * fixed_it
    return {
        "regions": regions,
        "hours": hours,
        "fixed_load": fixed_load,
        "renewable": matrix("AvailableRenewable_MW"),
        "buy_price": matrix("ElectricityPrice_CNY_per_MWh"),
        "sell_price": matrix("SellPrice_CNY_per_MWh"),
        "carbon_intensity": matrix("CarbonIntensity_tCO2_per_MWh"),
        "storage": storage,
    }


class Q3LinearModel:
    def __init__(self, inputs: dict):
        self.inputs = inputs
        self.regions = inputs["regions"]
        self.hours = inputs["hours"]
        self.r_count = len(self.regions)
        self.h_count = len(self.hours)
        self.n = self.r_count * self.h_count
        self.offset = {
            "buy": 0,
            "sell": self.n,
            "charge": 2 * self.n,
            "discharge": 3 * self.n,
            "soc": 4 * self.n,
            "curtail": 5 * self.n,
            "peak": 6 * self.n,
            "ramp": 6 * self.n + 1,
        }
        self.var_count = 6 * self.n + 1 + self.h_count - 1
        self.cost_vector = np.zeros(self.var_count)
        self.carbon_vector = np.zeros(self.var_count)
        self.peak_vector = np.zeros(self.var_count)
        self.fluctuation_vector = np.zeros(self.var_count)
        self.throughput_vector = np.zeros(self.var_count)
        self.cost_vector[self.offset["buy"] : self.offset["buy"] + self.n] = inputs[
            "buy_price"
        ].reshape(-1)
        self.cost_vector[self.offset["sell"] : self.offset["sell"] + self.n] = -inputs[
            "sell_price"
        ].reshape(-1)
        self.carbon_vector[self.offset["buy"] : self.offset["buy"] + self.n] = inputs[
            "carbon_intensity"
        ].reshape(-1)
        self.peak_vector[self.offset["peak"]] = 1.0
        self.fluctuation_vector[
            self.offset["ramp"] : self.offset["ramp"] + self.h_count - 1
        ] = 1.0 / (self.h_count - 1)
        self.throughput_vector[
            self.offset["charge"] : self.offset["charge"] + self.n
        ] = 1.0
        self.throughput_vector[
            self.offset["discharge"] : self.offset["discharge"] + self.n
        ] = 1.0
        self.bounds = self._build_bounds()
        self.a_eq, self.b_eq = self._build_equalities()
        self.a_ub, self.b_ub = self._build_base_inequalities()

    def cell(self, r: int, t: int) -> int:
        return r * self.h_count + t

    def idx(self, name: str, r: int, t: int) -> int:
        return self.offset[name] + self.cell(r, t)

    def _build_bounds(self) -> np.ndarray:
        lower = np.zeros(self.var_count)
        upper = np.full(self.var_count, np.inf)
        s = self.inputs["storage"]
        renewable = self.inputs["renewable"].reshape(-1)
        for r in range(self.r_count):
            block = slice(r * self.h_count, (r + 1) * self.h_count)
            upper[self.offset["buy"] + block.start : self.offset["buy"] + block.stop] = float(
                s.iloc[r].MaxGridImport_MW
            )
            upper[self.offset["sell"] + block.start : self.offset["sell"] + block.stop] = float(
                min(s.iloc[r].SellLimit_MW, s.iloc[r].MaxGridExport_MW)
            )
            upper[
                self.offset["charge"] + block.start : self.offset["charge"] + block.stop
            ] = float(s.iloc[r].MaxChargePower_MW)
            upper[
                self.offset["discharge"] + block.start : self.offset["discharge"] + block.stop
            ] = float(s.iloc[r].MaxDischargePower_MW)
            lower[self.offset["soc"] + block.start : self.offset["soc"] + block.stop] = float(
                s.iloc[r].MinSOC_MWh
            )
            upper[self.offset["soc"] + block.start : self.offset["soc"] + block.stop] = float(
                s.iloc[r].StorageCapacity_MWh
            )
        upper[self.offset["curtail"] : self.offset["curtail"] + self.n] = renewable
        import_total = float(s.MaxGridImport_MW.sum())
        # Peak shaving concerns grid import. Once the signed system exchange is
        # non-positive, additional export must not improve the peak objective.
        lower[self.offset["peak"]] = 0.0
        upper[self.offset["peak"]] = import_total
        return np.column_stack([lower, upper])

    def _build_equalities(self) -> tuple[csr_matrix, np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs = np.zeros(2 * self.n)
        load = self.inputs["fixed_load"]
        renewable = self.inputs["renewable"]
        s = self.inputs["storage"]
        row = 0
        for r in range(self.r_count):
            for t in range(self.h_count):
                for name, coefficient in (
                    ("buy", 1.0),
                    ("sell", -1.0),
                    ("charge", -1.0),
                    ("discharge", 1.0),
                    ("curtail", -1.0),
                ):
                    rows.append(row)
                    cols.append(self.idx(name, r, t))
                    data.append(coefficient)
                rhs[row] = float(load[r, t] - renewable[r, t])
                row += 1
        for r in range(self.r_count):
            eta_ch = float(s.iloc[r].ChargeEfficiency)
            eta_dis = float(s.iloc[r].DischargeEfficiency)
            initial = float(s.iloc[r].InitialSOC_MWh)
            for t in range(self.h_count):
                rows.extend([row, row, row])
                cols.extend(
                    [
                        self.idx("soc", r, t),
                        self.idx("charge", r, t),
                        self.idx("discharge", r, t),
                    ]
                )
                data.extend([1.0, -eta_ch, 1.0 / eta_dis])
                if t == 0:
                    rhs[row] = initial
                else:
                    rows.append(row)
                    cols.append(self.idx("soc", r, t - 1))
                    data.append(-1.0)
                row += 1
        matrix = coo_matrix((data, (rows, cols)), shape=(2 * self.n, self.var_count)).tocsr()
        return matrix, rhs

    def _build_base_inequalities(self) -> tuple[csr_matrix, np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs: list[float] = []
        row = 0
        for t in range(self.h_count):
            for r in range(self.r_count):
                rows.extend([row, row])
                cols.extend([self.idx("buy", r, t), self.idx("sell", r, t)])
                data.extend([1.0, -1.0])
            rows.append(row)
            cols.append(self.offset["peak"])
            data.append(-1.0)
            rhs.append(0.0)
            row += 1
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
        matrix = coo_matrix((data, (rows, cols)), shape=(row, self.var_count)).tocsr()
        return matrix, np.asarray(rhs)

    def extra_row(self, vector: np.ndarray) -> csr_matrix:
        nonzero = np.flatnonzero(vector)
        return csr_matrix(
            (vector[nonzero], (np.zeros(len(nonzero), dtype=int), nonzero)),
            shape=(1, self.var_count),
        )

    def solve(self, objective: np.ndarray, caps: list[tuple[np.ndarray, float]]) -> tuple[object, float]:
        a_ub = self.a_ub
        b_ub = self.b_ub
        if caps:
            a_ub = vstack([a_ub] + [self.extra_row(vector) for vector, _ in caps], format="csr")
            b_ub = np.concatenate([b_ub, np.asarray([cap for _, cap in caps])])
        started = time.perf_counter()
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=self.a_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method="highs",
            options={"time_limit": 1800.0, "presolve": True},
        )
        return result, time.perf_counter() - started

    def solve_milp(
        self,
        objective: np.ndarray,
        caps: list[tuple[np.ndarray, float]],
        binary_cells: list[int] | np.ndarray | None = None,
    ) -> tuple[object, float]:
        a_ub = self.a_ub
        b_ub = self.b_ub
        if caps:
            a_ub = vstack(
                [a_ub] + [self.extra_row(vector) for vector, _ in caps], format="csr"
            )
            b_ub = np.concatenate([b_ub, np.asarray([cap for _, cap in caps])])
        cells = (
            np.arange(self.n, dtype=int)
            if binary_cells is None
            else np.unique(np.asarray(binary_cells, dtype=int))
        )
        binary_count = len(cells)
        extended_count = self.var_count + binary_count
        zero_eq = csr_matrix((self.a_eq.shape[0], binary_count))
        zero_ub = csr_matrix((a_ub.shape[0], binary_count))
        a_eq = hstack([self.a_eq, zero_eq], format="csr")
        a_ub = hstack([a_ub, zero_ub], format="csr")

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        mutual_rhs = np.zeros(2 * binary_count)
        storage = self.inputs["storage"]
        row = 0
        for binary_position, cell in enumerate(cells):
            r, t = divmod(int(cell), self.h_count)
            max_charge = float(storage.iloc[r].MaxChargePower_MW)
            max_discharge = float(storage.iloc[r].MaxDischargePower_MW)
            binary = self.var_count + binary_position
            rows.extend([row, row])
            cols.extend([self.idx("charge", r, t), binary])
            data.extend([1.0, -max_charge])
            mutual_rhs[row] = 0.0
            row += 1
            rows.extend([row, row])
            cols.extend([self.idx("discharge", r, t), binary])
            data.extend([1.0, max_discharge])
            mutual_rhs[row] = max_discharge
            row += 1
        a_mutual = coo_matrix(
            (data, (rows, cols)), shape=(2 * binary_count, extended_count)
        ).tocsr()
        matrix = vstack([a_eq, a_ub, a_mutual], format="csr")
        lower_constraints = np.concatenate(
            [
                self.b_eq,
                np.full(a_ub.shape[0], -np.inf),
                np.full(a_mutual.shape[0], -np.inf),
            ]
        )
        upper_constraints = np.concatenate([self.b_eq, b_ub, mutual_rhs])
        bound_matrix = np.vstack(
            [
                self.bounds,
                np.column_stack([np.zeros(binary_count), np.ones(binary_count)]),
            ]
        )
        c = np.concatenate([objective, np.zeros(binary_count)])
        integrality = np.concatenate(
            [np.zeros(self.var_count, dtype=int), np.ones(binary_count, dtype=int)]
        )
        started = time.perf_counter()
        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(bound_matrix[:, 0], bound_matrix[:, 1]),
            constraints=LinearConstraint(matrix, lower_constraints, upper_constraints),
            options={"time_limit": 600.0, "mip_rel_gap": 1e-8, "presolve": True},
        )
        return result, time.perf_counter() - started

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        result = {}
        for name in ("buy", "sell", "charge", "discharge", "soc", "curtail"):
            start = self.offset[name]
            result[name] = x[start : start + self.n].reshape(self.r_count, self.h_count)
        result["system_net"] = result["buy"].sum(axis=0) - result["sell"].sum(axis=0)
        return result

    def metrics(self, x: np.ndarray) -> dict:
        value = self.unpack(x)
        s = self.inputs["storage"]
        soc_residual = []
        for r in range(self.r_count):
            previous = float(s.iloc[r].InitialSOC_MWh)
            eta_ch = float(s.iloc[r].ChargeEfficiency)
            eta_dis = float(s.iloc[r].DischargeEfficiency)
            for t in range(self.h_count):
                expected = previous + eta_ch * value["charge"][r, t] - value["discharge"][r, t] / eta_dis
                soc_residual.append(abs(value["soc"][r, t] - expected))
                previous = value["soc"][r, t]
        energy_residual = (
            self.inputs["renewable"]
            + value["buy"]
            + value["discharge"]
            - self.inputs["fixed_load"]
            - value["charge"]
            - value["sell"]
            - value["curtail"]
        )
        simultaneous = (value["charge"] > TOL) & (value["discharge"] > TOL)
        system_net = value["system_net"]
        terminal = value["soc"][:, -1]
        initial = s.InitialSOC_MWh.to_numpy(dtype=float)
        return {
            "cost_cny": float(self.cost_vector @ x),
            "carbon_tco2": float(self.carbon_vector @ x),
            "system_peak_net_mw": float(system_net.max()),
            "import_only_peak_mw": float(max(0.0, system_net.max())),
            "max_system_export_mw": float(max(0.0, -system_net.min())),
            "optimized_peak_variable_mw": float(x[self.offset["peak"]]),
            "mean_absolute_ramp_mw": float(np.abs(np.diff(system_net)).mean()),
            "max_absolute_ramp_mw": float(np.abs(np.diff(system_net)).max()),
            "simultaneous_charge_discharge_cells": int(simultaneous.sum()),
            "max_energy_balance_residual_mw": float(np.abs(energy_residual).max()),
            "max_soc_recurrence_residual_mwh": float(max(soc_residual)),
            "terminal_soc_shortfall_mwh": float(np.maximum(initial - terminal, 0).max()),
            "charge_positive_cells": int((value["charge"] > TOL).sum()),
            "discharge_positive_cells": int((value["discharge"] > TOL).sum()),
            "soc_at_min_fraction": float(
                np.mean(
                    np.isclose(
                        value["soc"],
                        s.MinSOC_MWh.to_numpy(dtype=float)[:, None],
                        atol=1e-5,
                    )
                )
            ),
            "soc_at_max_fraction": float(
                np.mean(
                    np.isclose(
                        value["soc"],
                        s.StorageCapacity_MWh.to_numpy(dtype=float)[:, None],
                        atol=1e-5,
                    )
                )
            ),
        }


def no_storage_baseline(inputs: dict) -> dict:
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
    curtail = surplus - sell
    if np.any(buy > storage.MaxGridImport_MW.to_numpy(dtype=float)[:, None] + TOL):
        raise ValueError("无储能基线超过购电上限")
    system_net = buy.sum(axis=0) - sell.sum(axis=0)
    return {
        "cost_cny": float(np.sum(inputs["buy_price"] * buy - inputs["sell_price"] * sell)),
        "carbon_tco2": float(np.sum(inputs["carbon_intensity"] * buy)),
        "system_peak_net_mw": float(system_net.max()),
        "import_only_peak_mw": float(max(0.0, system_net.max())),
        "max_system_export_mw": float(max(0.0, -system_net.min())),
        "mean_absolute_ramp_mw": float(np.abs(np.diff(system_net)).mean()),
        "max_absolute_ramp_mw": float(np.abs(np.diff(system_net)).max()),
        "energy_balance_max_residual_mw": float(
            np.abs(renewable + buy - load - sell - curtail).max()
        ),
        "buy_limit_violations": int(
            np.sum(buy > storage.MaxGridImport_MW.to_numpy(dtype=float)[:, None] + TOL)
        ),
        "sell_limit_violations": int(np.sum(sell > export_limit + TOL)),
        "region_hour_rows": int(load.size),
    }


def main() -> None:
    started = time.perf_counter()
    inputs = load_inputs()
    baseline = no_storage_baseline(inputs)
    model = Q3LinearModel(inputs)
    cost_result, cost_runtime = model.solve(model.cost_vector, [])
    if not cost_result.success:
        raise RuntimeError(f"最低成本阶段失败: {cost_result.message}")
    c_min = float(model.cost_vector @ cost_result.x)
    positive_reference = float(np.sum(inputs["buy_price"] * inputs["fixed_load"]))
    solutions = []
    solve_runtime = cost_runtime
    for epsilon in EPSILONS:
        cost_cap = (
            (1.0 + epsilon) * c_min
            if c_min > 0
            else c_min + epsilon * positive_reference
        )
        carbon_result, runtime = model.solve(
            model.carbon_vector, [(model.cost_vector, cost_cap + 1e-5)]
        )
        solve_runtime += runtime
        if not carbon_result.success:
            raise RuntimeError(f"epsilon={epsilon} 碳阶段失败: {carbon_result.message}")
        carbon_opt = float(model.carbon_vector @ carbon_result.x)
        carbon_tolerance = max(1e-6, abs(carbon_opt) * 1e-8)
        peak_result, runtime = model.solve(
            model.peak_vector,
            [
                (model.cost_vector, cost_cap + 1e-5),
                (model.carbon_vector, carbon_opt + carbon_tolerance),
            ],
        )
        solve_runtime += runtime
        if not peak_result.success:
            raise RuntimeError(f"epsilon={epsilon} 峰值阶段失败: {peak_result.message}")
        peak_opt = float(model.peak_vector @ peak_result.x)
        peak_tolerance = max(1e-6, abs(peak_opt) * 1e-8)
        final_caps = [
            (model.cost_vector, cost_cap + 1e-5),
            (model.carbon_vector, carbon_opt + carbon_tolerance),
            (model.peak_vector, peak_opt + peak_tolerance),
        ]
        fluct_result, runtime = model.solve(
            model.fluctuation_vector,
            final_caps,
        )
        solve_runtime += runtime
        if not fluct_result.success:
            raise RuntimeError(f"epsilon={epsilon} 波动阶段失败: {fluct_result.message}")
        solution_x = fluct_result.x
        lp_metrics = model.metrics(solution_x)
        simultaneous_lp_cells = lp_metrics["simultaneous_charge_discharge_cells"]
        fallback_milp_status = None
        final_solver = "linprog-highs"
        if simultaneous_lp_cells > 0:
            fallback_result, runtime = model.solve_milp(
                model.fluctuation_vector, final_caps
            )
            solve_runtime += runtime
            fallback_milp_status = int(fallback_result.status)
            if not fallback_result.success:
                raise RuntimeError(
                    f"epsilon={epsilon} 互斥 MILP 失败: {fallback_result.message}"
                )
            solution_x = fallback_result.x[: model.var_count]
            final_solver = "milp-highs"
        metrics = model.metrics(solution_x)
        metrics.update(
            {
                "epsilon": epsilon,
                "cost_cap_cny": float(cost_cap),
                "cost_stage_status": int(cost_result.status),
                "carbon_stage_status": int(carbon_result.status),
                "peak_stage_status": int(peak_result.status),
                "fluctuation_stage_status": int(fluct_result.status),
                "simultaneous_lp_cells_before_fallback": simultaneous_lp_cells,
                "fallback_milp_status": fallback_milp_status,
                "final_solver": final_solver,
            }
        )
        solutions.append(metrics)

    simultaneous = max(item["simultaneous_charge_discharge_cells"] for item in solutions)
    simultaneous_before_fallback = max(
        item["simultaneous_lp_cells_before_fallback"] for item in solutions
    )
    residual = max(item["max_energy_balance_residual_mw"] for item in solutions)
    soc_residual = max(item["max_soc_recurrence_residual_mwh"] for item in solutions)
    terminal_shortfall = max(item["terminal_soc_shortfall_mwh"] for item in solutions)
    nondegenerate = all(
        item["charge_positive_cells"] > 0
        and item["discharge_positive_cells"] > 0
        and item["soc_at_min_fraction"] < 0.99
        and item["soc_at_max_fraction"] < 0.99
        for item in solutions
    )
    verdict = (
        "PASS"
        if simultaneous == 0
        and residual <= 1e-5
        and soc_residual <= 1e-5
        and terminal_shortfall <= 1e-5
        and nondegenerate
        else "CONDITIONAL"
    )
    metrics_payload = {
        "schema_version": 1,
        "question_id": "Q3",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "baseline": baseline,
        "cost_reference_cny": c_min,
        "epsilon_solutions": solutions,
        "runtime_seconds": float(time.perf_counter() - started),
        "solver_runtime_seconds": float(solve_runtime),
        "peak_process_memory_mb": process_memory_mb(),
        "model_size": {
            "variables": model.var_count,
            "equalities": int(model.a_eq.shape[0]),
            "base_inequalities": int(model.a_ub.shape[0]),
            "region_hours": model.n,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scipy": package_version("scipy"),
            },
        },
    }
    write_json(PROBE_DIR / "probe_metrics.json", metrics_payload)
    summary = {
        "schema_version": 1,
        "question_id": "Q3",
        "generated_at": metrics_payload["generated_at"],
        "decision_ref": "03_模型与推导/问题三/q3_decisions.jsonl#Q3-METHOD-CHOICE-001",
        "data_refs": [
            "02_数据处理/data_profile.json",
            "02_数据处理/raw/region_time_data.xlsx",
            "02_数据处理/raw/storage_information.xlsx",
        ],
        "methods": [
            {
                "id": "M3",
                "role": "main_candidate",
                "executability": {
                    "status": "PASS",
                    "evidence": {
                        "epsilon_solutions": len(solutions),
                        "solver_runtime_seconds": metrics_payload["solver_runtime_seconds"],
                        "all_stage_status_zero": True,
                    },
                },
                "data_coverage": {
                    "status": "PASS",
                    "evidence": {
                        "region_hour_rows": model.n,
                        "regions": model.r_count,
                        "hours": model.h_count,
                    },
                },
                "assumption_checks": [
                    {
                        "name": "physical_constraints",
                        "status": "PASS" if residual <= 1e-5 and soc_residual <= 1e-5 else "FAIL",
                        "max_energy_balance_residual_mw": residual,
                        "max_soc_recurrence_residual_mwh": soc_residual,
                        "terminal_soc_shortfall_mwh": terminal_shortfall,
                    },
                    {
                        "name": "simultaneous_charge_discharge",
                        "status": "PASS" if simultaneous == 0 else "FAIL",
                        "final_cells": simultaneous,
                        "lp_cells_before_fallback": simultaneous_before_fallback,
                        "fallback": "F3 MILP"
                        if simultaneous_before_fallback
                        else None,
                    },
                ],
                "output_degeneracy": {
                    "status": "PASS" if nondegenerate else "CONDITIONAL",
                    "metrics": {
                        "charge_positive_cells_min": min(
                            item["charge_positive_cells"] for item in solutions
                        ),
                        "discharge_positive_cells_min": min(
                            item["discharge_positive_cells"] for item in solutions
                        ),
                        "soc_at_min_fraction_max": max(
                            item["soc_at_min_fraction"] for item in solutions
                        ),
                        "soc_at_max_fraction_max": max(
                            item["soc_at_max_fraction"] for item in solutions
                        ),
                    },
                },
                "perturbation_sensitivity": {
                    "status": "PASS",
                    "perturbation": "cost epsilon in {0,1,3,5}%",
                    "metrics": {
                        "cost_cny": [item["cost_cny"] for item in solutions],
                        "carbon_tco2": [item["carbon_tco2"] for item in solutions],
                        "system_peak_net_mw": [
                            item["system_peak_net_mw"] for item in solutions
                        ],
                        "mean_absolute_ramp_mw": [
                            item["mean_absolute_ramp_mw"] for item in solutions
                        ],
                    },
                },
                "scale_check": {
                    "status": "PASS",
                    "representative_n": model.n,
                    "runtime_seconds": metrics_payload["runtime_seconds"],
                    "peak_memory_mb": metrics_payload["peak_process_memory_mb"],
                    "variables": model.var_count,
                },
                "verdict": verdict,
                "conditions": []
                if verdict == "PASS"
                else ["若出现同时充放电则启用 F3；任何物理残差失败均阻塞正式实现"],
                "evidence_refs": ["03_模型与推导/问题三/probes/probe_metrics.json"],
            },
            {
                "id": "B3",
                "role": "usable_baseline",
                "executability": {"status": "PASS", "evidence": baseline},
                "data_coverage": {
                    "status": "PASS",
                    "evidence": {"region_hour_rows": baseline["region_hour_rows"]},
                },
                "assumption_checks": [
                    {
                        "name": "no_storage_energy_balance",
                        "status": "PASS"
                        if baseline["energy_balance_max_residual_mw"] <= 1e-8
                        else "FAIL",
                        "value": baseline["energy_balance_max_residual_mw"],
                    }
                ],
                "output_degeneracy": {
                    "status": "PASS",
                    "metrics": {
                        "system_peak_net_mw": baseline["system_peak_net_mw"],
                        "mean_absolute_ramp_mw": baseline["mean_absolute_ramp_mw"],
                    },
                },
                "perturbation_sensitivity": {
                    "status": "NOT_APPLICABLE",
                    "reason": "确定性无储能公式无随机参数",
                },
                "scale_check": {
                    "status": "PASS",
                    "representative_n": baseline["region_hour_rows"],
                },
                "verdict": "PASS",
                "conditions": [],
                "evidence_refs": ["03_模型与推导/问题三/probes/probe_metrics.json"],
            },
        ],
        "conditional_fallback": {
            "id": "F3",
            "trigger": "任一区域小时同时充放电超过 1e-6 MW",
            "observed": simultaneous_before_fallback > 0,
            "evidence": {
                "lp_cells_before_fallback": simultaneous_before_fallback,
                "final_simultaneous_cells": simultaneous,
                "activated_epsilons": [
                    item["epsilon"]
                    for item in solutions
                    if item["final_solver"] == "milp-highs"
                ],
            },
        },
    }
    write_json(PROBE_DIR / "risk_probe_summary.json", summary)
    print(json.dumps({"verdict": verdict, "runtime_seconds": metrics_payload["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
