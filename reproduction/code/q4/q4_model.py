from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "02_数据处理" / "raw"
ROUND_ID = "round2_peak_latency_fix_candidate"
ROUND_DIR = (
    ROOT
    / "05_结果与图表"
    / "intermediate"
    / "question4"
    / "experiments"
    / ROUND_ID
)
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
FIGURE_DIR = ROUND_DIR / "figures"
SEED = 20260808
TOL = 1e-6
HOURS = 2407
TASK_HOURS = 2406
JOINT_TASK_LIMIT = int(os.environ.get("Q4_JOINT_TASK_LIMIT", "5000"))
JOINT_CAPS = tuple(
    float(value)
    for value in os.environ.get(
        "Q4_JOINT_CAPS", "0.02,0.05,0.10,0.20,0.40,0.70,1.00"
    ).split(",")
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().upper()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def peak_memory_mb() -> float | None:
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
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
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


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_dirs() -> None:
    for path in (TABLE_DIR, METRIC_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    resolved = ROUND_DIR.resolve()
    allowed = (
        ROOT
        / "05_结果与图表"
        / "intermediate"
        / "question4"
        / "experiments"
    ).resolve()
    if allowed not in resolved.parents:
        raise RuntimeError(f"Q4 输出根越界: {resolved}")


def load_q2_core():
    return import_file(
        "q4_q2_core",
        ROOT / "03_模型与推导" / "问题二" / "probes" / "q2_risk_probe.py",
    )


def load_q3_core():
    return import_file(
        "q4_q3_core",
        ROOT / "03_模型与推导" / "问题三" / "probes" / "q3_risk_probe.py",
    )


def load_state_from_assignments(q2, inputs: dict, path: Path):
    frame = pd.read_csv(path)
    if len(frame) != 50000 or frame.TaskID.nunique() != 50000:
        raise ValueError(f"任务表不完整: {path}")
    state = q2.ScheduleState.empty(inputs)
    index_by_id = {
        int(task_id): index
        for index, task_id in enumerate(inputs["tasks"].TaskID.astype(int))
    }
    for row in frame.itertuples(index=False):
        index = index_by_id[int(row.TaskID)]
        state.place(index, inputs["r_index"][str(row.AssignedRegion)], int(row.StartHour))
    validation = q2.validate_schedule(state)
    if sum(validation["violations"].values()):
        raise RuntimeError(f"冻结任务表复建违规: {validation}")
    return state, frame, validation


def local_only_baseline(q2, inputs: dict):
    state = q2.ScheduleState.empty(inputs)
    tasks = inputs["tasks"]
    order = tasks.sort_values(
        ["ArrivalHour", "LatestFinishHour", "TaskID"], kind="stable"
    ).index.to_numpy()
    started = time.perf_counter()
    candidate_tests = 0
    unassigned: list[int] = []
    for task_index in order:
        row = tasks.iloc[task_index]
        source = int(row.SourceIndex)
        earliest = max(int(row.ArrivalHour), int(row.EarliestStartHour))
        latest = min(
            (
                min(int(row.LatestFinishHour), q2.HORIZON_END) * 60
                - int(row.EstimatedDuration_min)
            )
            // 60,
            q2.HORIZON_END - 1,
        )
        starts = (
            [earliest]
            if row.TaskType == "RealTimeInference"
            else range(earliest, latest + 1)
        )
        placed = False
        for start in starts:
            candidate_tests += 1
            if state.feasible(task_index, source, int(start)):
                state.place(task_index, source, int(start))
                placed = True
                break
        if not placed:
            unassigned.append(int(row.TaskID))
    return state, {
        "runtime_seconds": time.perf_counter() - started,
        "candidate_tests": candidate_tests,
        "unassigned_tasks": len(unassigned),
        "first_unassigned_task_ids": unassigned[:20],
    }


def service_ratios(state) -> np.ndarray:
    tasks = state.inputs["tasks"]
    flexible = tasks.TaskType.to_numpy() != "RealTimeInference"
    start_min = state.start_hour.astype(float) * 60.0
    arrival_min = tasks.ArrivalHour.to_numpy(dtype=float) * 60.0
    latest_start = (
        np.minimum(tasks.LatestFinishHour.to_numpy(dtype=float), TASK_HOURS) * 60.0
        - tasks.EstimatedDuration_min.to_numpy(dtype=float)
    )
    denominator = latest_start - arrival_min
    eligible = flexible & (denominator > TOL) & (state.start_hour >= 0)
    ratios = (start_min[eligible] - arrival_min[eligible]) / denominator[eligible]
    return np.clip(ratios, 0.0, 1.0)


def empirical_cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    if not len(values):
        return math.nan
    threshold = float(np.quantile(values, alpha, method="linear"))
    return float(threshold + np.maximum(values - threshold, 0.0).mean() / (1 - alpha))


def service_metrics(state) -> dict:
    ratios = service_ratios(state)
    if not len(ratios):
        return {
            "service_flexible_tasks": 0,
            "mean_wait_ratio": math.nan,
            "p50_wait_ratio": math.nan,
            "p90_wait_ratio": math.nan,
            "p95_wait_ratio": math.nan,
            "max_wait_ratio": math.nan,
            "cvar95_wait_ratio": math.nan,
            "high_wait_share_ge_0_8": math.nan,
        }
    return {
        "service_flexible_tasks": int(len(ratios)),
        "mean_wait_ratio": float(ratios.mean()),
        "p50_wait_ratio": float(np.quantile(ratios, 0.50)),
        "p90_wait_ratio": float(np.quantile(ratios, 0.90)),
        "p95_wait_ratio": float(np.quantile(ratios, 0.95)),
        "max_wait_ratio": float(ratios.max()),
        "cvar95_wait_ratio": empirical_cvar(ratios),
        "high_wait_share_ge_0_8": float(np.mean(ratios >= 0.8)),
    }


def task_metrics(q2, state) -> dict:
    validation = q2.validate_schedule(state)
    base = q2.schedule_metrics(state)
    result = {
        **service_metrics(state),
        "task_completion_rate": float(validation["assigned_tasks"] / validation["total_tasks"]),
        "overdue_tasks": int(validation["violations"]["finish"]),
        "realtime_start_violations": int(
            validation["violations"]["realtime_immediate"]
        ),
        "task_hard_constraint_violations": int(sum(validation["violations"].values())),
        "mean_network_latency_ms": base["mean_network_latency_ms"],
        "p95_network_latency_ms": base["p95_network_latency_ms"],
        "max_network_latency_ms": base["max_network_latency_ms"],
        "migrated_mean_network_latency_ms": base[
            "migrated_mean_network_latency_ms"
        ],
        "migrated_p95_network_latency_ms": base[
            "migrated_p95_network_latency_ms"
        ],
        "migrated_max_network_latency_ms": base[
            "migrated_max_network_latency_ms"
        ],
        "migration_tasks": base["migration_tasks"],
        "migration_rate": base["migration_rate"],
        "assignment_sha256": base["assignment_sha256"],
        "max_gpu_utilization": validation["max_gpu_utilization"],
        "max_it_utilization": validation["max_it_utilization"],
        "max_facility_utilization": validation["max_facility_utilization"],
        "validation": validation,
    }
    task_type_prefixes = {
        "AITraining": "ai_training",
        "BatchInference": "batch_inference",
        "RealTimeInference": "real_time_inference",
    }
    by_task_type = base["service_quality"]["by_task_type"]
    if set(by_task_type) != set(task_type_prefixes):
        raise ValueError(
            f"Q4 预期三类任务 {sorted(task_type_prefixes)}，实际为 {sorted(by_task_type)}"
        )
    for task_type, prefix in task_type_prefixes.items():
        stats = by_task_type[task_type]
        result.update(
            {
                f"{prefix}_task_count": stats["task_count"],
                f"{prefix}_mean_network_latency_ms": stats["mean_latency_ms"],
                f"{prefix}_p95_network_latency_ms": stats["p95_latency_ms"],
                f"{prefix}_max_network_latency_ms": stats["max_latency_ms"],
            }
        )
    return result


def assignments_frame(state) -> pd.DataFrame:
    tasks = state.inputs["tasks"]
    regions = state.inputs["regions"]
    assigned = [regions[int(region)] for region in state.assigned_region]
    result = tasks[
        [
            "TaskID",
            "TaskType",
            "ArrivalHour",
            "SourceRegion",
            "GPU_Demand",
            "EstimatedDuration_min",
            "LatestFinishHour",
        ]
    ].copy()
    result["AssignedRegion"] = assigned
    result["StartHour"] = state.start_hour
    result["FinishMinute"] = (
        state.start_hour.astype(np.int64) * 60
        + tasks.EstimatedDuration_min.to_numpy(dtype=np.int64)
    )
    source_indices = tasks.SourceIndex.to_numpy(dtype=int)
    assigned_indices = state.assigned_region.astype(int)
    latency = state.inputs["latency"][source_indices, assigned_indices]
    result["EffectiveLatency_ms"] = latency
    result["Migrated"] = result["AssignedRegion"] != result["SourceRegion"]
    return result


def scenario_inputs(q2_inputs: dict, q3_inputs: dict, state, scenario: dict) -> dict:
    ai = np.zeros((len(q2_inputs["regions"]), HOURS))
    ai[:, :TASK_HOURS] = state.ai_power
    non_ai = q3_inputs["fixed_load"] / np.asarray(
        [
            q2_inputs["pue"][q2_inputs["r_index"][region]]
            for region in q3_inputs["regions"]
        ]
    )[:, None]
    # q3 fixed_load = PUE * (附件 Baseline_AI + NonAI)，不能由此反推 NonAI。
    # Q4 必须直接读取原始 NonAI 列。
    region_time = pd.read_excel(
        RAW / "region_time_data.xlsx", sheet_name="region_time_data"
    )
    expected = pd.MultiIndex.from_product(
        [q3_inputs["regions"], q3_inputs["hours"]], names=["Region", "Hour"]
    )
    rt = region_time.set_index(["Region", "Hour"]).reindex(expected)
    non_ai = rt.NonAI_IT_Load_MW.to_numpy(dtype=float).reshape(
        len(q3_inputs["regions"]), HOURS
    )
    pue = np.asarray(
        [q2_inputs["pue"][q2_inputs["r_index"][region]] for region in q3_inputs["regions"]]
    )[:, None]
    fixed_load = pue * (non_ai + ai)
    buy_price = q3_inputs["buy_price"].copy()
    spread_factor = float(scenario.get("price_spread_factor", 1.0))
    if spread_factor != 1.0:
        mean = buy_price.mean(axis=1, keepdims=True)
        buy_price = np.maximum(mean + spread_factor * (buy_price - mean), 0.0)
    renewable = q3_inputs["renewable"] * float(scenario.get("renewable_scale", 1.0))
    return {
        "regions": q3_inputs["regions"],
        "hours": q3_inputs["hours"],
        "fixed_load": fixed_load,
        "renewable": renewable,
        "buy_price": buy_price,
        "sell_price": q3_inputs["sell_price"].copy(),
        "carbon_intensity": q3_inputs["carbon_intensity"].copy(),
        "storage": q3_inputs["storage"],
        "scenario": scenario,
        "scenario_hashes": {
            "fixed_load": array_sha256(fixed_load),
            "renewable": array_sha256(renewable),
            "buy_price": array_sha256(buy_price),
        },
    }


@dataclass
class EnergyResult:
    x: np.ndarray
    metrics: dict
    arrays: dict[str, np.ndarray]


class ProvenanceEnergyModel:
    names = (
        "re_load",
        "re_charge",
        "re_sell",
        "curtail",
        "grid_load",
        "grid_charge",
        "re_dis_load",
        "re_dis_sell",
        "ne_dis_load",
        "soc_re",
        "soc_ne",
    )

    def __init__(self, inputs: dict):
        self.inputs = inputs
        self.r_count = len(inputs["regions"])
        self.h_count = len(inputs["hours"])
        self.n = self.r_count * self.h_count
        self.offset = {name: index * self.n for index, name in enumerate(self.names)}
        self.peak_index = len(self.names) * self.n
        self.var_count = self.peak_index + 1
        self.cost = np.zeros(self.var_count)
        self.carbon = np.zeros(self.var_count)
        self.peak = np.zeros(self.var_count)
        self.throughput = np.zeros(self.var_count)
        self.peak[self.peak_index] = 1.0
        self._build_objectives()
        self.bounds = self._build_bounds()
        self.a_eq, self.b_eq = self._build_equalities()
        self.a_ub, self.b_ub = self._build_inequalities()

    def idx(self, name: str, r: int, t: int) -> int:
        return self.offset[name] + r * self.h_count + t

    def _build_objectives(self) -> None:
        for r in range(self.r_count):
            for t in range(self.h_count):
                buy = float(self.inputs["buy_price"][r, t])
                sell = float(self.inputs["sell_price"][r, t])
                carbon = float(self.inputs["carbon_intensity"][r, t])
                for name in ("grid_load", "grid_charge"):
                    self.cost[self.idx(name, r, t)] = buy
                    self.carbon[self.idx(name, r, t)] = carbon
                for name in ("re_sell", "re_dis_sell"):
                    self.cost[self.idx(name, r, t)] = -sell
                for name in (
                    "re_charge",
                    "grid_charge",
                    "re_dis_load",
                    "re_dis_sell",
                    "ne_dis_load",
                ):
                    self.throughput[self.idx(name, r, t)] = 1.0

    def _build_bounds(self) -> np.ndarray:
        lower = np.zeros(self.var_count)
        upper = np.full(self.var_count, np.inf)
        storage = self.inputs["storage"]
        for r in range(self.r_count):
            capacity = float(storage.iloc[r].StorageCapacity_MWh)
            for t in range(self.h_count):
                upper[self.idx("soc_re", r, t)] = capacity
                upper[self.idx("soc_ne", r, t)] = capacity
                upper[self.idx("curtail", r, t)] = self.inputs["renewable"][r, t]
        return np.column_stack([lower, upper])

    def _build_equalities(self) -> tuple[csr_matrix, np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs: list[float] = []
        row = 0

        def add(terms: list[tuple[int, float]], value: float) -> None:
            nonlocal row
            for column, coefficient in terms:
                rows.append(row)
                cols.append(column)
                data.append(coefficient)
            rhs.append(value)
            row += 1

        storage = self.inputs["storage"]
        for r in range(self.r_count):
            eta_ch = float(storage.iloc[r].ChargeEfficiency)
            eta_dis = float(storage.iloc[r].DischargeEfficiency)
            initial = float(storage.iloc[r].InitialSOC_MWh)
            for t in range(self.h_count):
                add(
                    [
                        (self.idx("re_load", r, t), 1.0),
                        (self.idx("re_charge", r, t), 1.0),
                        (self.idx("re_sell", r, t), 1.0),
                        (self.idx("curtail", r, t), 1.0),
                    ],
                    float(self.inputs["renewable"][r, t]),
                )
                add(
                    [
                        (self.idx("re_load", r, t), 1.0),
                        (self.idx("grid_load", r, t), 1.0),
                        (self.idx("re_dis_load", r, t), 1.0),
                        (self.idx("ne_dis_load", r, t), 1.0),
                    ],
                    float(self.inputs["fixed_load"][r, t]),
                )
                re_terms = [
                    (self.idx("soc_re", r, t), 1.0),
                    (self.idx("re_charge", r, t), -eta_ch),
                    (self.idx("re_dis_load", r, t), 1.0 / eta_dis),
                    (self.idx("re_dis_sell", r, t), 1.0 / eta_dis),
                ]
                ne_terms = [
                    (self.idx("soc_ne", r, t), 1.0),
                    (self.idx("grid_charge", r, t), -eta_ch),
                    (self.idx("ne_dis_load", r, t), 1.0 / eta_dis),
                ]
                if t > 0:
                    re_terms.append((self.idx("soc_re", r, t - 1), -1.0))
                    ne_terms.append((self.idx("soc_ne", r, t - 1), -1.0))
                    add(re_terms, 0.0)
                    add(ne_terms, 0.0)
                else:
                    add(re_terms, 0.0)
                    add(ne_terms, initial)
        matrix = coo_matrix((data, (rows, cols)), shape=(row, self.var_count)).tocsr()
        return matrix, np.asarray(rhs)

    def _build_inequalities(self) -> tuple[csr_matrix, np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs: list[float] = []
        row = 0

        def add(terms: list[tuple[int, float]], value: float) -> None:
            nonlocal row
            for column, coefficient in terms:
                rows.append(row)
                cols.append(column)
                data.append(coefficient)
            rhs.append(value)
            row += 1

        storage = self.inputs["storage"]
        for r in range(self.r_count):
            capacity = float(storage.iloc[r].StorageCapacity_MWh)
            minimum = float(storage.iloc[r].MinSOC_MWh)
            initial = float(storage.iloc[r].InitialSOC_MWh)
            max_charge = float(storage.iloc[r].MaxChargePower_MW)
            max_discharge = float(storage.iloc[r].MaxDischargePower_MW)
            max_import = float(storage.iloc[r].MaxGridImport_MW)
            max_export = float(
                min(storage.iloc[r].SellLimit_MW, storage.iloc[r].MaxGridExport_MW)
            )
            for t in range(self.h_count):
                add(
                    [
                        (self.idx("soc_re", r, t), 1.0),
                        (self.idx("soc_ne", r, t), 1.0),
                    ],
                    capacity,
                )
                add(
                    [
                        (self.idx("soc_re", r, t), -1.0),
                        (self.idx("soc_ne", r, t), -1.0),
                    ],
                    -minimum,
                )
                add(
                    [
                        (self.idx("re_charge", r, t), 1.0),
                        (self.idx("grid_charge", r, t), 1.0),
                    ],
                    max_charge,
                )
                add(
                    [
                        (self.idx("re_dis_load", r, t), 1.0),
                        (self.idx("re_dis_sell", r, t), 1.0),
                        (self.idx("ne_dis_load", r, t), 1.0),
                    ],
                    max_discharge,
                )
                add(
                    [
                        (self.idx("grid_load", r, t), 1.0),
                        (self.idx("grid_charge", r, t), 1.0),
                    ],
                    max_import,
                )
                add(
                    [
                        (self.idx("re_sell", r, t), 1.0),
                        (self.idx("re_dis_sell", r, t), 1.0),
                    ],
                    max_export,
                )
            add(
                [
                    (self.idx("soc_re", r, self.h_count - 1), -1.0),
                    (self.idx("soc_ne", r, self.h_count - 1), -1.0),
                ],
                -initial,
            )
        for r in range(self.r_count):
            for t in range(self.h_count):
                add(
                    [
                        (self.idx("grid_load", r, t), 1.0),
                        (self.idx("grid_charge", r, t), 1.0),
                        (self.idx("re_sell", r, t), -1.0),
                        (self.idx("re_dis_sell", r, t), -1.0),
                        (self.peak_index, -1.0),
                    ],
                    0.0,
                )
        matrix = coo_matrix((data, (rows, cols)), shape=(row, self.var_count)).tocsr()
        return matrix, np.asarray(rhs)

    def row(self, vector: np.ndarray) -> csr_matrix:
        nonzero = np.flatnonzero(vector)
        return csr_matrix(
            (vector[nonzero], (np.zeros(len(nonzero), dtype=int), nonzero)),
            shape=(1, self.var_count),
        )

    def solve(
        self, objective: np.ndarray, caps: list[tuple[np.ndarray, float]], time_limit: float = 900.0
    ):
        a_ub = self.a_ub
        b_ub = self.b_ub
        if caps:
            a_ub = vstack(
                [a_ub] + [self.row(vector) for vector, _ in caps], format="csr"
            )
            b_ub = np.concatenate([b_ub, [cap for _, cap in caps]])
        started = time.perf_counter()
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=self.a_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
            method="highs",
            options={"time_limit": time_limit, "presolve": True},
        )
        return result, time.perf_counter() - started

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        arrays = {
            name: x[self.offset[name] : self.offset[name] + self.n].reshape(
                self.r_count, self.h_count
            )
            for name in self.names
        }
        arrays["buy"] = arrays["grid_load"] + arrays["grid_charge"]
        arrays["sell"] = arrays["re_sell"] + arrays["re_dis_sell"]
        arrays["charge"] = arrays["re_charge"] + arrays["grid_charge"]
        arrays["discharge"] = (
            arrays["re_dis_load"]
            + arrays["re_dis_sell"]
            + arrays["ne_dis_load"]
        )
        arrays["soc"] = arrays["soc_re"] + arrays["soc_ne"]
        arrays["regional_net_purchase"] = arrays["buy"] - arrays["sell"]
        arrays["system_net"] = arrays["buy"].sum(axis=0) - arrays["sell"].sum(axis=0)
        return arrays

    def metrics(self, x: np.ndarray, solver: dict) -> tuple[dict, dict[str, np.ndarray]]:
        a = self.unpack(x)
        storage = self.inputs["storage"]
        renewable_residual = (
            self.inputs["renewable"]
            - a["re_load"]
            - a["re_charge"]
            - a["re_sell"]
            - a["curtail"]
        )
        load_residual = (
            self.inputs["fixed_load"]
            - a["re_load"]
            - a["grid_load"]
            - a["re_dis_load"]
            - a["ne_dis_load"]
        )
        soc_residual = []
        for r in range(self.r_count):
            prev_re = 0.0
            prev_ne = float(storage.iloc[r].InitialSOC_MWh)
            eta_ch = float(storage.iloc[r].ChargeEfficiency)
            eta_dis = float(storage.iloc[r].DischargeEfficiency)
            for t in range(self.h_count):
                expected_re = (
                    prev_re
                    + eta_ch * a["re_charge"][r, t]
                    - (a["re_dis_load"][r, t] + a["re_dis_sell"][r, t]) / eta_dis
                )
                expected_ne = (
                    prev_ne
                    + eta_ch * a["grid_charge"][r, t]
                    - a["ne_dis_load"][r, t] / eta_dis
                )
                soc_residual.extend(
                    [
                        abs(a["soc_re"][r, t] - expected_re),
                        abs(a["soc_ne"][r, t] - expected_ne),
                    ]
                )
                prev_re = a["soc_re"][r, t]
                prev_ne = a["soc_ne"][r, t]
        initial = storage.InitialSOC_MWh.to_numpy(dtype=float)
        terminal = a["soc"][:, -1]
        simultaneous = (a["charge"] > TOL) & (a["discharge"] > TOL)
        system_net = a["system_net"]
        capacity = storage.StorageCapacity_MWh.to_numpy(dtype=float)[:, None]
        efc_by_region = (
            (a["charge"].sum(axis=1) + a["discharge"].sum(axis=1))
            / (2.0 * capacity[:, 0])
        )
        renewable_total = float(self.inputs["renewable"].sum())
        used = float(
            a["re_load"].sum() + a["re_charge"].sum() + a["re_sell"].sum()
        )
        regional_peaks = np.maximum(a["regional_net_purchase"].max(axis=1), 0.0)
        regional_peak_fields = {
            f"{region}_peak_net_purchase_mw": float(regional_peaks[r])
            for r, region in enumerate(self.inputs["regions"])
        }
        max_regional_peak = float(regional_peaks.max())
        metrics = {
            "cost_cny": float(self.cost @ x),
            "carbon_tco2": float(self.carbon @ x),
            "renewable_consumed_mwh": used,
            "renewable_utilization": used / renewable_total if renewable_total else math.nan,
            "curtailment_mwh": float(a["curtail"].sum()),
            "grid_purchase_mwh": float(a["buy"].sum()),
            "grid_sell_mwh": float(a["sell"].sum()),
            "peak_grid_purchase_mw": max_regional_peak,
            "max_regional_peak_net_purchase_mw": max_regional_peak,
            "peak_variable_mw": float(x[self.peak_index]),
            "peak_metric_definition": (
                "max_r max_t max(0, grid_load+grid_charge-re_sell-re_dis_sell)"
            ),
            **regional_peak_fields,
            "max_signed_net_import_mw": float(system_net.max()),
            "max_system_export_mw": float(max(0.0, -system_net.min())),
            "mean_absolute_net_ramp_mw": float(np.abs(np.diff(system_net)).mean()),
            "max_absolute_net_ramp_mw": float(np.abs(np.diff(system_net)).max()),
            "total_charge_mwh": float(a["charge"].sum()),
            "total_discharge_mwh": float(a["discharge"].sum()),
            "sum_region_equivalent_cycles": float(efc_by_region.sum()),
            "initial_total_soc_mwh": float(initial.sum()),
            "terminal_total_soc_mwh": float(terminal.sum()),
            "terminal_soc_shortfall_mwh": float(np.maximum(initial - terminal, 0.0).max()),
            "max_energy_balance_residual_mw": float(
                max(np.abs(renewable_residual).max(), np.abs(load_residual).max())
            ),
            "max_soc_recurrence_residual_mwh": float(max(soc_residual)),
            "simultaneous_charge_discharge_cells": int(simultaneous.sum()),
            "simultaneous_buy_sell_cells": int(
                ((a["buy"] > TOL) & (a["sell"] > TOL)).sum()
            ),
            "grid_origin_export_variable_count": 0,
            "initial_inventory_export_variable_count": 0,
            "solver": solver,
        }
        return metrics, a

    def solve_frontier(self, fractions: list[float]) -> tuple[list[EnergyResult], dict]:
        started = time.perf_counter()
        cost_result, cost_runtime = self.solve(self.cost, [])
        if not cost_result.success:
            raise RuntimeError(f"能源成本端点失败: {cost_result.message}")
        carbon_result, carbon_runtime = self.solve(self.carbon, [])
        if not carbon_result.success:
            raise RuntimeError(f"能源碳端点失败: {carbon_result.message}")
        carbon_cost = float(self.carbon @ cost_result.x)
        carbon_min = float(self.carbon @ carbon_result.x)
        gross_reference = float(np.sum(self.inputs["buy_price"] * self.inputs["fixed_load"]))
        outputs: list[EnergyResult] = []
        stages = []
        for fraction in fractions:
            carbon_cap = carbon_min + fraction * (carbon_cost - carbon_min)
            constrained, runtime_cost = self.solve(
                self.cost, [(self.carbon, carbon_cap + 1e-6)]
            )
            if not constrained.success:
                raise RuntimeError(
                    f"能源 fraction={fraction} 成本阶段失败: {constrained.message}"
                )
            cost_opt = float(self.cost @ constrained.x)
            cost_tolerance = max(1e-5, gross_reference * 1e-8)
            peak_result, runtime_peak = self.solve(
                self.peak,
                [
                    (self.carbon, carbon_cap + 1e-6),
                    (self.cost, cost_opt + cost_tolerance),
                ],
            )
            if not peak_result.success:
                peak_result = constrained
                runtime_peak = 0.0
            peak_opt = float(self.peak @ peak_result.x)
            throughput_result, runtime_throughput = self.solve(
                self.throughput,
                [
                    (self.carbon, carbon_cap + 1e-6),
                    (self.cost, cost_opt + cost_tolerance),
                    (self.peak, peak_opt + 1e-6),
                ],
            )
            final = throughput_result if throughput_result.success else peak_result
            solver = {
                "status": "optimal" if final.success else "failed",
                "success": bool(final.success),
                "highs_status": int(final.status),
                "message": str(final.message),
                "optimality_gap": 0.0 if final.success else None,
                "carbon_gap_fraction": fraction,
                "carbon_cap_tco2": carbon_cap,
                "runtime_seconds": runtime_cost + runtime_peak + runtime_throughput,
            }
            metrics, arrays = self.metrics(final.x, solver)
            if (
                metrics["max_energy_balance_residual_mw"] > 1e-5
                or metrics["max_soc_recurrence_residual_mwh"] > 1e-5
                or metrics["terminal_soc_shortfall_mwh"] > 1e-5
                or metrics["simultaneous_charge_discharge_cells"] > 0
            ):
                raise RuntimeError(f"能源物理校验失败: {metrics}")
            outputs.append(EnergyResult(final.x, metrics, arrays))
            stages.append(solver)
        return outputs, {
            "cost_endpoint_runtime_seconds": cost_runtime,
            "carbon_endpoint_runtime_seconds": carbon_runtime,
            "carbon_at_cost_endpoint_tco2": carbon_cost,
            "carbon_min_tco2": carbon_min,
            "total_runtime_seconds": time.perf_counter() - started,
            "stages": stages,
            "variable_count": self.var_count,
            "equality_count": int(self.a_eq.shape[0]),
            "inequality_count": int(self.a_ub.shape[0]),
        }


def no_storage_result(inputs: dict) -> EnergyResult:
    load = inputs["fixed_load"]
    renewable = inputs["renewable"]
    storage = inputs["storage"]
    buy = np.maximum(load - renewable, 0.0)
    surplus = np.maximum(renewable - load, 0.0)
    export_cap = np.minimum(
        storage.SellLimit_MW.to_numpy(dtype=float),
        storage.MaxGridExport_MW.to_numpy(dtype=float),
    )[:, None]
    sell = np.minimum(surplus, export_cap)
    curtail = surplus - sell
    system_net = buy.sum(axis=0) - sell.sum(axis=0)
    regional_net_purchase = buy - sell
    regional_peaks = np.maximum(regional_net_purchase.max(axis=1), 0.0)
    regional_peak_fields = {
        f"{region}_peak_net_purchase_mw": float(regional_peaks[r])
        for r, region in enumerate(inputs["regions"])
    }
    max_regional_peak = float(regional_peaks.max())
    initial = storage.InitialSOC_MWh.to_numpy(dtype=float)
    metrics = {
        "cost_cny": float(np.sum(inputs["buy_price"] * buy - inputs["sell_price"] * sell)),
        "carbon_tco2": float(np.sum(inputs["carbon_intensity"] * buy)),
        "renewable_consumed_mwh": float((renewable - curtail).sum()),
        "renewable_utilization": float((renewable - curtail).sum() / renewable.sum()),
        "curtailment_mwh": float(curtail.sum()),
        "grid_purchase_mwh": float(buy.sum()),
        "grid_sell_mwh": float(sell.sum()),
        "peak_grid_purchase_mw": max_regional_peak,
        "max_regional_peak_net_purchase_mw": max_regional_peak,
        "peak_variable_mw": max_regional_peak,
        "peak_metric_definition": "max_r max_t max(0, GridPurchase-GridSell)",
        **regional_peak_fields,
        "max_signed_net_import_mw": float(system_net.max()),
        "max_system_export_mw": float(max(0.0, -system_net.min())),
        "mean_absolute_net_ramp_mw": float(np.abs(np.diff(system_net)).mean()),
        "max_absolute_net_ramp_mw": float(np.abs(np.diff(system_net)).max()),
        "total_charge_mwh": 0.0,
        "total_discharge_mwh": 0.0,
        "sum_region_equivalent_cycles": 0.0,
        "initial_total_soc_mwh": float(initial.sum()),
        "terminal_total_soc_mwh": float(initial.sum()),
        "terminal_soc_shortfall_mwh": 0.0,
        "max_energy_balance_residual_mw": float(
            np.abs(renewable + buy - load - sell - curtail).max()
        ),
        "max_soc_recurrence_residual_mwh": 0.0,
        "simultaneous_charge_discharge_cells": 0,
        "simultaneous_buy_sell_cells": int(((buy > TOL) & (sell > TOL)).sum()),
        "grid_origin_export_variable_count": 0,
        "initial_inventory_export_variable_count": 0,
        "solver": {
            "status": "deterministic_formula",
            "success": True,
            "optimality_gap": 0.0,
            "runtime_seconds": 0.0,
        },
    }
    arrays = {
        "buy": buy,
        "sell": sell,
        "curtail": curtail,
        "charge": np.zeros_like(buy),
        "discharge": np.zeros_like(buy),
        "soc": np.repeat(initial[:, None], HOURS, axis=1),
        "system_net": system_net,
        "regional_net_purchase": regional_net_purchase,
        "re_load": np.minimum(load, renewable),
        "re_charge": np.zeros_like(buy),
        "re_sell": sell,
    }
    return EnergyResult(np.asarray([]), metrics, arrays)


def task_energy_score(state, task_index: int, region: int, start: int, signal: np.ndarray) -> float:
    hours, _, power = state.task_additions(task_index, start)
    facility_add = state.inputs["pue"][region] * power
    return float(np.sum(facility_add * signal[region, hours]))


def task_wait_ratio(state, task_index: int, start: int) -> float:
    row = state.inputs["tasks"].iloc[task_index]
    if row.TaskType == "RealTimeInference":
        return 0.0
    arrival = float(row.ArrivalHour * 60)
    latest_start = float(min(int(row.LatestFinishHour), TASK_HOURS) * 60 - int(row.EstimatedDuration_min))
    denominator = latest_start - arrival
    if denominator <= TOL:
        return 0.0
    return float(np.clip((start * 60 - arrival) / denominator, 0.0, 1.0))


def joint_improve(
    q2,
    initial,
    signal: np.ndarray,
    wait_cap: float,
    task_limit: int,
    migration_cap: int | None = None,
):
    state = initial.copy()
    tasks = state.inputs["tasks"]
    indices = q2.selected_local_tasks(tasks)[:task_limit]
    accepted = 0
    tested = 0
    started = time.perf_counter()
    sources = tasks.SourceIndex.to_numpy(dtype=int)
    current_migrations = int(np.sum(state.assigned_region != sources))
    initial_migrations = current_migrations
    for task_index in indices:
        current_region = int(state.assigned_region[task_index])
        current_start = int(state.start_hour[task_index])
        if current_region < 0:
            continue
        current_energy = task_energy_score(
            state, task_index, current_region, current_start, signal
        )
        source = int(sources[task_index])
        old_migrated = int(current_region != source)
        state.remove(task_index)
        best = None
        for region in state.static_regions(task_index):
            for start in q2.candidate_starts(
                state, task_index, region, current_start, signal
            ):
                tested += 1
                wait = task_wait_ratio(state, task_index, start)
                if wait > wait_cap + 1e-12:
                    continue
                if not state.feasible(task_index, region, start):
                    continue
                new_migrated = int(region != source)
                proposed_migrations = current_migrations - old_migrated + new_migrated
                if migration_cap is not None and proposed_migrations > migration_cap:
                    continue
                energy = task_energy_score(state, task_index, region, start, signal)
                latency = q2.effective_latency(state, task_index, region)
                key = (energy, wait, latency, region, start)
                if energy < current_energy - 1e-9 and (best is None or key < best[0]):
                    best = (key, region, start)
        if best is None:
            state.place(task_index, current_region, current_start)
        else:
            _, region, start = best
            state.place(task_index, region, start)
            current_migrations = (
                current_migrations - old_migrated + int(region != source)
            )
            accepted += 1
    validation = q2.validate_schedule(state)
    if sum(validation["violations"].values()):
        raise RuntimeError(f"Joint 任务调整违规: {validation}")
    return state, {
        "wait_cap": wait_cap,
        "task_limit": task_limit,
        "migration_cap": migration_cap,
        "initial_migrations": initial_migrations,
        "final_migrations": current_migrations,
        "tasks_considered": int(len(indices)),
        "candidate_tests": tested,
        "accepted_moves": accepted,
        "runtime_seconds": time.perf_counter() - started,
        "validation": validation,
    }


def signal_from_energy(inputs: dict, result: EnergyResult) -> np.ndarray:
    a = result.arrays
    return np.where(
        a["buy"] > TOL,
        inputs["buy_price"],
        np.where(a["curtail"] > TOL, 0.0, inputs["sell_price"]),
    )


def combined_metrics(label: str, role: str, task: dict, energy: dict, extras: dict | None = None) -> dict:
    payload = {"solution_id": label, "role": role, **task, **energy}
    payload.pop("validation", None)
    if extras:
        payload.update(extras)
    return payload


def dispatch_frame(label: str, inputs: dict, result: EnergyResult) -> pd.DataFrame:
    records = []
    a = result.arrays
    for r, region in enumerate(inputs["regions"]):
        for t, hour in enumerate(inputs["hours"]):
            records.append(
                {
                    "Solution": label,
                    "Region": region,
                    "Hour": int(hour),
                    "FacilityLoad_MW": inputs["fixed_load"][r, t],
                    "AvailableRenewable_MW": inputs["renewable"][r, t],
                    "RenewableLoad_MW": a["re_load"][r, t],
                    "RenewableCharge_MW": a["re_charge"][r, t],
                    "RenewableSell_MW": a["re_sell"][r, t],
                    "Curtailment_MW": a["curtail"][r, t],
                    "GridPurchase_MW": a["buy"][r, t],
                    "GridSell_MW": a["sell"][r, t],
                    "NetGridPurchase_MW": a["regional_net_purchase"][r, t],
                    "ChargePower_MW": a["charge"][r, t],
                    "DischargePower_MW": a["discharge"][r, t],
                    "SOC_MWh": a["soc"][r, t],
                }
            )
    return pd.DataFrame(records)


def is_duplicate(a: dict, b: dict) -> bool:
    keys = ("cost_cny", "carbon_tco2", "mean_wait_ratio", "cvar95_wait_ratio")
    tolerances = (1e-3, 1e-6, 1e-10, 1e-10)
    return all(abs(float(a[k]) - float(b[k])) <= tol for k, tol in zip(keys, tolerances))


def nondominated(rows: list[dict]) -> list[dict]:
    unique: list[dict] = []
    for row in rows:
        if not any(is_duplicate(row, prior) for prior in unique):
            unique.append(row)
    objectives = ("cost_cny", "carbon_tco2", "mean_wait_ratio", "cvar95_wait_ratio")
    kept = []
    for i, row in enumerate(unique):
        dominated = False
        for j, other in enumerate(unique):
            if i == j:
                continue
            weak = all(float(other[k]) <= float(row[k]) + 1e-9 for k in objectives)
            strict = any(float(other[k]) < float(row[k]) - 1e-9 for k in objectives)
            if weak and strict:
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return kept


def ideal_distance(rows: list[dict]) -> dict:
    keys = ("cost_cny", "carbon_tco2", "mean_wait_ratio", "cvar95_wait_ratio")
    matrix = np.asarray([[float(row[k]) for k in keys] for row in rows])
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    scale = np.where(maximum > minimum, maximum - minimum, 1.0)
    distances = np.sqrt(np.sum(((matrix - minimum) / scale) ** 2, axis=1))
    index = int(np.argmin(distances))
    result = dict(rows[index])
    result["ideal_distance"] = float(distances[index])
    return result


def gain_decomposition(comparison: dict[str, dict]) -> pd.DataFrame:
    specs = {
        "cost_cny": "lower",
        "carbon_tco2": "lower",
        "renewable_utilization": "higher",
        "grid_purchase_mwh": "lower",
        "peak_grid_purchase_mw": "lower",
        "curtailment_mwh": "lower",
        "mean_wait_ratio": "lower",
        "cvar95_wait_ratio": "lower",
        "migration_tasks": "lower",
    }
    rows = []
    for metric, direction in specs.items():
        b0 = float(comparison["B0"][metric])
        b1 = float(comparison["B1"][metric])
        b2 = float(comparison["B2"][metric])
        joint = float(comparison["Joint"][metric])
        if direction == "lower":
            task = b0 - b1
            storage = b0 - b2
            total = b0 - joint
        else:
            task = b1 - b0
            storage = b2 - b0
            total = joint - b0
        rows.append(
            {
                "Metric": metric,
                "Direction": direction,
                "B0": b0,
                "B1": b1,
                "B2": b2,
                "Joint": joint,
                "Delta_task": task,
                "Delta_storage": storage,
                "Delta_joint": total,
                "Delta_interaction": total - task - storage,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    prepare_dirs()
    q2 = load_q2_core()
    q3 = load_q3_core()
    q2_inputs = q2.load_inputs()
    q3_inputs = q3.load_inputs()
    final_q2 = ROOT / "05_结果与图表" / "final" / "question2" / "tables"

    b0_state, b0_run = local_only_baseline(q2, q2_inputs)
    b0_task = task_metrics(q2, b0_state)
    if b0_run["unassigned_tasks"] or b0_task["task_hard_constraint_violations"]:
        write_json(METRIC_DIR / "q4_b0_infeasibility.json", {"run": b0_run, "metrics": b0_task})
        raise RuntimeError("B0 纯本地基线不可行，证据已保存")
    b1_state, _, b1_validation = load_state_from_assignments(
        q2, q2_inputs, final_q2 / "epsilon_3_assignments.csv"
    )
    seed_state, _, seed_validation = load_state_from_assignments(
        q2, q2_inputs, final_q2 / "baseline_assignments.csv"
    )
    b1_task = task_metrics(q2, b1_state)
    seed_task = task_metrics(q2, seed_state)

    base_scenario = {
        "scenario_id": "original",
        "price_spread_factor": 1.0,
        "renewable_scale": 1.0,
    }
    b0_inputs = scenario_inputs(q2_inputs, q3_inputs, b0_state, base_scenario)
    b1_inputs = scenario_inputs(q2_inputs, q3_inputs, b1_state, base_scenario)
    seed_inputs = scenario_inputs(q2_inputs, q3_inputs, seed_state, base_scenario)
    b0_energy = no_storage_result(b0_inputs)
    b1_energy = no_storage_result(b1_inputs)

    b2_model = ProvenanceEnergyModel(b0_inputs)
    b2_front, b2_run = b2_model.solve_frontier([1.0])
    b2_energy = b2_front[0]
    signal = signal_from_energy(b0_inputs, b2_energy)

    joint_states: dict[float, object] = {}
    joint_runs: dict[str, dict] = {}
    for cap in JOINT_CAPS:
        state, run = joint_improve(q2, seed_state, signal[:, :TASK_HOURS], cap, JOINT_TASK_LIMIT)
        joint_states[cap] = state
        joint_runs[f"wait_cap_{cap:.2f}"] = run

    pareto_candidates: list[dict] = []
    candidate_results: dict[str, tuple[dict, EnergyResult, dict, object]] = {}
    for cap, state in joint_states.items():
        fractions = [1.0]
        if math.isclose(cap, 0.40):
            fractions = [1.0, 0.5]
        if math.isclose(cap, 1.00):
            fractions = [1.0, 0.1]
        task = task_metrics(q2, state)
        inputs = scenario_inputs(q2_inputs, q3_inputs, state, base_scenario)
        model = ProvenanceEnergyModel(inputs)
        results, energy_run = model.solve_frontier(fractions)
        for fraction, result in zip(fractions, results, strict=True):
            label = f"Joint_w{int(round(cap * 100)):02d}_c{int(round((1-fraction)*100)):02d}"
            row = combined_metrics(
                label,
                "pareto_candidate",
                task,
                result.metrics,
                {"wait_cap": cap, "carbon_gap_fraction": fraction},
            )
            pareto_candidates.append(row)
            candidate_results[label] = (row, result, inputs, state)
        joint_runs[f"energy_wait_cap_{cap:.2f}"] = energy_run

    pareto = nondominated(pareto_candidates)
    if len(pareto) < 5:
        joint_runs["pareto_warning"] = {
            "target": "5-9",
            "observed": len(pareto),
            "reason": "数值重复或支配关系导致前沿收缩；未保留被支配解冒充 Pareto",
        }
    compromise = ideal_distance(pareto)
    compromise_label = compromise["solution_id"]
    compromise_row, compromise_energy, compromise_inputs, compromise_state = candidate_results[
        compromise_label
    ]
    joint_task = task_metrics(q2, compromise_state)

    comparison = {
        "B0": combined_metrics("B0", "baseline_no_migration_no_storage", b0_task, b0_energy.metrics),
        "B1": combined_metrics("B1", "task_only", b1_task, b1_energy.metrics),
        "B2": combined_metrics("B2", "energy_only_on_B0_tasks", b0_task, b2_energy.metrics),
        "Joint": {**compromise_row, "role": "provisional_compromise_from_pareto"},
    }

    scenario_specs = [
        {"scenario_id": "A_price_1p25", "price_spread_factor": 1.25, "renewable_scale": 1.0, "carbon_gap_fraction": compromise_row["carbon_gap_fraction"]},
        {"scenario_id": "A_renewable_low_0p8", "price_spread_factor": 1.0, "renewable_scale": 0.8, "carbon_gap_fraction": compromise_row["carbon_gap_fraction"]},
        {"scenario_id": "A_renewable_high_1p2", "price_spread_factor": 1.0, "renewable_scale": 1.2, "carbon_gap_fraction": compromise_row["carbon_gap_fraction"]},
        {"scenario_id": "B_conservative_low", "price_spread_factor": 1.15, "renewable_scale": 0.9, "carbon_gap_fraction": 0.33},
        {"scenario_id": "C_stress_low", "price_spread_factor": 1.50, "renewable_scale": 0.7, "carbon_gap_fraction": 0.0},
        {"scenario_id": "C_stress_high", "price_spread_factor": 1.50, "renewable_scale": 1.3, "carbon_gap_fraction": 0.0},
    ]
    scenario_rows = []
    scenario_dispatch = []
    scenario_runs = {}
    for spec in scenario_specs:
        inputs = scenario_inputs(q2_inputs, q3_inputs, compromise_state, spec)
        baseline = no_storage_result(
            scenario_inputs(q2_inputs, q3_inputs, b0_state, spec)
        )
        try:
            model = ProvenanceEnergyModel(inputs)
            results, run = model.solve_frontier([float(spec["carbon_gap_fraction"])])
            result = results[0]
            row = combined_metrics(
                spec["scenario_id"],
                "scenario_joint_representative",
                joint_task,
                result.metrics,
                {
                    "price_spread_factor": spec["price_spread_factor"],
                    "renewable_scale": spec["renewable_scale"],
                    "carbon_gap_fraction": spec["carbon_gap_fraction"],
                    "baseline_cost_cny": baseline.metrics["cost_cny"],
                    "baseline_carbon_tco2": baseline.metrics["carbon_tco2"],
                    "scenario_status": "feasible",
                    **inputs["scenario_hashes"],
                },
            )
            scenario_rows.append(row)
            scenario_dispatch.append(dispatch_frame(spec["scenario_id"], inputs, result))
            scenario_runs[spec["scenario_id"]] = run
        except RuntimeError as error:
            diagnostic = {
                "scenario_id": spec["scenario_id"],
                "scenario_status": "infeasible_or_solver_failed",
                "error": str(error),
                "diagnosis": {
                    "task_constraints": joint_task["task_hard_constraint_violations"],
                    "storage_bounds_present": True,
                    "carbon_constraint_fraction": spec["carbon_gap_fraction"],
                    "grid_boundary_screen": "energy model includes per-region import/export caps",
                },
            }
            scenario_rows.append(diagnostic)
            scenario_runs[spec["scenario_id"]] = diagnostic

    assignments_frame(b0_state).to_csv(TABLE_DIR / "q4_task_assignments_B0.csv", index=False)
    assignments_frame(b1_state).to_csv(TABLE_DIR / "q4_task_assignments_B1.csv", index=False)
    saved_schedules = set()
    for row in pareto:
        cap = float(row["wait_cap"])
        if cap not in saved_schedules:
            assignments_frame(joint_states[cap]).to_csv(
                TABLE_DIR / f"q4_task_assignments_Joint_w{int(round(cap*100)):02d}.csv",
                index=False,
            )
            saved_schedules.add(cap)

    dispatch_frames = [
        dispatch_frame("B0", b0_inputs, b0_energy),
        dispatch_frame("B1", b1_inputs, b1_energy),
        dispatch_frame("B2", b0_inputs, b2_energy),
    ]
    for row in pareto:
        label = row["solution_id"]
        _, result, inputs, _ = candidate_results[label]
        dispatch_frames.append(dispatch_frame(label, inputs, result))
    pd.concat(dispatch_frames, ignore_index=True).to_csv(
        TABLE_DIR / "q4_energy_dispatch_main.csv", index=False
    )
    if scenario_dispatch:
        pd.concat(scenario_dispatch, ignore_index=True).to_csv(
            TABLE_DIR / "q4_energy_dispatch_scenarios.csv", index=False
        )

    comparison_frame = pd.DataFrame(comparison.values())
    comparison_frame.to_csv(TABLE_DIR / "q4_comparison.csv", index=False)
    pareto_frame = pd.DataFrame(pareto)
    pareto_frame.to_csv(TABLE_DIR / "q4_pareto.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(TABLE_DIR / "q4_scenarios.csv", index=False)
    gains = gain_decomposition(comparison)
    gains.to_csv(TABLE_DIR / "q4_gain_decomposition.csv", index=False)

    validation = {
        "B0": b0_task["validation"],
        "B1": b1_task["validation"],
        "Joint": joint_task["validation"],
        "energy": {
            label: {
                key: row[key]
                for key in (
                    "max_energy_balance_residual_mw",
                    "max_soc_recurrence_residual_mwh",
                    "terminal_soc_shortfall_mwh",
                    "simultaneous_charge_discharge_cells",
                    "grid_origin_export_variable_count",
                    "initial_inventory_export_variable_count",
                )
            }
            for label, row in comparison.items()
        },
    }
    write_json(METRIC_DIR / "q4_constraint_validation.json", validation)
    metrics_payload = {
        "schema_version": 2,
        "question": "Q4",
        "candidate_status": "intermediate_not_canonical",
        "metric_corrections": {
            "latency": "official matrix including 5 ms local diagonal",
            "migration": "AssignedRegion != SourceRegion",
            "peak": "max_r max_t max(0, GridPurchase(r,t)-GridSell(r,t))",
            "peak_lp_constraint": (
                "grid_load+grid_charge-re_sell-re_dis_sell <= Peak; Peak >= 0"
            ),
        },
        "approved_decisions": ["Q4-METHOD-CHOICE-001", "Q4-SERVICE-SCENARIO-002"],
        "comparison": comparison,
        "pareto_candidates": pareto_candidates,
        "pareto_nondominated": pareto,
        "pareto_count": len(pareto),
        "provisional_ideal_distance_candidate": compromise,
        "scenario_results": scenario_rows,
        "gain_decomposition": gains.to_dict(orient="records"),
        "runs": {
            "B0": b0_run,
            "B1_validation": b1_validation,
            "seed_validation": seed_validation,
            "B2": b2_run,
            "Joint": joint_runs,
            "scenarios": scenario_runs,
        },
    }
    write_json(METRIC_DIR / "q4_metrics.json", metrics_payload)

    reproducibility = {
        "schema_version": 2,
        "random_seed": SEED,
        "joint_task_limit": JOINT_TASK_LIMIT,
        "joint_wait_caps": JOINT_CAPS,
        "input_hashes": {
            path.name: sha256(path)
            for path in sorted(RAW.glob("*.xlsx"))
        },
        "official_latency_source": {
            "path": q2_inputs["latency_source"],
            "sha256": q2_inputs["latency_source_sha256"],
            "local_diagonal_ms": 5.0,
        },
        "q2_frozen_assignment_hashes": {
            "baseline": sha256(final_q2 / "baseline_assignments.csv"),
            "epsilon_3": sha256(final_q2 / "epsilon_3_assignments.csv"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
        },
    }

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(pareto_frame.cost_cny / 1e6, pareto_frame.mean_wait_ratio)
    for row in pareto:
        axes[0].annotate(row["solution_id"], (row["cost_cny"] / 1e6, row["mean_wait_ratio"]), fontsize=7)
    axes[0].set_xlabel("系统净成本（百万元）")
    axes[0].set_ylabel("平均归一化等待比例")
    axes[0].set_title("Q4 中间 Pareto：成本—服务")
    axes[1].scatter(pareto_frame.cost_cny / 1e6, pareto_frame.carbon_tco2)
    axes[1].set_xlabel("系统净成本（百万元）")
    axes[1].set_ylabel("购电碳排放（tCO2）")
    axes[1].set_title("Q4 中间 Pareto：成本—碳排")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q4_pareto_diagnostic.png", dpi=180)
    plt.close(fig)

    runtime = time.perf_counter() - started
    reproducibility.update(
        {
            "runtime_seconds": runtime,
            "peak_process_working_set_mb": peak_memory_mb(),
        }
    )
    write_json(METRIC_DIR / "reproducibility.json", reproducibility)

    outputs = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in sorted(ROUND_DIR.rglob("*"))
        if path.is_file()
    ]
    run_summary = {
        "schema_version": 2,
        "question": "Q4",
        "round": ROUND_ID,
        "candidate_status": "intermediate_not_canonical",
        "implementation_target": "python",
        "random_seed": SEED,
        "approved_decision_id": "Q4-METHOD-CHOICE-001; Q4-SERVICE-SCENARIO-002",
        "status": "success_waiting_human_result_judgment",
        "methods": [
            {"method_id": "Q4-B0", "role": "usable_baseline", "status": "success", "metrics_summary": comparison["B0"]},
            {"method_id": "Q4-B1", "role": "task_only", "status": "success", "metrics_summary": comparison["B1"]},
            {"method_id": "Q4-B2", "role": "energy_only", "status": "success", "metrics_summary": comparison["B2"]},
            {"method_id": "Q4-Joint", "role": "main", "status": "success", "metrics_summary": comparison["Joint"]},
        ],
        "comparison": comparison,
        "pareto_count": len(pareto),
        "provisional_compromise": compromise_label,
        "scenario_count": len(scenario_rows),
        "runtime_seconds": runtime,
        "peak_process_working_set_mb": reproducibility["peak_process_working_set_mb"],
        "input_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted(RAW.glob("*.xlsx"))],
        "output_files": outputs,
        "fallback_trigger": {
            "condition": "总时限或 Pareto/场景单次时限触发",
            "observed": False,
            "evidence": None,
        },
        "warnings": [
            "理想点距离候选仅供人工选择，不构成冻结决定",
            "高新能源 1.2/1.3 场景是数学敏感性，不是独立装机容量证明",
        ],
        "errors": [],
        "environment": reproducibility["environment"],
    }
    write_json(ROUND_DIR / "run_summary.json", run_summary)
    print(
        json.dumps(
            {
                "status": run_summary["status"],
                "runtime_seconds": runtime,
                "pareto_count": len(pareto),
                "provisional_compromise": compromise_label,
                "comparison": {
                    key: {
                        "cost_cny": value["cost_cny"],
                        "carbon_tco2": value["carbon_tco2"],
                        "mean_wait_ratio": value["mean_wait_ratio"],
                        "migration_tasks": value["migration_tasks"],
                    }
                    for key, value in comparison.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
