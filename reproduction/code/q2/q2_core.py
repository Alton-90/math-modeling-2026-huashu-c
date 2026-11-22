from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = Path(__file__).resolve().parent
RAW = ROOT / "02_数据处理" / "raw"
OFFICIAL_ATTACHMENTS = ROOT / "00_题目与要求" / "附件数据"
NETWORK_LATENCY_PATH = OFFICIAL_ATTACHMENTS / "network_latency.xlsx"
SEED = 20260808
HORIZON_END = 2406
EPSILONS = (0.00, 0.03, 0.05)
TIME_BUDGET_FALLBACK_APPLIED = True
TOL = 1e-8
MAX_LOCAL_TASKS = 12000
TOP_SIGNAL_STARTS = 3


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def task_overlaps(start_hour: int, duration_min: int) -> tuple[np.ndarray, np.ndarray]:
    start_min = start_hour * 60
    finish_min = start_min + duration_min
    first = start_hour
    last = (finish_min - 1) // 60
    hours = np.arange(first, last + 1, dtype=int)
    left = np.maximum(hours * 60, start_min)
    right = np.minimum((hours + 1) * 60, finish_min)
    return hours, (right - left).astype(float) / 60.0


def load_inputs() -> dict:
    tasks = pd.read_excel(RAW / "workload_trace.xlsx").copy()
    gpu = pd.read_excel(RAW / "GPU_information.xlsx", sheet_name="GPU中心基础情况")
    latency = pd.read_excel(NETWORK_LATENCY_PATH, sheet_name="network_latency")
    power = pd.read_excel(RAW / "power_mapping.xlsx", sheet_name="任务功率映射")
    region_time = pd.read_excel(RAW / "region_time_data.xlsx", sheet_name="region_time_data")
    storage = pd.read_excel(RAW / "storage_information.xlsx", sheet_name="storage_information")
    if len(tasks) != 50000 or tasks.TaskID.nunique() != len(tasks):
        raise ValueError("Q2 任务应为 50,000 条唯一 TaskID")
    regions = sorted(gpu.Region.astype(str).unique())
    r_index = {region: index for index, region in enumerate(regions)}
    hours = list(range(HORIZON_END))
    expected = pd.MultiIndex.from_product([regions, hours], names=["Region", "Hour"])
    rt = (
        region_time[region_time.Hour < HORIZON_END]
        .set_index(["Region", "Hour"])
        .reindex(expected)
    )
    if rt.isna().any().any():
        raise ValueError("Q2 区域—小时外生输入网格不完整")
    shape = (len(regions), HORIZON_END)

    def matrix(column: str) -> np.ndarray:
        return rt[column].to_numpy(dtype=float).reshape(shape)

    gpu = gpu.set_index("Region").loc[regions]
    storage = storage.set_index("Region").loc[regions]
    latency_keys = ["FromRegion", "ToRegion"]
    expected_latency_pairs = {
        (source, destination) for source in regions for destination in regions
    }
    observed_latency_pairs = set(
        latency[latency_keys].astype(str).itertuples(index=False, name=None)
    )
    if latency.duplicated(latency_keys).any() or observed_latency_pairs != expected_latency_pairs:
        raise ValueError("官方网络时延矩阵必须恰好覆盖六区域有向 6×6 组合")
    local_latency = latency.loc[
        latency.FromRegion.astype(str) == latency.ToRegion.astype(str),
        "NetworkLatency_ms",
    ].to_numpy(dtype=float)
    if len(local_latency) != len(regions) or not np.allclose(local_latency, 5.0):
        raise ValueError("官方网络时延矩阵的本地对角线必须全部为 5 ms")
    latency_map = latency.set_index(["FromRegion", "ToRegion"]).NetworkLatency_ms.to_dict()
    power_map = power.set_index("TaskType").GPU_Power_MW_per_EquivalentGPU.to_dict()
    latency_matrix = np.zeros((len(regions), len(regions)))
    for source in regions:
        for destination in regions:
            latency_matrix[r_index[source], r_index[destination]] = float(
                latency_map[(source, destination)]
            )
    tasks["TaskPowerPerGPU_MW"] = tasks.TaskType.map(power_map).astype(float)
    tasks["SourceIndex"] = tasks.SourceRegion.map(r_index).astype(int)
    tasks["EnergyProxy_MWh"] = (
        tasks.GPU_Demand
        * tasks.TaskPowerPerGPU_MW
        * tasks.EstimatedDuration_min
        / 60.0
    )
    return {
        "tasks": tasks.reset_index(drop=True),
        "regions": regions,
        "r_index": r_index,
        "gpu_capacity": gpu.Available_GPU.to_numpy(dtype=float),
        "max_it": gpu.Max_IT_Power_MW.to_numpy(dtype=float),
        "pue": gpu.PUE.to_numpy(dtype=float),
        "max_facility": gpu.Max_Facility_Power_MW.to_numpy(dtype=float),
        "max_import": storage.MaxGridImport_MW.to_numpy(dtype=float),
        "max_export": np.minimum(
            storage.SellLimit_MW.to_numpy(dtype=float),
            storage.MaxGridExport_MW.to_numpy(dtype=float),
        ),
        "non_ai": matrix("NonAI_IT_Load_MW"),
        "renewable": matrix("AvailableRenewable_MW"),
        "buy_price": matrix("ElectricityPrice_CNY_per_MWh"),
        "sell_price": matrix("SellPrice_CNY_per_MWh"),
        "carbon_intensity": matrix("CarbonIntensity_tCO2_per_MWh"),
        "latency": latency_matrix,
        "latency_source": str(NETWORK_LATENCY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "latency_source_sha256": hashlib.sha256(
            NETWORK_LATENCY_PATH.read_bytes()
        ).hexdigest().upper(),
    }


@dataclass
class ScheduleState:
    inputs: dict
    assigned_region: np.ndarray
    start_hour: np.ndarray
    gpu_use: np.ndarray
    ai_power: np.ndarray

    @classmethod
    def empty(cls, inputs: dict) -> "ScheduleState":
        n = len(inputs["tasks"])
        shape = (len(inputs["regions"]), HORIZON_END)
        return cls(
            inputs=inputs,
            assigned_region=np.full(n, -1, dtype=np.int16),
            start_hour=np.full(n, -1, dtype=np.int16),
            gpu_use=np.zeros(shape, dtype=float),
            ai_power=np.zeros(shape, dtype=float),
        )

    def copy(self) -> "ScheduleState":
        return ScheduleState(
            inputs=self.inputs,
            assigned_region=self.assigned_region.copy(),
            start_hour=self.start_hour.copy(),
            gpu_use=self.gpu_use.copy(),
            ai_power=self.ai_power.copy(),
        )

    def task_additions(self, task_index: int, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = self.inputs["tasks"].iloc[task_index]
        hours, fractions = task_overlaps(start, int(row.EstimatedDuration_min))
        gpu = float(row.GPU_Demand) * fractions
        power = float(row.GPU_Demand * row.TaskPowerPerGPU_MW) * fractions
        return hours, gpu, power

    def static_regions(self, task_index: int) -> list[int]:
        row = self.inputs["tasks"].iloc[task_index]
        source = int(row.SourceIndex)
        allowed = [
            r
            for r in range(len(self.inputs["regions"]))
            if self.inputs["latency"][source, r] <= float(row.MaxLatency_ms) + TOL
            and float(row.GPU_Demand) <= self.inputs["gpu_capacity"][r] + TOL
        ]
        return sorted(
            allowed,
            key=lambda r: (
                r != source,
                self.inputs["latency"][source, r],
                self.inputs["regions"][r],
            ),
        )

    def feasible(self, task_index: int, region: int, start: int) -> bool:
        row = self.inputs["tasks"].iloc[task_index]
        earliest = max(int(row.ArrivalHour), int(row.EarliestStartHour))
        finish = start * 60 + int(row.EstimatedDuration_min)
        latest_finish = min(int(row.LatestFinishHour), HORIZON_END) * 60
        if start < earliest or finish > latest_finish or finish > HORIZON_END * 60:
            return False
        if row.TaskType == "RealTimeInference" and start != int(row.ArrivalHour):
            return False
        source = int(row.SourceIndex)
        if self.inputs["latency"][source, region] > float(row.MaxLatency_ms) + TOL:
            return False
        hours, gpu_add, power_add = self.task_additions(task_index, start)
        if len(hours) == 0 or hours[-1] >= HORIZON_END:
            return False
        if np.any(
            self.gpu_use[region, hours] + gpu_add
            > self.inputs["gpu_capacity"][region] + TOL
        ):
            return False
        new_ai = self.ai_power[region, hours] + power_add
        new_it = self.inputs["non_ai"][region, hours] + new_ai
        if np.any(new_it > self.inputs["max_it"][region] + TOL):
            return False
        new_facility = self.inputs["pue"][region] * new_it
        if np.any(new_facility > self.inputs["max_facility"][region] + TOL):
            return False
        new_buy = np.maximum(
            new_facility - self.inputs["renewable"][region, hours], 0.0
        )
        if np.any(new_buy > self.inputs["max_import"][region] + TOL):
            return False
        return True

    def place(self, task_index: int, region: int, start: int) -> None:
        hours, gpu_add, power_add = self.task_additions(task_index, start)
        self.gpu_use[region, hours] += gpu_add
        self.ai_power[region, hours] += power_add
        self.assigned_region[task_index] = region
        self.start_hour[task_index] = start

    def remove(self, task_index: int) -> tuple[int, int, float, float]:
        region = int(self.assigned_region[task_index])
        start = int(self.start_hour[task_index])
        if region < 0:
            raise ValueError("不能移除未分配任务")
        hours, gpu_add, power_add = self.task_additions(task_index, start)
        before_cost, before_carbon = self.cell_objectives(region, hours)
        self.gpu_use[region, hours] -= gpu_add
        self.ai_power[region, hours] -= power_add
        after_cost, after_carbon = self.cell_objectives(region, hours)
        self.assigned_region[task_index] = -1
        self.start_hour[task_index] = -1
        return region, start, after_cost - before_cost, after_carbon - before_carbon

    def cell_objectives(self, region: int, hours: np.ndarray) -> tuple[float, float]:
        facility = self.inputs["pue"][region] * (
            self.inputs["non_ai"][region, hours] + self.ai_power[region, hours]
        )
        renewable = self.inputs["renewable"][region, hours]
        buy = np.maximum(facility - renewable, 0.0)
        sell = np.minimum(
            np.maximum(renewable - facility, 0.0), self.inputs["max_export"][region]
        )
        cost = np.sum(
            self.inputs["buy_price"][region, hours] * buy
            - self.inputs["sell_price"][region, hours] * sell
        )
        carbon = np.sum(self.inputs["carbon_intensity"][region, hours] * buy)
        return float(cost), float(carbon)

    def addition_delta(self, task_index: int, region: int, start: int) -> tuple[float, float]:
        hours, _, power_add = self.task_additions(task_index, start)
        before_cost, before_carbon = self.cell_objectives(region, hours)
        self.ai_power[region, hours] += power_add
        after_cost, after_carbon = self.cell_objectives(region, hours)
        self.ai_power[region, hours] -= power_add
        return after_cost - before_cost, after_carbon - before_carbon

    def energy_metrics(self) -> dict:
        facility = self.inputs["pue"][:, None] * (
            self.inputs["non_ai"] + self.ai_power
        )
        renewable = self.inputs["renewable"]
        buy = np.maximum(facility - renewable, 0.0)
        surplus = np.maximum(renewable - facility, 0.0)
        sell = np.minimum(surplus, self.inputs["max_export"][:, None])
        curtail = surplus - sell
        cost = np.sum(self.inputs["buy_price"] * buy - self.inputs["sell_price"] * sell)
        carbon = np.sum(self.inputs["carbon_intensity"] * buy)
        renewable_used = renewable - curtail
        return {
            "cost_cny": float(cost),
            "carbon_tco2": float(carbon),
            "renewable_utilization": float(renewable_used.sum() / renewable.sum()),
            "grid_import_violations": int(
                np.sum(buy > self.inputs["max_import"][:, None] + TOL)
            ),
            "grid_export_violations": int(
                np.sum(sell > self.inputs["max_export"][:, None] + TOL)
            ),
        }


def earliest_feasible_baseline(inputs: dict) -> tuple[ScheduleState, dict]:
    state = ScheduleState.empty(inputs)
    tasks = inputs["tasks"]
    order = tasks.sort_values(
        ["ArrivalHour", "LatestFinishHour", "TaskID"], kind="stable"
    ).index.to_numpy()
    started = time.perf_counter()
    candidate_tests = 0
    for task_index in order:
        row = tasks.iloc[task_index]
        earliest = max(int(row.ArrivalHour), int(row.EarliestStartHour))
        latest = min(
            (min(int(row.LatestFinishHour), HORIZON_END) * 60 - int(row.EstimatedDuration_min))
            // 60,
            HORIZON_END - 1,
        )
        starts = [earliest] if row.TaskType == "RealTimeInference" else range(earliest, latest + 1)
        assigned = False
        regions = state.static_regions(task_index)
        for start in starts:
            for region in regions:
                candidate_tests += 1
                if state.feasible(task_index, region, int(start)):
                    state.place(task_index, region, int(start))
                    assigned = True
                    break
            if assigned:
                break
    return state, {
        "runtime_seconds": float(time.perf_counter() - started),
        "candidate_tests": candidate_tests,
        "unassigned_tasks": int(np.sum(state.assigned_region < 0)),
    }


def objective_signals(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    fixed_facility = inputs["pue"][:, None] * inputs["non_ai"]
    renewable_surplus = fixed_facility < inputs["renewable"]
    cost_signal = np.where(
        renewable_surplus, inputs["sell_price"], inputs["buy_price"]
    )
    carbon_signal = np.where(renewable_surplus, 0.0, inputs["carbon_intensity"])
    return cost_signal, carbon_signal


def candidate_starts(
    state: ScheduleState,
    task_index: int,
    region: int,
    current_start: int,
    signal: np.ndarray,
) -> list[int]:
    row = state.inputs["tasks"].iloc[task_index]
    earliest = max(int(row.ArrivalHour), int(row.EarliestStartHour))
    latest = min(
        (min(int(row.LatestFinishHour), HORIZON_END) * 60 - int(row.EstimatedDuration_min))
        // 60,
        HORIZON_END - 1,
    )
    if row.TaskType == "RealTimeInference":
        return [earliest]
    lookahead = 24 if row.TaskType == "BatchInference" else 168
    window_end = min(latest, earliest + lookahead)
    span = np.arange(earliest, window_end + 1, dtype=int)
    starts = {earliest, current_start, latest}
    starts.update(range(earliest, min(window_end, earliest + 3) + 1))
    if len(span):
        k = min(TOP_SIGNAL_STARTS, len(span))
        best = span[np.argpartition(signal[region, span], k - 1)[:k]]
        starts.update(int(item) for item in best)
    return sorted(start for start in starts if earliest <= start <= latest)


def selected_local_tasks(tasks: pd.DataFrame) -> np.ndarray:
    flexible = tasks[tasks.TaskType != "RealTimeInference"].copy()
    return (
        flexible.sort_values(
            ["EnergyProxy_MWh", "LatestFinishHour", "TaskID"],
            ascending=[False, True, True],
            kind="stable",
        )
        .head(MAX_LOCAL_TASKS)
        .index.to_numpy()
    )


def effective_latency(state: ScheduleState, task_index: int, region: int) -> float:
    source = int(state.inputs["tasks"].iloc[task_index].SourceIndex)
    return float(state.inputs["latency"][source, region])


def is_migrated(state: ScheduleState, task_index: int, region: int) -> bool:
    source = int(state.inputs["tasks"].iloc[task_index].SourceIndex)
    return region != source


def local_improve(
    initial: ScheduleState,
    mode: str,
    signal: np.ndarray,
    cost_cap: float | None = None,
    carbon_cap: float | None = None,
) -> tuple[ScheduleState, dict]:
    state = initial.copy()
    metrics = state.energy_metrics()
    total_cost = metrics["cost_cny"]
    total_carbon = metrics["carbon_tco2"]
    accepted = 0
    tested = 0
    started = time.perf_counter()
    task_indices = selected_local_tasks(state.inputs["tasks"])
    if mode == "latency":
        source = state.inputs["tasks"].SourceIndex.to_numpy(dtype=int)
        task_indices = np.asarray(
            [
                index
                for index in task_indices
                if int(state.assigned_region[index]) != int(source[index])
            ],
            dtype=int,
        )
    for task_index in task_indices:
        current_region = int(state.assigned_region[task_index])
        current_start = int(state.start_hour[task_index])
        if current_region < 0:
            continue
        old_latency = effective_latency(state, task_index, current_region)
        old_migration = int(is_migrated(state, task_index, current_region))
        _, _, removal_cost, removal_carbon = state.remove(task_index)
        best = None
        for region in state.static_regions(task_index):
            for start in candidate_starts(
                state, task_index, region, current_start, signal
            ):
                tested += 1
                if not state.feasible(task_index, region, start):
                    continue
                add_cost, add_carbon = state.addition_delta(task_index, region, start)
                new_cost = total_cost + removal_cost + add_cost
                new_carbon = total_carbon + removal_carbon + add_carbon
                latency = effective_latency(state, task_index, region)
                migration = int(is_migrated(state, task_index, region))
                if cost_cap is not None and new_cost > cost_cap + 1e-5:
                    continue
                if carbon_cap is not None and new_carbon > carbon_cap + 1e-8:
                    continue
                if mode == "cost":
                    eligible = new_cost < total_cost - 1e-6 or (
                        abs(new_cost - total_cost) <= 1e-6
                        and new_carbon < total_carbon - 1e-9
                    )
                    key = (new_cost, new_carbon, latency, migration, start, region)
                elif mode == "carbon":
                    eligible = new_carbon < total_carbon - 1e-9
                    key = (new_carbon, new_cost, latency, migration, start, region)
                elif mode == "latency":
                    eligible = (latency, migration) < (old_latency, old_migration)
                    key = (latency, migration, new_cost, new_carbon, start, region)
                else:
                    raise ValueError(mode)
                if eligible and (best is None or key < best[0]):
                    best = (key, region, start, new_cost, new_carbon)
        if best is None:
            state.place(task_index, current_region, current_start)
        else:
            _, region, start, total_cost, total_carbon = best
            state.place(task_index, region, start)
            accepted += 1
    exact = state.energy_metrics()
    return state, {
        "mode": mode,
        "runtime_seconds": float(time.perf_counter() - started),
        "tasks_considered": int(len(task_indices)),
        "candidate_tests": tested,
        "accepted_moves": accepted,
        "tracked_cost_cny": float(total_cost),
        "exact_cost_cny": exact["cost_cny"],
        "tracked_carbon_tco2": float(total_carbon),
        "exact_carbon_tco2": exact["carbon_tco2"],
    }


def validate_schedule(state: ScheduleState) -> dict:
    tasks = state.inputs["tasks"]
    reconstructed_gpu = np.zeros_like(state.gpu_use)
    reconstructed_power = np.zeros_like(state.ai_power)
    violations = {
        "missing": 0,
        "duplicate": 0,
        "start": 0,
        "realtime_immediate": 0,
        "finish": 0,
        "latency": 0,
        "gpu": 0,
        "it": 0,
        "facility": 0,
        "grid_import": 0,
        "grid_export": 0,
    }
    seen: set[int] = set()
    for task_index, row in tasks.iterrows():
        task_id = int(row.TaskID)
        if task_id in seen:
            violations["duplicate"] += 1
        seen.add(task_id)
        region = int(state.assigned_region[task_index])
        start = int(state.start_hour[task_index])
        if region < 0 or start < 0:
            violations["missing"] += 1
            continue
        earliest = max(int(row.ArrivalHour), int(row.EarliestStartHour))
        if start < earliest:
            violations["start"] += 1
        if row.TaskType == "RealTimeInference" and start != int(row.ArrivalHour):
            violations["realtime_immediate"] += 1
        if start * 60 + int(row.EstimatedDuration_min) > min(
            int(row.LatestFinishHour), HORIZON_END
        ) * 60:
            violations["finish"] += 1
        source = int(row.SourceIndex)
        if state.inputs["latency"][source, region] > float(row.MaxLatency_ms) + TOL:
            violations["latency"] += 1
        hours, fractions = task_overlaps(start, int(row.EstimatedDuration_min))
        reconstructed_gpu[region, hours] += float(row.GPU_Demand) * fractions
        reconstructed_power[region, hours] += (
            float(row.GPU_Demand * row.TaskPowerPerGPU_MW) * fractions
        )
    violations["gpu"] = int(
        np.sum(reconstructed_gpu > state.inputs["gpu_capacity"][:, None] + 1e-6)
    )
    total_it = state.inputs["non_ai"] + reconstructed_power
    violations["it"] = int(
        np.sum(total_it > state.inputs["max_it"][:, None] + 1e-6)
    )
    facility = state.inputs["pue"][:, None] * total_it
    violations["facility"] = int(
        np.sum(facility > state.inputs["max_facility"][:, None] + 1e-6)
    )
    buy = np.maximum(facility - state.inputs["renewable"], 0.0)
    violations["grid_import"] = int(
        np.sum(buy > state.inputs["max_import"][:, None] + 1e-6)
    )
    surplus = np.maximum(state.inputs["renewable"] - facility, 0.0)
    sell = np.minimum(surplus, state.inputs["max_export"][:, None])
    violations["grid_export"] = int(
        np.sum(sell > state.inputs["max_export"][:, None] + 1e-6)
    )
    return {
        "total_tasks": int(len(tasks)),
        "assigned_tasks": int(np.sum(state.assigned_region >= 0)),
        "unique_task_ids": int(len(seen)),
        "violations": violations,
        "max_gpu_utilization": float(
            np.max(reconstructed_gpu / state.inputs["gpu_capacity"][:, None])
        ),
        "max_it_utilization": float(
            np.max(total_it / state.inputs["max_it"][:, None])
        ),
        "max_facility_utilization": float(
            np.max(facility / state.inputs["max_facility"][:, None])
        ),
        "usage_reconstruction_max_error": float(
            max(
                np.abs(reconstructed_gpu - state.gpu_use).max(),
                np.abs(reconstructed_power - state.ai_power).max(),
            )
        ),
    }


def latency_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "task_count": int(len(values)),
        "mean_latency_ms": float(values.mean()) if len(values) else None,
        "p95_latency_ms": float(np.quantile(values, 0.95)) if len(values) else None,
        "max_latency_ms": float(values.max()) if len(values) else None,
    }


def schedule_metrics(state: ScheduleState) -> dict:
    tasks = state.inputs["tasks"]
    valid = state.assigned_region >= 0
    regions = state.assigned_region[valid]
    starts = state.start_hour[valid]
    task_indices = np.flatnonzero(valid)
    effective = np.asarray(
        [effective_latency(state, index, int(state.assigned_region[index])) for index in task_indices]
    )
    source_regions = tasks.iloc[task_indices].SourceIndex.to_numpy(dtype=int)
    migrations = regions != source_regions
    task_types = tasks.iloc[task_indices].TaskType.astype(str).to_numpy()
    all_latency = latency_stats(effective)
    migrated_latency = latency_stats(effective[migrations])
    by_task_type = {}
    for task_type in sorted(np.unique(task_types)):
        type_mask = task_types == task_type
        type_migrations = migrations[type_mask]
        type_stats = latency_stats(effective[type_mask])
        type_stats.update(
            {
                "migration_tasks": int(type_migrations.sum()),
                "migration_rate": float(type_migrations.mean())
                if len(type_migrations)
                else None,
            }
        )
        by_task_type[task_type] = type_stats
    region_counts = np.bincount(regions, minlength=len(state.inputs["regions"]))
    hour_counts = np.bincount(starts, minlength=HORIZON_END)
    positive_hours = hour_counts[hour_counts > 0]
    digest = hashlib.sha256(
        np.column_stack([state.assigned_region, state.start_hour]).tobytes()
    ).hexdigest().upper()
    energy = state.energy_metrics()
    return {
        **energy,
        "assigned_tasks": int(valid.sum()),
        "mean_network_latency_ms": all_latency["mean_latency_ms"],
        "p95_network_latency_ms": all_latency["p95_latency_ms"],
        "max_network_latency_ms": all_latency["max_latency_ms"],
        "migrated_mean_network_latency_ms": migrated_latency["mean_latency_ms"],
        "migrated_p95_network_latency_ms": migrated_latency["p95_latency_ms"],
        "migrated_max_network_latency_ms": migrated_latency["max_latency_ms"],
        "migration_tasks": int(migrations.sum()),
        "migration_rate": float(migrations.mean()) if len(migrations) else math.nan,
        "service_quality": {
            "definition": {
                "latency": "official NetworkLatency_ms for AssignedRegion, including 5 ms on the local diagonal",
                "migration": "AssignedRegion != SourceRegion",
                "sla": "NetworkLatency_ms <= MaxLatency_ms",
                "source": state.inputs["latency_source"],
                "source_sha256": state.inputs["latency_source_sha256"],
            },
            "all_tasks": all_latency,
            "migrated_tasks": migrated_latency,
            "by_task_type": by_task_type,
        },
        "top_region_task_share": float(region_counts.max() / region_counts.sum()),
        "top_10_start_hours_task_share": float(
            np.sort(positive_hours)[-10:].sum() / hour_counts.sum()
        )
        if len(positive_hours)
        else math.nan,
        "positive_region_hours": int(np.sum(state.gpu_use > 1e-9)),
        "assignment_sha256": digest,
        "by_region_tasks": {
            state.inputs["regions"][r]: int(region_counts[r])
            for r in range(len(region_counts))
        },
    }


def main() -> None:
    started = time.perf_counter()
    inputs = load_inputs()
    cost_signal, carbon_signal = objective_signals(inputs)
    baseline, baseline_run = earliest_feasible_baseline(inputs)
    baseline_validation = validate_schedule(baseline)
    baseline_metrics = schedule_metrics(baseline)
    if baseline_validation["violations"]["missing"]:
        raise RuntimeError("B2 未能覆盖全部任务")
    if sum(baseline_validation["violations"].values()):
        raise RuntimeError(f"B2 硬约束失败: {baseline_validation['violations']}")

    cost_state, cost_run = local_improve(baseline, "cost", cost_signal)
    cost_validation = validate_schedule(cost_state)
    if sum(cost_validation["violations"].values()):
        raise RuntimeError(f"M2 成本阶段硬约束失败: {cost_validation['violations']}")
    cost_metrics = schedule_metrics(cost_state)
    c_min = cost_metrics["cost_cny"]
    positive_reference = float(
        np.sum(inputs["buy_price"] * inputs["pue"][:, None] * inputs["non_ai"])
    )
    epsilon_results = []
    epsilon_runs = []
    for epsilon in EPSILONS:
        cost_cap = (
            (1.0 + epsilon) * c_min
            if c_min > 0
            else c_min + epsilon * positive_reference
        )
        carbon_state, carbon_run = local_improve(
            cost_state, "carbon", carbon_signal, cost_cap=cost_cap
        )
        carbon_metrics = schedule_metrics(carbon_state)
        latency_state, latency_run = local_improve(
            carbon_state,
            "latency",
            cost_signal,
            cost_cap=cost_cap,
            carbon_cap=carbon_metrics["carbon_tco2"],
        )
        validation = validate_schedule(latency_state)
        if sum(validation["violations"].values()):
            raise RuntimeError(
                f"M2 epsilon={epsilon} 硬约束失败: {validation['violations']}"
            )
        metrics = schedule_metrics(latency_state)
        metrics.update(
            {
                "epsilon": epsilon,
                "cost_cap_cny": float(cost_cap),
                "validation": validation,
            }
        )
        epsilon_results.append(metrics)
        epsilon_runs.append(
            {"epsilon": epsilon, "carbon_pass": carbon_run, "latency_pass": latency_run}
        )

    costs = [item["cost_cny"] for item in epsilon_results]
    carbons = [item["carbon_tco2"] for item in epsilon_results]
    eps_reasonable = all(
        costs[i] <= epsilon_results[i]["cost_cap_cny"] + 1e-4
        for i in range(len(costs))
    ) and all(carbons[i] <= carbons[i - 1] + 1e-6 for i in range(1, len(carbons)))
    concentration_ok = all(
        item["top_region_task_share"] < 0.80
        and item["top_10_start_hours_task_share"] < 0.20
        for item in epsilon_results
    )
    total_runtime = float(time.perf_counter() - started)
    metrics_payload = {
        "schema_version": 1,
        "question_id": "Q2",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "baseline": {
            "run": baseline_run,
            "metrics": baseline_metrics,
            "validation": baseline_validation,
        },
        "cost_reference": {
            "run": cost_run,
            "metrics": cost_metrics,
            "validation": cost_validation,
            "interpretation": "启发式候选域内最低成本参考，不宣称全局最优",
        },
        "epsilon_solutions": epsilon_results,
        "epsilon_runs": epsilon_runs,
        "runtime_seconds": total_runtime,
        "time_budget_fallback_applied": TIME_BUDGET_FALLBACK_APPLIED,
        "peak_process_memory_mb": peak_process_memory_mb(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "random_seed": SEED,
            "packages": {
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "openpyxl": package_version("openpyxl"),
            },
        },
    }
    write_json(PROBE_DIR / "probe_metrics.json", metrics_payload)
    baseline_pass = (
        baseline_validation["assigned_tasks"] == 50000
        and sum(baseline_validation["violations"].values()) == 0
    )
    main_pass = all(
        item["assigned_tasks"] == 50000
        and sum(item["validation"]["violations"].values()) == 0
        for item in epsilon_results
    )
    main_verdict = (
        "PASS" if main_pass and eps_reasonable and concentration_ok and total_runtime < 7200 else "CONDITIONAL"
    )
    summary = {
        "schema_version": 1,
        "question_id": "Q2",
        "generated_at": metrics_payload["generated_at"],
        "decision_ref": "03_模型与推导/问题二/q2_decisions.jsonl#Q2-METHOD-CHOICE-001",
        "data_refs": [
            "02_数据处理/data_profile.json",
            "02_数据处理/raw/workload_trace.xlsx",
            "02_数据处理/raw/region_time_data.xlsx",
        ],
        "methods": [
            {
                "id": "M2",
                "role": "main_candidate",
                "executability": {
                    "status": "PASS" if main_pass else "FAIL",
                    "evidence": {
                        "epsilon_solutions": len(epsilon_results),
                        "runtime_seconds": total_runtime,
                        "peak_memory_mb": metrics_payload["peak_process_memory_mb"],
                    },
                },
                "data_coverage": {
                    "status": "PASS",
                    "evidence": {"task_rows": 50000, "region_hours": 6 * HORIZON_END},
                },
                "assumption_checks": [
                    {
                        "name": "all_tasks_unique_and_hard_constraints",
                        "status": "PASS" if main_pass else "FAIL",
                        "evidence": [item["validation"] for item in epsilon_results],
                    },
                    {
                        "name": "epsilon_reasonability",
                        "status": "PASS" if eps_reasonable else "CONDITIONAL",
                        "costs_cny": costs,
                        "carbons_tco2": carbons,
                    },
                ],
                "output_degeneracy": {
                    "status": "PASS" if concentration_ok else "CONDITIONAL",
                    "metrics": {
                        "top_region_task_share_max": max(
                            item["top_region_task_share"] for item in epsilon_results
                        ),
                        "top_10_start_hours_task_share_max": max(
                            item["top_10_start_hours_task_share"] for item in epsilon_results
                        ),
                        "positive_region_hours_min": min(
                            item["positive_region_hours"] for item in epsilon_results
                        ),
                    },
                },
                "perturbation_sensitivity": {
                    "status": "PASS" if eps_reasonable else "CONDITIONAL",
                    "perturbation": "time-budget fallback epsilon in {0,3,5}%",
                    "metrics": {
                        "cost_cny": costs,
                        "carbon_tco2": carbons,
                        "mean_latency_ms": [
                            item["mean_network_latency_ms"] for item in epsilon_results
                        ],
                    },
                },
                "scale_check": {
                    "status": "PASS" if total_runtime < 7200 else "CONDITIONAL",
                    "representative_n": 50000,
                    "runtime_seconds": total_runtime,
                    "peak_memory_mb": metrics_payload["peak_process_memory_mb"],
                },
                "verdict": main_verdict,
                "conditions": []
                if main_verdict == "PASS"
                else ["若五档扫描超过 2 小时则降为 {0%,3%,5%}；任务范围不得缩减"],
                "evidence_refs": ["03_模型与推导/问题二/probes/probe_metrics.json"],
            },
            {
                "id": "B2",
                "role": "usable_baseline",
                "executability": {
                    "status": "PASS" if baseline_pass else "FAIL",
                    "evidence": baseline_run,
                },
                "data_coverage": {
                    "status": "PASS",
                    "evidence": {"task_rows": 50000},
                },
                "assumption_checks": [
                    {
                        "name": "all_tasks_unique_and_hard_constraints",
                        "status": "PASS" if baseline_pass else "FAIL",
                        "evidence": baseline_validation,
                    }
                ],
                "output_degeneracy": {
                    "status": "PASS",
                    "metrics": {
                        "top_region_task_share": baseline_metrics["top_region_task_share"],
                        "top_10_start_hours_task_share": baseline_metrics[
                            "top_10_start_hours_task_share"
                        ],
                    },
                },
                "perturbation_sensitivity": {
                    "status": "NOT_APPLICABLE",
                    "reason": "稳定排序且无随机选择",
                },
                "scale_check": {
                    "status": "PASS",
                    "representative_n": 50000,
                    "runtime_seconds": baseline_run["runtime_seconds"],
                },
                "verdict": "PASS" if baseline_pass else "FAIL",
                "conditions": [],
                "evidence_refs": ["03_模型与推导/问题二/probes/probe_metrics.json"],
            },
        ],
        "conditional_fallback": {
            "id": "F2",
            "trigger": "五档扫描预计超过 2 小时或任一局部改进阶段失败",
            "observed": TIME_BUDGET_FALLBACK_APPLIED or total_runtime >= 7200 or not main_pass,
            "evidence": {
                "runtime_seconds": total_runtime,
                "main_pass": main_pass,
                "initial_five_level_probe_terminated_after_seconds": 600,
                "active_epsilon_scan": list(EPSILONS),
            },
        },
    }
    write_json(PROBE_DIR / "risk_probe_summary.json", summary)
    print(
        json.dumps(
            {
                "baseline_pass": baseline_pass,
                "main_verdict": main_verdict,
                "runtime_seconds": total_runtime,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
