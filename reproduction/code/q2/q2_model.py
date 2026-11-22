from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[2]
ROUND_ID = "round2_latency_fix_candidate"
PRIOR_ROUND_DIR = (
    ROOT / "05_结果与图表" / "intermediate" / "question2" / "experiments" / "round1"
)
ROUND_DIR = (
    ROOT
    / "05_结果与图表"
    / "intermediate"
    / "question2"
    / "experiments"
    / ROUND_ID
)
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
FIGURE_DIR = ROUND_DIR / "figures"
REPORT_DIR = ROUND_DIR / "reports"
CORE_PATH = ROOT / "03_模型与推导" / "问题二" / "probes" / "q2_risk_probe.py"
DECISION_ID = "Q2-METHOD-CHOICE-001"


def load_core():
    spec = importlib.util.spec_from_file_location("q2_validated_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法载入 Q2 已验证核心: {CORE_PATH}")
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def prepare_dirs() -> None:
    for path in (TABLE_DIR, METRIC_DIR, FIGURE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def assignments_frame(core, state) -> pd.DataFrame:
    tasks = state.inputs["tasks"]
    frame = tasks[
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
    frame["AssignedRegion"] = [
        state.inputs["regions"][int(region)] for region in state.assigned_region
    ]
    frame["StartHour"] = state.start_hour.astype(int)
    frame["FinishMinute"] = (
        frame.StartHour * 60 + frame.EstimatedDuration_min
    ).astype(int)
    frame["EffectiveLatency_ms"] = [
        core.effective_latency(state, index, int(state.assigned_region[index]))
        for index in range(len(frame))
    ]
    frame["Migrated"] = frame.AssignedRegion != frame.SourceRegion
    return frame


def region_hour_frame(state, scenario: str) -> pd.DataFrame:
    inputs = state.inputs
    facility = inputs["pue"][:, None] * (inputs["non_ai"] + state.ai_power)
    buy = np.maximum(facility - inputs["renewable"], 0.0)
    surplus = np.maximum(inputs["renewable"] - facility, 0.0)
    sell = np.minimum(surplus, inputs["max_export"][:, None])
    curtail = surplus - sell
    frames = []
    for r, region in enumerate(inputs["regions"]):
        frames.append(
            pd.DataFrame(
                {
                    "Scenario": scenario,
                    "Region": region,
                    "Hour": np.arange(core_horizon(state)),
                    "GPU_Use": state.gpu_use[r],
                    "AI_IT_Load_MW": state.ai_power[r],
                    "NonAI_IT_Load_MW": inputs["non_ai"][r],
                    "Facility_Load_MW": facility[r],
                    "AvailableRenewable_MW": inputs["renewable"][r],
                    "GridPurchase_MW": buy[r],
                    "GridSell_MW": sell[r],
                    "Curtailment_MW": curtail[r],
                    "CarbonEmission_tCO2": buy[r] * inputs["carbon_intensity"][r],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def core_horizon(state) -> int:
    return int(state.gpu_use.shape[1])


def save_assignments(core, state, filename: str) -> str:
    path = TABLE_DIR / filename
    assignments_frame(core, state).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.relative_to(ROOT)).replace("\\", "/")


def assignment_change_details(core, state, prior_filename: str) -> dict:
    prior_path = PRIOR_ROUND_DIR / "tables" / prior_filename
    if not prior_path.exists():
        return {"status": "NOT_AVAILABLE", "prior_file": str(prior_path)}
    before = pd.read_csv(
        prior_path,
        usecols=["TaskID", "AssignedRegion", "StartHour"],
    )
    after = assignments_frame(core, state)[["TaskID", "AssignedRegion", "StartHour"]]
    joined = before.merge(
        after,
        on="TaskID",
        how="outer",
        suffixes=("_before", "_after"),
        indicator=True,
        validate="one_to_one",
    )
    common = joined._merge == "both"
    region_changed = common & (
        joined.AssignedRegion_before != joined.AssignedRegion_after
    )
    start_changed = common & (joined.StartHour_before != joined.StartHour_after)
    any_changed = region_changed | start_changed | ~common
    return {
        "status": "COMPARED",
        "prior_file": str(prior_path.relative_to(ROOT)).replace("\\", "/"),
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "task_id_set_difference": int((~common).sum()),
        "assigned_region_changed_tasks": int(region_changed.sum()),
        "start_hour_changed_tasks": int(start_changed.sum()),
        "any_assignment_changed_tasks": int(any_changed.sum()),
        "sample_changed_task_ids": joined.loc[any_changed, "TaskID"].head(20).tolist(),
    }


def service_quality_rows(scenario: str, metrics: dict) -> list[dict]:
    quality = metrics["service_quality"]
    overall_migrations = metrics["migration_tasks"]
    overall_rate = metrics["migration_rate"]
    rows = [
        {
            "Scenario": scenario,
            "Scope": "all_tasks",
            "TaskType": "ALL",
            **quality["all_tasks"],
            "migration_tasks": overall_migrations,
            "migration_rate": overall_rate,
        },
        {
            "Scenario": scenario,
            "Scope": "migrated_tasks",
            "TaskType": "ALL",
            **quality["migrated_tasks"],
            "migration_tasks": overall_migrations,
            "migration_rate": overall_rate,
        },
    ]
    for task_type, stats in quality["by_task_type"].items():
        rows.append(
            {
                "Scenario": scenario,
                "Scope": "task_type",
                "TaskType": task_type,
                **stats,
            }
        )
    return rows


def main() -> None:
    prepare_dirs()
    tracemalloc.start()
    run_started = time.perf_counter()
    core = load_core()
    inputs = core.load_inputs()
    cost_signal, carbon_signal = core.objective_signals(inputs)

    baseline, baseline_run = core.earliest_feasible_baseline(inputs)
    baseline_validation = core.validate_schedule(baseline)
    baseline_metrics = core.schedule_metrics(baseline)
    if sum(baseline_validation["violations"].values()):
        raise RuntimeError(f"B2 复现失败: {baseline_validation['violations']}")

    cost_state, cost_run = core.local_improve(baseline, "cost", cost_signal)
    cost_validation = core.validate_schedule(cost_state)
    cost_metrics = core.schedule_metrics(cost_state)
    if sum(cost_validation["violations"].values()):
        raise RuntimeError(f"成本参考复现失败: {cost_validation['violations']}")

    c_min = cost_metrics["cost_cny"]
    positive_reference = float(
        np.sum(inputs["buy_price"] * inputs["pue"][:, None] * inputs["non_ai"])
    )
    epsilon_states = []
    epsilon_metrics = []
    epsilon_runs = []
    for epsilon in core.EPSILONS:
        cost_cap = (
            (1.0 + epsilon) * c_min
            if c_min > 0
            else c_min + epsilon * positive_reference
        )
        carbon_state, carbon_run = core.local_improve(
            cost_state, "carbon", carbon_signal, cost_cap=cost_cap
        )
        carbon_metrics = core.schedule_metrics(carbon_state)
        final_state, latency_run = core.local_improve(
            carbon_state,
            "latency",
            cost_signal,
            cost_cap=cost_cap,
            carbon_cap=carbon_metrics["carbon_tco2"],
        )
        validation = core.validate_schedule(final_state)
        if sum(validation["violations"].values()):
            raise RuntimeError(f"epsilon={epsilon} 复现失败: {validation['violations']}")
        metrics = core.schedule_metrics(final_state)
        metrics.update({"epsilon": epsilon, "cost_cap_cny": float(cost_cap)})
        epsilon_states.append(final_state)
        epsilon_metrics.append(metrics)
        epsilon_runs.append(
            {"epsilon": epsilon, "carbon_pass": carbon_run, "latency_pass": latency_run}
        )

    output_files = []
    output_files.append(save_assignments(core, baseline, "baseline_assignments.csv"))
    output_files.append(
        save_assignments(core, cost_state, "cost_reference_assignments.csv")
    )
    for state, metrics in zip(epsilon_states, epsilon_metrics, strict=True):
        level = int(round(metrics["epsilon"] * 100))
        output_files.append(
            save_assignments(core, state, f"epsilon_{level}_assignments.csv")
        )

    comparison = pd.DataFrame(epsilon_metrics)
    comparison_columns = [
        "epsilon",
        "cost_cap_cny",
        "cost_cny",
        "carbon_tco2",
        "renewable_utilization",
        "mean_network_latency_ms",
        "p95_network_latency_ms",
        "max_network_latency_ms",
        "migrated_mean_network_latency_ms",
        "migrated_p95_network_latency_ms",
        "migrated_max_network_latency_ms",
        "migration_tasks",
        "migration_rate",
        "top_region_task_share",
        "top_10_start_hours_task_share",
        "positive_region_hours",
        "assignment_sha256",
    ]
    comparison_path = TABLE_DIR / "q2_epsilon_comparison.csv"
    comparison[comparison_columns].to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )
    output_files.append(str(comparison_path.relative_to(ROOT)).replace("\\", "/"))

    quality_scenarios = [("B2", baseline_metrics)] + [
        (f"epsilon_{int(round(item['epsilon'] * 100))}", item)
        for item in epsilon_metrics
    ]
    quality_rows = []
    for scenario, metrics in quality_scenarios:
        quality_rows.extend(service_quality_rows(scenario, metrics))
    quality_path = TABLE_DIR / "q2_service_quality_metrics.csv"
    pd.DataFrame(quality_rows).to_csv(
        quality_path, index=False, encoding="utf-8-sig"
    )
    output_files.append(str(quality_path.relative_to(ROOT)).replace("\\", "/"))

    prior_reproducibility_path = PRIOR_ROUND_DIR / "metrics" / "reproducibility.json"
    prior_hashes = (
        read_json(prior_reproducibility_path).get("assignment_hashes", {})
        if prior_reproducibility_path.exists()
        else {}
    )
    current_hashes = {
        "baseline": baseline_metrics["assignment_sha256"],
        "cost_reference": cost_metrics["assignment_sha256"],
        **{
            f"epsilon_{int(round(item['epsilon'] * 100))}": item["assignment_sha256"]
            for item in epsilon_metrics
        },
    }
    state_by_hash_key = {
        "baseline": (baseline, "baseline_assignments.csv"),
        "cost_reference": (cost_state, "cost_reference_assignments.csv"),
        **{
            f"epsilon_{int(round(metrics['epsilon'] * 100))}": (
                state,
                f"epsilon_{int(round(metrics['epsilon'] * 100))}_assignments.csv",
            )
            for state, metrics in zip(epsilon_states, epsilon_metrics, strict=True)
        },
    }
    hash_comparisons = {}
    for key, after_hash in current_hashes.items():
        before_hash = prior_hashes.get(key)
        hash_comparisons[key] = {
            "before_sha256": before_hash,
            "after_sha256": after_hash,
            "assignment_unchanged": before_hash == after_hash,
        }
        if before_hash != after_hash:
            state, prior_filename = state_by_hash_key[key]
            hash_comparisons[key]["change_details"] = assignment_change_details(
                core, state, prior_filename
            )
    all_assignments_unchanged = bool(hash_comparisons) and all(
        item["assignment_unchanged"] for item in hash_comparisons.values()
    )
    hash_comparison_payload = {
        "schema_version": 1,
        "question_id": "Q2",
        "before_round": "round1",
        "after_round": ROUND_ID,
        "comparisons": hash_comparisons,
        "all_assignments_unchanged": all_assignments_unchanged,
        "classification": (
            "metric_definition_correction_only"
            if all_assignments_unchanged
            else "assignment_changed_requires_cause_analysis"
        ),
    }
    hash_comparison_path = METRIC_DIR / "assignment_hash_comparison.json"
    write_json(hash_comparison_path, hash_comparison_payload)
    output_files.append(
        str(hash_comparison_path.relative_to(ROOT)).replace("\\", "/")
    )

    region_hour = [region_hour_frame(baseline, "B2")]
    region_hour.append(region_hour_frame(cost_state, "C_min_H"))
    for state, metrics in zip(epsilon_states, epsilon_metrics, strict=True):
        region_hour.append(
            region_hour_frame(state, f"M2_epsilon_{int(round(metrics['epsilon'] * 100))}")
        )
    region_hour_path = TABLE_DIR / "q2_region_hour_metrics.csv"
    pd.concat(region_hour, ignore_index=True).to_csv(
        region_hour_path, index=False, encoding="utf-8-sig"
    )
    output_files.append(str(region_hour_path.relative_to(ROOT)).replace("\\", "/"))

    validation_payload = {
        "schema_version": 2,
        "question_id": "Q2",
        "latency_constraint": "NetworkLatency_ms <= MaxLatency_ms",
        "latency_source": inputs["latency_source"],
        "latency_source_sha256": inputs["latency_source_sha256"],
        "baseline": baseline_validation,
        "cost_reference": cost_validation,
        "epsilon": {
            str(metrics["epsilon"]): core.validate_schedule(state)
            for state, metrics in zip(epsilon_states, epsilon_metrics, strict=True)
        },
    }
    metrics_payload = {
        "schema_version": 2,
        "question_id": "Q2",
        "decision_id": DECISION_ID,
        "candidate_status": "intermediate_not_canonical",
        "latency_semantics_correction": {
            "source": inputs["latency_source"],
            "source_sha256": inputs["latency_source_sha256"],
            "local_diagonal_ms": 5.0,
            "effective_latency": "official matrix value for both local and cross-region assignments",
            "migration": "AssignedRegion != SourceRegion",
            "sla": "NetworkLatency_ms <= MaxLatency_ms",
        },
        "baseline": baseline_metrics,
        "cost_reference": {
            **cost_metrics,
            "interpretation": "启发式候选域内参考最小成本，不宣称全局最优",
        },
        "epsilon_solutions": epsilon_metrics,
        "stage_runs": {"baseline": baseline_run, "cost_reference": cost_run, "epsilon": epsilon_runs},
    }
    metrics_path = METRIC_DIR / "q2_metrics.json"
    validation_path = METRIC_DIR / "q2_constraint_validation.json"
    write_json(metrics_path, metrics_payload)
    write_json(validation_path, validation_payload)
    output_files.extend(
        [
            str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
            str(validation_path.relative_to(ROOT)).replace("\\", "/"),
        ]
    )

    figure_path = FIGURE_DIR / "q2_cost_carbon_tradeoff.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(comparison.cost_cny / 1e6, comparison.carbon_tco2, marker="o")
    for (cost_cny, carbon_tco2), group in comparison.groupby(
        ["cost_cny", "carbon_tco2"], sort=False
    ):
        label = ", ".join(f"{value:.0%}" for value in group.epsilon)
        ax.annotate(label, (cost_cny / 1e6, carbon_tco2), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Net electricity cost (million CNY)")
    ax.set_ylabel("Carbon emissions (tCO2)")
    ax.set_title("Q2 epsilon-constraint trade-off (intermediate)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    output_files.append(str(figure_path.relative_to(ROOT)).replace("\\", "/"))

    elapsed = float(time.perf_counter() - run_started)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    process_peak = core.peak_process_memory_mb()
    reproducibility = {
        "schema_version": 2,
        "random_seed": core.SEED,
        "deterministic_algorithm": True,
        "assignment_hashes": current_hashes,
        "assignment_hash_comparison": {
            "artifact": str(hash_comparison_path.relative_to(ROOT)).replace("\\", "/"),
            "all_assignments_unchanged": all_assignments_unchanged,
            "classification": hash_comparison_payload["classification"],
        },
        "official_latency_source": {
            "path": inputs["latency_source"],
            "sha256": inputs["latency_source_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "openpyxl": package_version("openpyxl"),
                "matplotlib": package_version("matplotlib"),
                "scipy": package_version("scipy"),
            },
        },
        "runtime_seconds": elapsed,
        "peak_process_working_set_mb": process_peak,
        "peak_python_traced_memory_mb": float(python_peak / 1024**2),
    }
    reproducibility_path = METRIC_DIR / "reproducibility.json"
    write_json(reproducibility_path, reproducibility)
    output_files.append(
        str(reproducibility_path.relative_to(ROOT)).replace("\\", "/")
    )

    all_zero_violations = all(
        sum(item["violations"].values()) == 0
        for item in [baseline_validation, cost_validation]
        + list(validation_payload["epsilon"].values())
    )
    run_summary = {
        "schema_version": 2,
        "question": "Q2",
        "round": ROUND_ID,
        "candidate_status": "intermediate_not_canonical",
        "implementation_target": "python",
        "random_seed": core.SEED,
        "approved_decision_id": DECISION_ID,
        "methods": [
            {
                "method_id": "B2",
                "role": "usable_baseline",
                "script": "04_代码实现/question2/q2_model.py",
                "status": "success",
                "execution_time_seconds": baseline_run["runtime_seconds"],
                "input_files": [
                    "02_数据处理/raw/workload_trace.xlsx",
                    "02_数据处理/raw/GPU_information.xlsx",
                    "00_题目与要求/附件数据/network_latency.xlsx",
                    "02_数据处理/raw/power_mapping.xlsx",
                    "02_数据处理/raw/region_time_data.xlsx",
                    "02_数据处理/raw/storage_information.xlsx",
                ],
                "output_files": [output_files[0]],
                "figure_files": [],
                "metrics_summary": baseline_metrics,
                "warnings": [],
                "errors": [],
            },
            {
                "method_id": "M2",
                "role": "main",
                "script": "04_代码实现/question2/q2_model.py",
                "status": "success" if all_zero_violations else "failed",
                "execution_time_seconds": elapsed,
                "input_files": [
                    "02_数据处理/raw/workload_trace.xlsx",
                    "00_题目与要求/附件数据/network_latency.xlsx",
                    "02_数据处理/raw/region_time_data.xlsx",
                ],
                "output_files": output_files,
                "figure_files": [str(figure_path.relative_to(ROOT)).replace("\\", "/")],
                "metrics_summary": {"cost_reference": cost_metrics, "epsilon": epsilon_metrics},
                "warnings": [
                    "C_min^H is negative; cost caps use the approved positive reference scale",
                    "epsilon 3% and 5% may coincide when the heuristic reaches zero grid-purchase carbon",
                    "This round is an intermediate latency-semantics correction candidate and does not replace frozen/final Q2 artifacts",
                ],
                "errors": [],
            },
        ],
        "comparison": {
            "baseline": baseline_metrics,
            "cost_reference": cost_metrics,
            "epsilon_solutions": epsilon_metrics,
        },
        "latency_semantics": metrics_payload["latency_semantics_correction"],
        "assignment_hash_comparison": hash_comparison_payload,
        "fallback_trigger": {
            "fallback_id": "F2",
            "condition": "five-level scan did not finish within 600 seconds in screening",
            "observed": True,
            "evidence": "03_模型与推导/问题二/probes/risk_probe_summary.json",
        },
        "environment": reproducibility["environment"],
    }
    write_json(ROUND_DIR / "run_summary.json", run_summary)
    candidate_manifest = {
        "schema_version": 1,
        "question_id": "Q2",
        "candidate_id": ROUND_ID,
        "status": "intermediate_not_canonical",
        "source_code_sha256": {
            "q2_risk_probe.py": file_sha256(CORE_PATH),
            "q2_model.py": file_sha256(Path(__file__).resolve()),
        },
        "official_latency_source": reproducibility["official_latency_source"],
        "assignment_hash_classification": hash_comparison_payload["classification"],
        "all_validated_schedules_zero_violations": all_zero_violations,
        "public_artifacts_modified": False,
        "submission_modified": False,
    }
    write_json(ROUND_DIR / "candidate_manifest.json", candidate_manifest)
    print(json.dumps({"status": "success", "runtime_seconds": elapsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
