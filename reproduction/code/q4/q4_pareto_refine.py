from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import q4_model as q4


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


MIGRATION_CAPS = (50, 100, 250, 500, 1000, 2200)
WAIT_CAP = 0.02
PRIOR_ROUND_DIR = (
    q4.ROOT
    / "05_结果与图表"
    / "intermediate"
    / "question4"
    / "experiments"
    / "round1"
)


DOMINANCE_OBJECTIVES = (
    "cost_cny",
    "carbon_tco2",
    "max_regional_peak_net_purchase_mw",
    "mean_wait_ratio",
    "cvar95_wait_ratio",
    "migration_tasks",
    "mean_network_latency_ms",
)


def dominance_pairs(rows: list[dict]) -> list[dict]:
    pairs = []
    for row in rows:
        for other in rows:
            if row["solution_id"] == other["solution_id"]:
                continue
            weak = all(
                float(other[key]) <= float(row[key]) + 1e-9
                for key in DOMINANCE_OBJECTIVES
            )
            strict = any(
                float(other[key]) < float(row[key]) - 1e-9
                for key in DOMINANCE_OBJECTIVES
            )
            if weak and strict:
                pairs.append(
                    {
                        "dominated": row["solution_id"],
                        "dominator": other["solution_id"],
                    }
                )
    return pairs


def migration_pareto(rows: list[dict]) -> list[dict]:
    unique = []
    for row in rows:
        duplicate = any(
            abs(float(row["cost_cny"]) - float(other["cost_cny"])) <= 1e-3
            and row["assignment_sha256"] == other["assignment_sha256"]
            for other in unique
        )
        if not duplicate:
            unique.append(row)
    kept = []
    for i, row in enumerate(unique):
        dominated = False
        for j, other in enumerate(unique):
            if i == j:
                continue
            weak = all(
                float(other[key]) <= float(row[key]) + 1e-9
                for key in DOMINANCE_OBJECTIVES
            )
            strict = any(
                float(other[key]) < float(row[key]) - 1e-9
                for key in DOMINANCE_OBJECTIVES
            )
            if weak and strict:
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return kept


def compromise(rows: list[dict]) -> dict:
    minimize = (
        "cost_cny",
        "carbon_tco2",
        "max_regional_peak_net_purchase_mw",
        "mean_wait_ratio",
        "cvar95_wait_ratio",
        "migration_tasks",
        "mean_network_latency_ms",
    )
    maximize = ("renewable_utilization",)
    matrix = np.asarray(
        [
            [float(row[key]) for key in minimize]
            + [-float(row[key]) for key in maximize]
            for row in rows
        ]
    )
    low = matrix.min(axis=0)
    high = matrix.max(axis=0)
    scale = np.where(high > low, high - low, 1.0)
    distance = np.sqrt(np.sum(((matrix - low) / scale) ** 2, axis=1))
    result = dict(rows[int(np.argmin(distance))])
    result["ideal_distance"] = float(distance.min())
    result["distance_dimensions"] = list(minimize + maximize)
    return result


def main() -> None:
    started = time.perf_counter()
    q2 = q4.load_q2_core()
    q3 = q4.load_q3_core()
    q2_inputs = q2.load_inputs()
    q3_inputs = q3.load_inputs()
    final_q2 = q4.ROOT / "05_结果与图表" / "final" / "question2" / "tables"
    seed_state, _, _ = q4.load_state_from_assignments(
        q2, q2_inputs, final_q2 / "baseline_assignments.csv"
    )

    existing_metrics_path = q4.METRIC_DIR / "q4_metrics.json"
    existing = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
    comparison = existing["comparison"]

    dispatch = pd.read_csv(q4.TABLE_DIR / "q4_energy_dispatch_main.csv")
    b2 = dispatch[dispatch.Solution == "B2"].copy()
    b2 = b2.sort_values(["Region", "Hour"])
    shape = (len(q3_inputs["regions"]), len(q3_inputs["hours"]))
    buy = b2.GridPurchase_MW.to_numpy(dtype=float).reshape(shape)
    curtail = b2.Curtailment_MW.to_numpy(dtype=float).reshape(shape)
    signal = np.where(
        buy > q4.TOL,
        q3_inputs["buy_price"],
        np.where(curtail > q4.TOL, 0.0, q3_inputs["sell_price"]),
    )

    boundary = dict(comparison["B2"])
    boundary.update(
        {
            "solution_id": "Joint_m0000",
            "role": "joint_boundary_migration_cap_0",
            "migration_cap": 0,
            "wait_cap": WAIT_CAP,
            "carbon_gap_fraction": 1.0,
        }
    )
    candidates = [boundary]
    results = {"Joint_m0000": (boundary, None, None, None)}
    refine_runs = {}

    for migration_cap in MIGRATION_CAPS:
        state, task_run = q4.joint_improve(
            q2,
            seed_state,
            signal[:, : q4.TASK_HOURS],
            WAIT_CAP,
            q4.JOINT_TASK_LIMIT,
            migration_cap=migration_cap,
        )
        task = q4.task_metrics(q2, state)
        inputs = q4.scenario_inputs(
            q2_inputs,
            q3_inputs,
            state,
            {
                "scenario_id": "original",
                "price_spread_factor": 1.0,
                "renewable_scale": 1.0,
            },
        )
        model = q4.ProvenanceEnergyModel(inputs)
        energy_results, energy_run = model.solve_frontier([1.0])
        energy = energy_results[0]
        label = f"Joint_m{migration_cap:04d}"
        row = q4.combined_metrics(
            label,
            "migration_epsilon_pareto_candidate",
            task,
            energy.metrics,
            {
                "migration_cap": migration_cap,
                "wait_cap": WAIT_CAP,
                "carbon_gap_fraction": 1.0,
            },
        )
        candidates.append(row)
        results[label] = (row, energy, inputs, state)
        refine_runs[label] = {"task": task_run, "energy": energy_run}

    pareto = migration_pareto(candidates)
    pareto_warning = None
    if len(pareto) < 5:
        pareto_warning = (
            f"修正区域峰值后仅保留 {len(pareto)} 个非支配候选；"
            "全部候选仍输出，但不得用被支配候选补足前沿"
        )
    chosen = compromise(pareto)
    chosen_label = chosen["solution_id"]
    chosen_row, chosen_energy, chosen_inputs, chosen_state = results[chosen_label]
    if chosen_label == "Joint_m0000":
        # 边界解与 B2 相同，不需要重新写能源表；任务表使用 B0。
        b0_path = q4.TABLE_DIR / "q4_task_assignments_B0.csv"
        chosen_tasks = pd.read_csv(b0_path)
        chosen_task_metrics = {
            key: chosen_row[key]
            for key in chosen_row
            if key.startswith("mean_")
            or key.startswith("cvar")
            or key.startswith("p")
            or key.startswith("max_wait")
            or key in ("migration_tasks", "migration_rate", "assignment_sha256")
        }
    else:
        chosen_tasks = q4.assignments_frame(chosen_state)
        chosen_task_metrics = q4.task_metrics(q2, chosen_state)

    for row in candidates:
        label = row["solution_id"]
        if label == "Joint_m0000":
            continue
        _, energy, inputs, state = results[label]
        q4.assignments_frame(state).to_csv(
            q4.TABLE_DIR / f"q4_task_assignments_{label}.csv", index=False
        )
    energy_frames = []
    for row in candidates:
        label = row["solution_id"]
        if label == "Joint_m0000":
            continue
        _, energy, inputs, _ = results[label]
        energy_frames.append(q4.dispatch_frame(label, inputs, energy))
    if energy_frames:
        pd.concat(energy_frames, ignore_index=True).to_csv(
            q4.TABLE_DIR / "q4_energy_dispatch_pareto_refined.csv", index=False
        )

    # 新折中方案的 A/B/C 场景重算。
    scenario_specs = [
        {"scenario_id": "A_price_1p25", "price_spread_factor": 1.25, "renewable_scale": 1.0, "carbon_gap_fraction": 1.0},
        {"scenario_id": "A_renewable_low_0p8", "price_spread_factor": 1.0, "renewable_scale": 0.8, "carbon_gap_fraction": 1.0},
        {"scenario_id": "A_renewable_high_1p2", "price_spread_factor": 1.0, "renewable_scale": 1.2, "carbon_gap_fraction": 1.0},
        {"scenario_id": "B_conservative_low", "price_spread_factor": 1.15, "renewable_scale": 0.9, "carbon_gap_fraction": 0.33},
        {"scenario_id": "C_stress_low", "price_spread_factor": 1.50, "renewable_scale": 0.7, "carbon_gap_fraction": 0.0},
        {"scenario_id": "C_stress_high", "price_spread_factor": 1.50, "renewable_scale": 1.3, "carbon_gap_fraction": 0.0},
    ]
    scenario_rows = []
    scenario_frames = []
    scenario_runs = {}
    scenario_state = chosen_state
    if chosen_label == "Joint_m0000":
        scenario_state, _, _ = q4.local_only_baseline(q2, q2_inputs)[0], None, None
    scenario_task = q4.task_metrics(q2, scenario_state)
    b0_state, _ = q4.local_only_baseline(q2, q2_inputs)
    for spec in scenario_specs:
        inputs = q4.scenario_inputs(q2_inputs, q3_inputs, scenario_state, spec)
        baseline_inputs = q4.scenario_inputs(q2_inputs, q3_inputs, b0_state, spec)
        baseline = q4.no_storage_result(baseline_inputs)
        try:
            model = q4.ProvenanceEnergyModel(inputs)
            energy_results, energy_run = model.solve_frontier(
                [float(spec["carbon_gap_fraction"])]
            )
            energy = energy_results[0]
            row = q4.combined_metrics(
                spec["scenario_id"],
                "scenario_joint_representative",
                scenario_task,
                energy.metrics,
                {
                    **spec,
                    "baseline_cost_cny": baseline.metrics["cost_cny"],
                    "baseline_carbon_tco2": baseline.metrics["carbon_tco2"],
                    "scenario_status": "feasible",
                    **inputs["scenario_hashes"],
                },
            )
            scenario_rows.append(row)
            scenario_frames.append(q4.dispatch_frame(spec["scenario_id"], inputs, energy))
            scenario_runs[spec["scenario_id"]] = energy_run
        except RuntimeError as error:
            diagnostic = {
                "scenario_id": spec["scenario_id"],
                "scenario_status": "infeasible_or_solver_failed",
                "error": str(error),
                "diagnosis": {
                    "task_constraints": scenario_task["task_hard_constraint_violations"],
                    "carbon_constraint_fraction": spec["carbon_gap_fraction"],
                    "storage_model": "SOC/power/terminal constraints active",
                    "grid_boundary": "per-region import/export caps active",
                },
            }
            scenario_rows.append(diagnostic)
            scenario_runs[spec["scenario_id"]] = diagnostic

    if scenario_frames:
        pd.concat(scenario_frames, ignore_index=True).to_csv(
            q4.TABLE_DIR / "q4_energy_dispatch_scenarios.csv", index=False
        )
    pd.DataFrame(scenario_rows).to_csv(q4.TABLE_DIR / "q4_scenarios.csv", index=False)
    pd.DataFrame(pareto).to_csv(q4.TABLE_DIR / "q4_pareto.csv", index=False)

    candidate_rows = [comparison[key] for key in ("B0", "B1", "B2")] + candidates
    candidate_columns = [
        "solution_id",
        "role",
        "cost_cny",
        "carbon_tco2",
        "renewable_utilization",
        "grid_purchase_mwh",
        "grid_sell_mwh",
        "RegionA_peak_net_purchase_mw",
        "RegionB_peak_net_purchase_mw",
        "RegionC_peak_net_purchase_mw",
        "RegionD_peak_net_purchase_mw",
        "RegionE_peak_net_purchase_mw",
        "RegionF_peak_net_purchase_mw",
        "max_regional_peak_net_purchase_mw",
        "mean_network_latency_ms",
        "p95_network_latency_ms",
        "max_network_latency_ms",
        "migrated_mean_network_latency_ms",
        "migrated_p95_network_latency_ms",
        "migrated_max_network_latency_ms",
        "ai_training_mean_network_latency_ms",
        "ai_training_p95_network_latency_ms",
        "ai_training_max_network_latency_ms",
        "batch_inference_mean_network_latency_ms",
        "batch_inference_p95_network_latency_ms",
        "batch_inference_max_network_latency_ms",
        "real_time_inference_mean_network_latency_ms",
        "real_time_inference_p95_network_latency_ms",
        "real_time_inference_max_network_latency_ms",
        "mean_wait_ratio",
        "p95_wait_ratio",
        "max_wait_ratio",
        "cvar95_wait_ratio",
        "migration_tasks",
        "migration_rate",
        "task_completion_rate",
        "task_hard_constraint_violations",
    ]
    pd.DataFrame(candidate_rows)[candidate_columns].to_csv(
        q4.TABLE_DIR / "q4_candidate_metrics.csv", index=False
    )

    region_peak_rows = []
    for row in candidate_rows:
        for region in q3_inputs["regions"]:
            region_peak_rows.append(
                {
                    "solution_id": row["solution_id"],
                    "Region": region,
                    "regional_peak_net_purchase_mw": row[
                        f"{region}_peak_net_purchase_mw"
                    ],
                    "max_regional_peak_net_purchase_mw": row[
                        "max_regional_peak_net_purchase_mw"
                    ],
                }
            )
    pd.DataFrame(region_peak_rows).to_csv(
        q4.TABLE_DIR / "q4_region_peak_metrics.csv", index=False
    )

    latency_rows = []
    latency_scopes = {
        "all_tasks": "",
        "migrated_tasks": "migrated_",
        "AITraining": "ai_training_",
        "BatchInference": "batch_inference_",
        "RealTimeInference": "real_time_inference_",
    }
    for row in candidate_rows:
        for scope, prefix in latency_scopes.items():
            latency_rows.append(
                {
                    "solution_id": row["solution_id"],
                    "scope": scope,
                    "mean_network_latency_ms": row[f"{prefix}mean_network_latency_ms"],
                    "p95_network_latency_ms": row[f"{prefix}p95_network_latency_ms"],
                    "max_network_latency_ms": row[f"{prefix}max_network_latency_ms"],
                    "migration_tasks": row["migration_tasks"],
                    "migration_rate": row["migration_rate"],
                }
            )
    pd.DataFrame(latency_rows).to_csv(
        q4.TABLE_DIR / "q4_latency_metrics.csv", index=False
    )

    prior_pareto_path = PRIOR_ROUND_DIR / "tables" / "q4_pareto.csv"
    prior_front = (
        pd.read_csv(prior_pareto_path).solution_id.astype(str).tolist()
        if prior_pareto_path.exists()
        else []
    )
    current_front = [row["solution_id"] for row in pareto]
    joint_m0500 = next(row for row in candidates if row["solution_id"] == "Joint_m0500")
    selection_checks = {
        "candidate_is_nondominated": "Joint_m0500" in current_front,
        "positive_additional_cost_benefit_vs_B2": float(comparison["B2"]["cost_cny"])
        - float(joint_m0500["cost_cny"])
        > 0.0,
        "migration_budget_is_500": int(joint_m0500["migration_tasks"]) == 500,
        "migration_rate_is_1_percent": math.isclose(
            float(joint_m0500["migration_rate"]), 0.01, abs_tol=1e-12
        ),
        "all_tasks_complete": math.isclose(
            float(joint_m0500["task_completion_rate"]), 1.0, abs_tol=1e-12
        ),
        "zero_hard_constraint_violations": int(
            joint_m0500["task_hard_constraint_violations"]
        )
        == 0,
    }
    dominance_comparison = {
        "schema_version": 1,
        "question_id": "Q4",
        "before_round": "round1",
        "after_round": q4.ROUND_ID,
        "dominance_objectives_after_correction": list(DOMINANCE_OBJECTIVES),
        "prior_front": prior_front,
        "current_front": current_front,
        "added_to_front": sorted(set(current_front) - set(prior_front)),
        "removed_from_front": sorted(set(prior_front) - set(current_front)),
        "relationship_changed": set(current_front) != set(prior_front),
        "current_dominance_pairs": dominance_pairs(candidates),
        "joint_m0500_selection_checks": selection_checks,
        "joint_m0500_satisfies_recorded_selection_conditions": all(
            selection_checks.values()
        ),
        "freeze_action": (
            "PENDING_HUMAN_REFREEZE"
            if all(selection_checks.values())
            else "STOP_FREEZE_REQUEST_NEW_HUMAN_SELECTION"
        ),
    }
    q4.write_json(
        q4.METRIC_DIR / "q4_dominance_comparison.json", dominance_comparison
    )

    comparison["Joint"] = {**chosen_row, "role": "provisional_compromise_from_refined_pareto"}
    comparison_frame = pd.DataFrame(comparison.values())
    comparison_frame.to_csv(q4.TABLE_DIR / "q4_comparison.csv", index=False)
    gains = q4.gain_decomposition(comparison)
    gains.to_csv(q4.TABLE_DIR / "q4_gain_decomposition.csv", index=False)

    existing["initial_pareto_candidates"] = existing.pop("pareto_candidates", [])
    existing["initial_pareto_nondominated"] = existing.pop("pareto_nondominated", [])
    existing["pareto_refinement"] = {
        "reason": "初始等待上限扫描退化为单点；按已批准网络时延/迁移维度增加迁移预算 epsilon",
        "migration_caps": [0, *MIGRATION_CAPS],
        "wait_cap": WAIT_CAP,
        "runs": refine_runs,
    }
    existing["pareto_candidates"] = candidates
    existing["pareto_nondominated"] = pareto
    existing["pareto_count"] = len(pareto)
    existing["pareto_warning"] = pareto_warning
    existing["dominance_comparison"] = dominance_comparison
    existing["provisional_ideal_distance_candidate"] = chosen
    existing["comparison"] = comparison
    existing["scenario_results"] = scenario_rows
    existing["gain_decomposition"] = gains.to_dict(orient="records")
    existing["runs"]["pareto_refinement"] = refine_runs
    existing["runs"]["scenarios_refined_compromise"] = scenario_runs
    q4.write_json(existing_metrics_path, existing)

    validation_path = q4.METRIC_DIR / "q4_constraint_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["Joint"] = scenario_task["validation"]
    validation["candidate_all"] = {
        row["solution_id"]: {
            "task_hard_constraint_violations": row["task_hard_constraint_violations"],
            "max_energy_balance_residual_mw": row["max_energy_balance_residual_mw"],
            "max_soc_recurrence_residual_mwh": row["max_soc_recurrence_residual_mwh"],
            "terminal_soc_shortfall_mwh": row["terminal_soc_shortfall_mwh"],
            "simultaneous_charge_discharge_cells": row[
                "simultaneous_charge_discharge_cells"
            ],
        }
        for row in candidates
    }
    q4.write_json(validation_path, validation)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    frame = pd.DataFrame(pareto)
    axes[0].plot(frame.migration_tasks, frame.cost_cny / 1e6, marker="o")
    axes[0].set_xlabel("迁移任务数")
    axes[0].set_ylabel("系统净成本（百万元）")
    axes[0].set_title("Q4 迁移预算 Pareto")
    axes[1].plot(frame.mean_network_latency_ms, frame.mean_wait_ratio, marker="o")
    axes[1].set_xlabel("平均网络时延（ms）")
    axes[1].set_ylabel("平均归一化等待比例")
    axes[1].set_title("Q4 服务代价")
    fig.tight_layout()
    fig.savefig(q4.FIGURE_DIR / "q4_pareto_refined_diagnostic.png", dpi=180)
    plt.close(fig)

    run_summary_path = q4.ROUND_DIR / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_summary["status"] = "success_waiting_human_result_judgment"
    run_summary["pareto_count"] = len(pareto)
    run_summary["provisional_compromise"] = chosen_label
    run_summary["comparison"] = comparison
    run_summary["dominance_comparison"] = dominance_comparison
    run_summary["freeze_action"] = dominance_comparison["freeze_action"]
    run_summary["scenario_count"] = len(scenario_rows)
    run_summary["pareto_refinement_runtime_seconds"] = time.perf_counter() - started
    run_summary["warnings"] = [
        warning
        for warning in run_summary.get("warnings", [])
        if "Pareto" not in warning
    ] + [
        "初始等待扫描退化为单点，已按迁移预算 epsilon 精炼为真实非支配前沿",
        "理想点距离候选仅供人工选择，不构成冻结决定",
    ] + ([pareto_warning] if pareto_warning else [])
    q4.write_json(run_summary_path, run_summary)

    reproducibility_path = q4.METRIC_DIR / "reproducibility.json"
    reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    reproducibility["pareto_refinement_runtime_seconds"] = time.perf_counter() - started
    reproducibility["migration_caps"] = [0, *MIGRATION_CAPS]
    reproducibility["peak_process_working_set_mb_after_refinement"] = q4.peak_memory_mb()
    q4.write_json(reproducibility_path, reproducibility)

    print(
        json.dumps(
            {
                "status": "success_waiting_human_result_judgment",
                "pareto_count": len(pareto),
                "provisional_compromise": chosen_label,
                "runtime_seconds": time.perf_counter() - started,
                "pareto": [
                    {
                        "solution_id": row["solution_id"],
                        "cost_cny": row["cost_cny"],
                        "migration_tasks": row["migration_tasks"],
                        "mean_wait_ratio": row["mean_wait_ratio"],
                    }
                    for row in pareto
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
