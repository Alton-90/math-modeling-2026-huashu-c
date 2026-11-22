from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = (
    ROOT
    / "05_结果与图表/intermediate/question4/experiments/"
    "round2_peak_latency_fix_candidate/tables/q4_candidate_metrics.csv"
)
OUTPUT_PATH = (
    ROOT
    / "05_结果与图表/intermediate/question4/experiments/"
    "round2_peak_latency_fix_candidate/metrics/q4_selection_rule_verification.json"
)
EXPECTED_JOINT_IDS = {
    "Joint_m0000",
    "Joint_m0050",
    "Joint_m0100",
    "Joint_m0250",
    "Joint_m0500",
    "Joint_m1000",
    "Joint_m2200",
}
EXPECTED_SELECTION = "Joint_m0500"
THRESHOLD_RATIO = Decimal("0.85")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def main() -> None:
    with TABLE_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    b2_rows = [row for row in rows if row["solution_id"] == "B2"]
    if len(b2_rows) != 1:
        raise RuntimeError(f"Expected one B2 row, found {len(b2_rows)}")

    joint_rows = [row for row in rows if row["solution_id"].startswith("Joint_")]
    joint_ids = {row["solution_id"] for row in joint_rows}
    if joint_ids != EXPECTED_JOINT_IDS:
        raise RuntimeError(
            f"Joint candidate set mismatch: {sorted(joint_ids)} != "
            f"{sorted(EXPECTED_JOINT_IDS)}"
        )

    b2_cost = Decimal(b2_rows[0]["cost_cny"])
    candidates: list[dict[str, object]] = []
    for row in joint_rows:
        cost = Decimal(row["cost_cny"])
        improvement = b2_cost - cost
        candidates.append(
            {
                "solution_id": row["solution_id"],
                "cost_cny": cost,
                "additional_cost_improvement_vs_B2_cny": improvement,
                "migration_tasks": int(row["migration_tasks"]),
                "migration_rate": Decimal(row["migration_rate"]),
            }
        )

    maximum_improvement = max(
        item["additional_cost_improvement_vs_B2_cny"] for item in candidates
    )
    threshold_improvement = maximum_improvement * THRESHOLD_RATIO
    for item in candidates:
        improvement = item["additional_cost_improvement_vs_B2_cny"]
        ratio = improvement / maximum_improvement
        item["share_of_maximum_improvement"] = ratio
        item["share_of_maximum_improvement_percent"] = ratio * Decimal("100")
        item["meets_85_percent_threshold"] = improvement >= threshold_improvement

    candidates.sort(key=lambda item: item["migration_tasks"])
    eligible = [item for item in candidates if item["meets_85_percent_threshold"]]
    selected = min(eligible, key=lambda item: item["migration_tasks"])
    if selected["solution_id"] != EXPECTED_SELECTION:
        raise RuntimeError(
            f"Selection rule chose {selected['solution_id']}, not {EXPECTED_SELECTION}"
        )

    def serialize_candidate(item: dict[str, object]) -> dict[str, object]:
        return {
            "solution_id": item["solution_id"],
            "cost_cny": decimal_text(item["cost_cny"]),
            "additional_cost_improvement_vs_B2_cny": decimal_text(
                item["additional_cost_improvement_vs_B2_cny"]
            ),
            "share_of_maximum_improvement": decimal_text(
                item["share_of_maximum_improvement"]
            ),
            "share_of_maximum_improvement_percent": decimal_text(
                item["share_of_maximum_improvement_percent"]
            ),
            "migration_tasks": item["migration_tasks"],
            "migration_rate": decimal_text(item["migration_rate"]),
            "meets_85_percent_threshold": item["meets_85_percent_threshold"],
        }

    result = {
        "schema_version": 1,
        "question_id": "Q4",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "verification_status": "PASSED",
        "source_table": str(TABLE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_table_sha256": sha256(TABLE_PATH),
        "baseline": "B2",
        "baseline_cost_cny": decimal_text(b2_cost),
        "candidate_count": len(candidates),
        "rule": {
            "threshold_share_of_maximum_improvement": decimal_text(THRESHOLD_RATIO),
            "tie_breaker": "minimum migration_tasks",
            "ordering": "ascending migration_tasks",
        },
        "maximum_additional_cost_improvement_vs_B2_cny": decimal_text(
            maximum_improvement
        ),
        "threshold_additional_cost_improvement_cny": decimal_text(
            threshold_improvement
        ),
        "candidates": [serialize_candidate(item) for item in candidates],
        "selected_solution": selected["solution_id"],
        "selection_is_expected": True,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
