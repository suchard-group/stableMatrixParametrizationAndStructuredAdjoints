#!/usr/bin/env python3
"""Summarize MAP fits from the K=5 simulation battery."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DIM = 5
PANEL_ORDER = {
    "Real/complex boundary": 0,
    "Non-orthogonal shear": 1,
    "Jordan(4,1) coupling": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("simulations/runs/final_battery"))
    parser.add_argument("--out-dir", type=Path, default=Path("simulations/results/raw/final_battery"))
    parser.add_argument("--relative-tol", type=float, default=1.0e-6)
    parser.add_argument("--write-raw-by-start", action="store_true")
    return parser.parse_args()


def read_manifest(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "inputs" / "task_manifest.tsv").open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_key_values(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def read_truth_matrix(path: Path) -> list[list[float]]:
    matrix = [[0.0 for _ in range(DIM)] for _ in range(DIM)]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            matrix[int(row["row"]) - 1][int(row["col"]) - 1] = float(row["value"])
    return matrix


def parse_beast_log(job_dir: Path) -> tuple[dict[str, str], str]:
    logs = sorted(path for path in job_dir.glob("*.log") if path.name != "input.log")
    if not logs:
        return {}, ""
    log_path = logs[0]
    header = None
    data = None
    for line in log_path.read_text(errors="ignore").splitlines():
        if line.startswith("state\tposterior"):
            header = line.rstrip("\n").split("\t")
            data = None
            continue
        if header is not None and line.strip() and not line.startswith("#"):
            data = line.rstrip("\n").split("\t")
    if header is None or data is None:
        return {}, str(log_path)
    return {name: data[index] for index, name in enumerate(header) if index < len(data)}, str(log_path)


def parse_optimizer_output(job_dir: Path, tol: float) -> dict[str, Any]:
    text = (job_dir / "output.txt").read_text(errors="ignore") if (job_dir / "output.txt").exists() else ""
    values = [
        (float(fx), float(gnorm))
        for fx, gnorm in re.findall(r"fx = ([^,]+), xnorm = [^,]+, gnorm = ([^,]+)", text)
    ]
    abs_delta = float("nan")
    rel_delta = float("nan")
    if len(values) >= 2:
        abs_delta = abs(values[-1][0] - values[-2][0])
        rel_delta = abs_delta / max(1.0, abs(values[-1][0]))
    returns = re.findall(r"LBFGS return: (\S+)", text)
    iterations = [int(value) for value in re.findall(r"Iteration\s+(\d+):", text)]
    return {
        "optimizer_return": returns[-1] if returns else "",
        "optimizer_iterations": max(iterations) if iterations else max(0, len(values) - 1),
        "optimizer_final_gnorm": values[-1][1] if values else float("nan"),
        "fx_count": len(values),
        "relative_logposterior_delta_last": rel_delta,
        "abs_logposterior_delta_last": abs_delta,
        "pass_relative_tol": rel_delta <= tol,
    }


def drift_rmse(estimate: list[list[float]], truth: list[list[float]]) -> float:
    return math.sqrt(sum((estimate[i][j] - truth[i][j]) ** 2 for i in range(DIM) for j in range(DIM)) / (DIM * DIM))


def quantile(values: list[float], prob: float) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * prob
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def summarize_raw(run_dir: Path, manifest: list[dict[str, str]], tol: float) -> list[dict[str, Any]]:
    truth_cache = {}
    rows = []
    for manifest_row in manifest:
        task_id = int(manifest_row["task_id"])
        job_dir = run_dir / "output" / f"job_{task_id:04d}"
        metadata = read_key_values(job_dir / "metadata.env")
        log_row, log_path = parse_beast_log(job_dir)
        estimate = [[float(log_row.get(f"ou.A[{i},{j}]", "nan")) for j in range(DIM)] for i in range(DIM)]
        truth_rel = manifest_row["truth_matrix_csv"]
        if truth_rel not in truth_cache:
            truth_cache[truth_rel] = read_truth_matrix(run_dir / "inputs" / truth_rel)
        finite_estimate = all(math.isfinite(value) for row in estimate for value in row)
        row: dict[str, Any] = dict(manifest_row)
        row.update({
            "exit_status": int(metadata.get("EXIT_STATUS", "999")),
            "runtime_seconds": float(metadata.get("RUNTIME_SECONDS", "nan")),
            "log_path": log_path,
            "posterior": float(log_row.get("posterior", "nan")),
            "likelihood": float(log_row.get("likelihood", "nan")),
            "prior": float(log_row.get("prior", "nan")),
            "map_drift_rmse": drift_rmse(estimate, truth_cache[truth_rel]) if finite_estimate else float("nan"),
        })
        row.update(parse_optimizer_output(job_dir, tol))
        rows.append(row)
    return rows


def select_best_starts(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_rows:
        key = (row["panel"], row["panel_key"], row["x_column"], row["x_value"], row["grid_tag"], row["replicate"], row["fit_family"])
        grouped[key].append(row)
    selected = []
    for group in grouped.values():
        best = max(group, key=lambda row: (row["likelihood"], row["posterior"]))
        best = dict(best)
        best["starts_considered"] = len(group)
        best["starts_exit0"] = sum(row["exit_status"] == 0 for row in group)
        best["starts_pass_relative_tol"] = sum(bool(row["pass_relative_tol"]) for row in group)
        best["selected_by"] = "max_likelihood_then_posterior"
        selected.append(best)
    selected.sort(key=lambda row: (PANEL_ORDER.get(row["panel"], 9), float(row["x_value"]), row["fit_family"], row["replicate"]))
    return selected


def summarize_grid(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in selected:
        key = (row["panel"], row["panel_key"], row["x_column"], row["x_value"], row["grid_tag"], row["fit_family"])
        grouped[key].append(row)
    out = []
    for (panel, panel_key, x_column, x_value, grid_tag, fit_family), group in grouped.items():
        rmses = [row["map_drift_rmse"] for row in group if math.isfinite(row["map_drift_rmse"])]
        likes = [row["likelihood"] for row in group if math.isfinite(row["likelihood"])]
        passed = sum(bool(row["pass_relative_tol"]) for row in group)
        out.append({
            "panel": panel,
            "panel_key": panel_key,
            "x_column": x_column,
            "x_value": x_value,
            "grid_tag": grid_tag,
            "fit_family": fit_family,
            "selected_replicates": len(group),
            "selected_exit0": sum(row["exit_status"] == 0 for row in group),
            "selected_pass_relative_tol": passed,
            "selected_success_rate_relative_tol": passed / len(group),
            "rmse_median": quantile(rmses, 0.5),
            "rmse_q1": quantile(rmses, 0.25),
            "rmse_q3": quantile(rmses, 0.75),
            "rmse_min": min(rmses) if rmses else float("nan"),
            "rmse_max": max(rmses) if rmses else float("nan"),
            "median_selected_likelihood": quantile(likes, 0.5),
        })
    out.sort(key=lambda row: (PANEL_ORDER.get(row["panel"], 9), float(row["x_value"]), row["fit_family"]))
    return out


def diagnostics(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["fit_family"])].append(row)
    out = []
    for (panel, fit_family), group in sorted(grouped.items()):
        passed = sum(bool(row["pass_relative_tol"]) for row in group)
        out.append({
            "panel": panel,
            "fit_family": fit_family,
            f"{prefix}_runs": len(group),
            f"{prefix}_exit0": sum(row["exit_status"] == 0 for row in group),
            f"{prefix}_pass_relative_tol": passed,
            f"{prefix}_pass_rate_relative_tol": passed / len(group),
        })
    return out


def write_table(path: Path, rows: list[dict[str, Any]], delimiter: str = "\t") -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_processed_summary(path: Path, grid: list[dict[str, Any]]) -> None:
    rows = []
    for row in grid:
        rows.append({
            "panel": row["panel"],
            "x_column": row["x_column"],
            "x_value": row["x_value"],
            "fit_family": row["fit_family"],
            "runs_total": row["selected_replicates"],
            "runs_completed": row["selected_exit0"],
            "runs_converged": row["selected_pass_relative_tol"],
            "completion_rate": row["selected_exit0"] / row["selected_replicates"],
            "optimization_success_rate": row["selected_success_rate_relative_tol"],
            "rmse_median_completed": row["rmse_median"],
            "rmse_q1_completed": row["rmse_q1"],
            "rmse_q3_completed": row["rmse_q3"],
            "rmse_min_completed": row["rmse_min"],
            "rmse_max_completed": row["rmse_max"],
            "median_selected_likelihood": row["median_selected_likelihood"],
        })
    write_table(path, rows, ",")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = summarize_raw(args.run_dir, read_manifest(args.run_dir), args.relative_tol)
    selected = select_best_starts(raw)
    grid = summarize_grid(selected)
    if args.write_raw_by_start:
        write_table(args.out_dir / "raw_by_start.tsv", raw)
    write_table(args.out_dir / "best_by_replicate.tsv", selected)
    write_table(args.out_dir / "grid_summary_best_by_likelihood.tsv", grid)
    write_table(args.out_dir / "selected_best_likelihood_diagnostics.csv", diagnostics(selected, "selected"), ",")
    write_table(args.out_dir / "raw_start_diagnostics.csv", diagnostics(raw, "raw"), ",")
    write_processed_summary(args.out_dir / "summary_for_plot_best_likelihood.csv", grid)
    summary = {
        "relative_tol": args.relative_tol,
        "raw_tasks": len(raw),
        "raw_exit_status_counts": dict(collections.Counter(str(row["exit_status"]) for row in raw)),
        "raw_pass_relative_tol": sum(bool(row["pass_relative_tol"]) for row in raw),
        "selected_rows": len(selected),
        "selected_pass_relative_tol": sum(bool(row["pass_relative_tol"]) for row in selected),
        "selected_failures": [
            row["task_id"] for row in selected
            if row["exit_status"] != 0 or not bool(row["pass_relative_tol"])
        ],
        "grid_rows": len(grid),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
