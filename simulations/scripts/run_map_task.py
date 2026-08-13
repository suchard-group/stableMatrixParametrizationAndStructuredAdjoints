#!/usr/bin/env python3
"""Run one generated BEAST MAP task from a simulation manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("simulations/runs/final_battery"))
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--beast-jar", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--java-opts", default="-Dparsers=development")
    return parser.parse_args()


def read_manifest(run_dir: Path) -> dict[int, dict[str, str]]:
    path = run_dir / "inputs" / "task_manifest.tsv"
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {int(row["task_id"]): row for row in rows}


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.run_dir)
    if args.task not in manifest:
        raise SystemExit(f"Task {args.task} not found in {args.run_dir / 'inputs/task_manifest.tsv'}")
    row = manifest[args.task]
    source_xml = args.run_dir / "inputs" / row["xml_template"]
    if not source_xml.exists():
        raise SystemExit(f"Missing XML: {source_xml}")
    if not args.beast_jar.exists():
        raise SystemExit(f"Missing BEAST jar: {args.beast_jar}")

    job_dir = args.run_dir / "output" / f"job_{args.task:04d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    xml_copy = job_dir / source_xml.name
    shutil.copy2(source_xml, xml_copy)

    command = [args.java, *args.java_opts.split(), "-jar", str(args.beast_jar), "-seed", row["beast_seed"], "-overwrite", str(xml_copy)]
    start = time.time()
    with (job_dir / "output.txt").open("w") as handle:
        completed = subprocess.run(command, cwd=job_dir, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.time() - start

    (job_dir / "metadata.env").write_text(
        "\n".join([
            f"TASK_ID={args.task:04d}",
            f"XML_TEMPLATE={row['xml_template']}",
            f"BEAST_SEED={row['beast_seed']}",
            f"EXIT_STATUS={completed.returncode}",
            f"RUNTIME_SECONDS={elapsed:.6f}",
            f"FIT_FAMILY={row['fit_family']}",
            f"PANEL_KEY={row['panel_key']}",
            f"GRID_TAG={row['grid_tag']}",
            f"REPLICATE={row['replicate']}",
            f"START_INDEX={row['start_index']}",
            "",
        ])
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
