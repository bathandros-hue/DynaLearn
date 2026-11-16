"""Synthetic demonstration or CSV-based mass-spring-damper identification."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .core import bootstrap_intervals, identify, simulate


def read_csv(path: Path) -> dict:
    columns = ["displacement", "velocity", "acceleration", "force"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError("CSV requires columns: " + ", ".join(columns))
        rows = list(reader)
    return {col: np.array([float(row[col]) for row in rows]) for col in columns}


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify damping and stiffness from SI measurements")
    parser.add_argument("--csv", type=Path, help="CSV with displacement, velocity, acceleration, force")
    parser.add_argument("--mass", type=float, default=2.0, help="Known mass [kg]")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export-demo", type=Path, help="Optional destination for generated synthetic CSV")
    args = parser.parse_args()
    if args.csv:
        data = read_csv(args.csv)
        source = str(args.csv)
    else:
        data = simulate(
            args.mass, damping=1.2, stiffness=25.0, duration=10.0, dt=0.02,
            force=lambda t: 3.0 * math.sin(1.8 * t) + math.sin(3.3 * t),
            x0=0.15, v0=0.0,
        )
        rng = np.random.default_rng(args.seed)
        data["acceleration"] = data["acceleration"] + rng.normal(0, 0.08, len(data["time"]))
        source = "synthetic forced oscillator (true c=1.2 Ns/m; k=25 N/m)"
        if args.export_demo:
            with args.export_demo.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", "displacement", "velocity", "acceleration", "force"])
                for row in zip(*(data[col] for col in
                                 ["time", "displacement", "velocity", "acceleration", "force"])):
                    writer.writerow(row)
    fit = identify(args.mass, **{k: data[k] for k in
                                 ("displacement", "velocity", "acceleration", "force")})
    intervals = bootstrap_intervals(
        args.mass, **{k: data[k] for k in
                      ("displacement", "velocity", "acceleration", "force")},
        resamples=args.bootstrap, seed=args.seed,
    )
    print(json.dumps({"source": source, "mass_kg": args.mass,
                      "fit": fit.as_dict(), "naive_bootstrap_95pct": intervals}, indent=2))


if __name__ == "__main__":
    main()
