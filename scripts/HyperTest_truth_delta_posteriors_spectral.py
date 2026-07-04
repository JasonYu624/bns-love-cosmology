#!/usr/bin/env python3
"""Create truth-delta pseudo-posteriors for the spectral-siren-only HyperPE run.

Each event gets a CSV with N identical rows at the injection truth.  The file
names intentionally match '*_reweighted_posterior_augmented.csv' so they can be
used directly with the existing HyperPE_spectral_SEOBNR.py --posterior-glob.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build spectral-only truth-delta pseudo-posteriors.")
    p.add_argument(
        "--pop-outdir",
        type=str,
        default="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_population_SEOBNR",
        help="Population directory containing event_XXXX/meta.json and/or accepted.jsonl.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_truth_delta_spectral_SEOBNR",
        help="Output directory for truth pseudo-posterior CSV files.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=2,
        help="Number of identical truth samples per event.",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum number of sorted events to write. Use <=0 for all available events.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files in the output directory.",
    )
    return p.parse_args()


def nested_get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def as_float(value: Any, name: str, source: str) -> float:
    if value is None:
        raise KeyError(f"Could not find {name} in {source}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite {name}={value!r} in {source}")
    return out


def truth_from_meta(meta_path: str) -> dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    inj = dict(meta.get("injection_parameters", {}))

    m1_det = first_present(meta, ["mass_1_detector", "m1_det", "m1_detector"])
    m2_det = first_present(meta, ["mass_2_detector", "m2_det", "m2_detector"])
    if m1_det is None:
        m1_src = first_present(inj, ["mass_1_source", "mass_1", "m1_source", "m1"])
        z = first_present(inj, ["redshift", "z", "redshift_sample"])
        if m1_src is not None and z is not None:
            m1_det = float(m1_src) * (1.0 + float(z))
    if m2_det is None:
        m2_src = first_present(inj, ["mass_2_source", "mass_2", "m2_source", "m2"])
        z = first_present(inj, ["redshift", "z", "redshift_sample"])
        if m2_src is not None and z is not None:
            m2_det = float(m2_src) * (1.0 + float(z))

    d_l = first_present(inj, ["luminosity_distance", "dL_mpc", "d_luminosity", "distance"])
    if d_l is None:
        d_l = first_present(meta, ["luminosity_distance", "dL_mpc", "d_luminosity", "distance"])

    event_name = Path(meta_path).parent.name
    return {
        "event_name": event_name,
        "source": meta_path,
        "mass_1_detector": as_float(m1_det, "mass_1_detector", meta_path),
        "mass_2_detector": as_float(m2_det, "mass_2_detector", meta_path),
        "luminosity_distance": as_float(d_l, "luminosity_distance", meta_path),
    }


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def truth_from_accepted(pop_outdir: str) -> list[dict[str, Any]]:
    accepted_path = os.path.join(pop_outdir, "accepted.jsonl")
    rows = read_jsonl(accepted_path)
    out = []
    for i, row in enumerate(rows, start=1):
        inj = dict(row.get("injection_parameters", row))
        event_index = row.get("event_index", row.get("event_id", i))
        if isinstance(event_index, str) and event_index.startswith("event_"):
            event_name = event_index
        else:
            event_name = f"event_{int(event_index):04d}"

        m1_det = first_present(row, ["mass_1_detector", "m1_det", "m1_detector"])
        m2_det = first_present(row, ["mass_2_detector", "m2_det", "m2_detector"])
        if m1_det is None:
            m1_det = first_present(inj, ["mass_1_detector", "m1_det", "m1_detector"])
        if m2_det is None:
            m2_det = first_present(inj, ["mass_2_detector", "m2_det", "m2_detector"])
        if m1_det is None:
            m1_src = first_present(inj, ["mass_1_source", "mass_1", "m1_source", "m1"])
            z = first_present(inj, ["redshift", "z", "redshift_sample"])
            if m1_src is not None and z is not None:
                m1_det = float(m1_src) * (1.0 + float(z))
        if m2_det is None:
            m2_src = first_present(inj, ["mass_2_source", "mass_2", "m2_source", "m2"])
            z = first_present(inj, ["redshift", "z", "redshift_sample"])
            if m2_src is not None and z is not None:
                m2_det = float(m2_src) * (1.0 + float(z))

        d_l = first_present(inj, ["luminosity_distance", "dL_mpc", "d_luminosity", "distance"])
        if d_l is None:
            d_l = first_present(row, ["luminosity_distance", "dL_mpc", "d_luminosity", "distance"])

        out.append(
            {
                "event_name": event_name,
                "source": accepted_path,
                "mass_1_detector": as_float(m1_det, "mass_1_detector", accepted_path),
                "mass_2_detector": as_float(m2_det, "mass_2_detector", accepted_path),
                "luminosity_distance": as_float(d_l, "luminosity_distance", accepted_path),
            }
        )
    return out


def load_truths(pop_outdir: str) -> list[dict[str, Any]]:
    meta_paths = sorted(glob.glob(os.path.join(pop_outdir, "event_*", "meta.json")))
    if meta_paths:
        return [truth_from_meta(p) for p in meta_paths]

    accepted_path = os.path.join(pop_outdir, "accepted.jsonl")
    if os.path.exists(accepted_path):
        return truth_from_accepted(pop_outdir)

    raise FileNotFoundError(
        f"Could not find event_*/meta.json or accepted.jsonl under {pop_outdir}"
    )


def main() -> None:
    args = get_args()
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")

    truths = load_truths(args.pop_outdir)
    truths = sorted(truths, key=lambda x: x["event_name"])
    if args.max_events and args.max_events > 0:
        truths = truths[: args.max_events]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for truth in truths:
        d_l = float(truth["luminosity_distance"])
        row = {
            "mass_1_detector": float(truth["mass_1_detector"]),
            "mass_2_detector": float(truth["mass_2_detector"]),
            "luminosity_distance": d_l,
            # The current spectral loader expects delta_a* columns and then
            # divides the prior by the corresponding Gaussian factor.  At truth
            # delta_a*=0, that factor is 1, so the retained prior remains dL^2.
            "delta_a0": 0.0,
            "delta_a1": 0.0,
            "delta_a2": 0.0,
            "prior": d_l**2,
        }
        df = pd.DataFrame([row] * args.n_samples)
        path = outdir / f"truth_{truth['event_name']}_reweighted_posterior_augmented.csv"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        df.to_csv(path, index=False)
        written.append(str(path))

    summary = {
        "mode": "spectral_truth_delta",
        "pop_outdir": os.path.abspath(args.pop_outdir),
        "outdir": str(outdir.resolve()),
        "n_events": len(truths),
        "n_identical_samples_per_event": int(args.n_samples),
        "posterior_glob": str(outdir.resolve() / "*_reweighted_posterior_augmented.csv"),
        "notes": [
            "Each event posterior contains identical truth samples.",
            "The prior column is dL^2 before the spectral loader removes the delta_a* Gaussian factor.",
            "Event-level MC variance diagnostics are not physically meaningful for truth-delta posteriors.",
        ],
        "files": written,
        "truths": truths,
    }
    summary_path = outdir / "truth_delta_spectral_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote {len(written)} spectral truth-delta posterior files to {outdir}")
    print(f"Posterior glob: {summary['posterior_glob']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
