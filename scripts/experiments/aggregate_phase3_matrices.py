from pathlib import Path
import argparse
import csv
import json

import numpy as np


def write_matrix_csv(path, matrix):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in matrix:
            writer.writerow([f"{float(value):.8f}" for value in row])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    root = Path(args.input_dir).resolve()

    result_data = json.loads((root / "results.json").read_text())
    seeds = [int(seed) for seed in result_data["arguments"]["seeds"]]
    raw_matrices = []
    for seed in seeds:
        raw_matrices.append(np.load(root / f"matrix_{seed}.npy"))
    raw = np.stack(raw_matrices, axis=0)
    normalized = raw / raw[:, :1, :1]
    normalized_mean = normalized.mean(axis=0)
    normalized_std = normalized.std(axis=0, ddof=1)
    raw_mean = raw.mean(axis=0)
    raw_std = raw.std(axis=0, ddof=1)

    mmd = np.array([
        result["forecaster_output_mmd_from_F0"]
        for result in result_data["results"]
    ], dtype=float)
    metrics = [result["test"] for result in result_data["results"]]
    metric_names = ["operating_cost", "objective", "terminal_soc_deviation"]
    test_summary = {}
    for name in metric_names:
        values = np.array([metric[name] for metric in metrics], dtype=float)
        test_summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }

    np.savez(
        root / "phase3_cross_evaluation_mean.npz",
        raw_matrices=raw,
        normalized_matrices=normalized,
        raw_mean=raw_mean,
        raw_std=raw_std,
        normalized_mean=normalized_mean,
        normalized_std=normalized_std,
        mmd_values=mmd,
        mmd_mean=mmd.mean(axis=0),
        mmd_std=mmd.std(axis=0, ddof=1),
        seeds=np.array(seeds),
    )
    write_matrix_csv(root / "normalized_mean.csv", normalized_mean)
    write_matrix_csv(root / "normalized_std.csv", normalized_std)

    summary = {
        "experiment": "phase3_corrected_cross_evaluation_five_seed_mean",
        "seeds": seeds,
        "normalized_mean": normalized_mean.tolist(),
        "normalized_std": normalized_std.tolist(),
        "mmd_mean": mmd.mean(axis=0).tolist(),
        "mmd_std": mmd.std(axis=0, ddof=1).tolist(),
        "test_summary": test_summary,
        "interpretation": {
            "diagonal": "Matched surrogate and forecaster states.",
            "superdiagonal": "Previous surrogate evaluated with the current forecaster.",
            "adaptation_gain": "For each i, compare E[i-1,i] against E[i,i].",
        },
    }
    (root / "phase3_cross_evaluation_mean_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Use the existing paper plotting script by writing its expected input names.
    np.savez(
        root / "phase3_cross_evaluation_matrix.npz",
        normalized_operating_cost=normalized_mean,
    )
    plot_summary = {
        "forecaster_output_mmd_from_F0": mmd.mean(axis=0).tolist(),
    }
    (root / "phase3_cross_evaluation_summary.json").write_text(
        json.dumps(plot_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
