#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_cross_evaluation import plot_cross_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', default='figures')
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 17, 27, 37, 47])
    parser.add_argument('--output-dir', default='figures/phase3_cross_evaluation_mean5')
    args = parser.parse_args()

    matrices = []
    raw_matrices = []
    baselines = []
    summaries = []
    for seed in args.seeds:
        directory = Path(args.input_root) / f'phase3_cross_evaluation_seed{seed}'
        data = np.load(directory / 'phase3_cross_evaluation_matrix.npz')
        summary = json.loads((directory / 'phase3_cross_evaluation_summary.json').read_text(encoding='utf-8'))
        matrices.append(data['normalized_operating_cost'])
        raw_matrices.append(data['raw_operating_cost'])
        baselines.append(summary['baseline_operating_cost'])
        summaries.append(summary)

    normalized = np.stack(matrices)
    raw = np.stack(raw_matrices)
    mean_normalized = normalized.mean(axis=0)
    std_normalized = normalized.std(axis=0, ddof=1)
    mean_raw = raw.mean(axis=0)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plot_cross_matrix(mean_normalized, output / 'phase3_cross_evaluation_mean5.png')
    plot_cross_matrix(mean_normalized, output / 'phase3_cross_evaluation_mean5.svg')
    np.savez(
        output / 'phase3_cross_evaluation_mean5.npz',
        normalized_operating_cost_mean=mean_normalized,
        normalized_operating_cost_std=std_normalized,
        raw_operating_cost_mean=mean_raw,
        normalized_operating_cost_by_seed=normalized,
        raw_operating_cost_by_seed=raw,
    )

    with (output / 'phase3_cross_evaluation_mean5.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['surrogate_state', 'forecast_state', 'mean_normalized_operating_cost', 'std_normalized_operating_cost', 'mean_operating_cost'])
        for i in range(mean_normalized.shape[0]):
            for j in range(mean_normalized.shape[1]):
                writer.writerow([f'S{i}', f'F{j}', f'{mean_normalized[i,j]:.10f}', f'{std_normalized[i,j]:.10f}', f'{mean_raw[i,j]:.10f}'])

    summary = {
        'experiment': 'phase3_cross_evaluation_iterative_adaptation_mean_over_random_seeds',
        'seeds': args.seeds,
        'num_seeds': len(args.seeds),
        'definition': 'The reported matrix is the element-wise mean of each seed-specific normalized operating-cost matrix; each seed uses E[0,0] as its own common baseline.',
        'mean_baseline_operating_cost': float(np.mean(baselines)),
        'std_baseline_operating_cost': float(np.std(baselines, ddof=1)),
        'mean_normalized_operating_cost': mean_normalized.tolist(),
        'std_normalized_operating_cost': std_normalized.tolist(),
        'diagonal_mean': np.diag(mean_normalized).tolist(),
        'diagonal_std': np.diag(std_normalized).tolist(),
        'fixed_S0_last_mean_relative_change': float(mean_normalized[0, -1] - mean_normalized[0, 0]),
        'fixed_S0_last_std': float(std_normalized[0, -1]),
        'matched_last_mean_relative_change': float(mean_normalized[-1, -1] - mean_normalized[0, 0]),
        'matched_last_std': float(std_normalized[-1, -1]),
        'outputs': {
            'png': str(output / 'phase3_cross_evaluation_mean5.png'),
            'svg': str(output / 'phase3_cross_evaluation_mean5.svg'),
            'csv': str(output / 'phase3_cross_evaluation_mean5.csv'),
            'npz': str(output / 'phase3_cross_evaluation_mean5.npz'),
        },
    }
    (output / 'phase3_cross_evaluation_mean5_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('mean normalized matrix:')
    for row in mean_normalized:
        print(' '.join(f'{value:.4f}' for value in row))
    print(f"fixed_S0_last_mean_relative_change={summary['fixed_S0_last_mean_relative_change']:+.4%}")
    print(f"matched_last_mean_relative_change={summary['matched_last_mean_relative_change']:+.4%}")
    print(f"fixed_S0_last_std={summary['fixed_S0_last_std']:.4f}")
    print(f"matched_last_std={summary['matched_last_std']:.4f}")
    print(f'outputs={output.resolve()}')


if __name__ == '__main__':
    main()
